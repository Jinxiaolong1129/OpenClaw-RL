"""Inline Docker CLI wrapper for the CPU agent pod.

The agent loop and Docker management run in the same process (no separate
HTTP server). These helpers call ``docker`` via subprocess — same semantics
as ``swe-rl/server/swe_exec_server.py`` but without the Flask layer.
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("swe_agent_server.docker_ops")

DEFAULT_CONTAINER_PIDS_LIMIT = os.getenv("SWE_CONTAINER_PIDS_LIMIT", "1024")
DEFAULT_CONTAINER_MEMORY = os.getenv("SWE_CONTAINER_MEMORY", "8g")
DEFAULT_CWD = "/testbed"
DEFAULT_EXEC_TIMEOUT = 180
EVAL_OUTPUT_MAX_CHARS = int(os.getenv("SWE_EVAL_OUTPUT_MAX_CHARS", "20000000"))


def _run_docker_sync(*args: str, timeout: int = 300) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["docker", *args],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


async def _run_docker(*args: str, timeout: int = 300) -> subprocess.CompletedProcess:
    """Async wrapper — spawns docker CLI in a thread so we don't block the event loop."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: _run_docker_sync(*args, timeout=timeout))


def ensure_docker_daemon() -> bool:
    """Start dockerd if not running. CPU pods are privileged so DinD is allowed."""
    r = _run_docker_sync("info", "--format", "{{.ContainersRunning}}", timeout=10)
    if r.returncode == 0:
        logger.info("Docker daemon already running.")
        return True

    logger.warning("Docker daemon not reachable; starting dockerd...")
    for attempt in (1, 2, 3):
        # --data-root=/dev/shm/docker_data avoids overflowing the container FS
        subprocess.Popen(
            ["dockerd", "--data-root=/dev/shm/docker_data"],
            stdout=open("/tmp/dockerd.log", "ab"),
            stderr=subprocess.STDOUT,
        )
        for _ in range(30):
            time.sleep(1)
            r = _run_docker_sync("info", "--format", "{{.ContainersRunning}}", timeout=10)
            if r.returncode == 0:
                logger.info("Docker daemon started on attempt %d.", attempt)
                return True
        logger.warning("dockerd startup attempt %d failed; retrying.", attempt)
        time.sleep(2)
    logger.error("dockerd failed to start. Check /tmp/dockerd.log")
    return False


def load_image_from_tar(tar_path: str) -> bool:
    """``docker load -i <path>`` if a tarball exists for this instance."""
    if not tar_path or not os.path.isfile(tar_path):
        return False
    r = _run_docker_sync("load", "-i", tar_path, timeout=600)
    if r.returncode != 0:
        logger.warning("docker load %s failed: %s", tar_path, r.stderr.strip()[:500])
        return False
    logger.info("Loaded image from %s", tar_path)
    return True


def _clip_eval_output(text: str) -> tuple[str, bool]:
    if EVAL_OUTPUT_MAX_CHARS <= 0 or len(text) <= EVAL_OUTPUT_MAX_CHARS:
        return text, False
    return text[-EVAL_OUTPUT_MAX_CHARS:], True


@dataclass
class Container:
    id: str
    name: str
    image: str
    created_at: float = field(default_factory=time.time)


class DockerManager:
    """Track active containers for debug / cleanup on shutdown."""

    def __init__(self):
        self._active: dict[str, Container] = {}
        self._lock = asyncio.Lock()

    async def create(self, image: str, cwd: str = DEFAULT_CWD, timeout: int = 120) -> Container:
        name = f"swe-{uuid.uuid4().hex[:12]}"
        r = await _run_docker(
            "run", "-d",
            "--init",
            "--name", name,
            "--pull", "never",
            "-w", cwd,
            "--pids-limit", DEFAULT_CONTAINER_PIDS_LIMIT,
            "--memory", DEFAULT_CONTAINER_MEMORY,
            image,
            "sleep", "infinity",
            timeout=timeout,
        )
        if r.returncode != 0:
            raise RuntimeError(f"docker run failed for {image}: {r.stderr.strip()}")
        cid = r.stdout.strip()
        c = Container(id=cid, name=name, image=image)
        async with self._lock:
            self._active[cid] = c
        logger.info("[docker] created %s (%s) from %s", cid[:12], name, image)
        return c

    async def exec(
        self,
        container_id: str,
        command: str,
        cwd: str = DEFAULT_CWD,
        timeout: int = DEFAULT_EXEC_TIMEOUT,
        env: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        env_args: list[str] = []
        for k, v in (env or {}).items():
            env_args.extend(["-e", f"{k}={v}"])
        try:
            r = await _run_docker(
                "exec", "-w", cwd, *env_args, container_id,
                "bash", "-lc", command,
                timeout=timeout,
            )
            return {
                "ok": True,
                "returncode": r.returncode,
                "output": r.stdout + r.stderr,
            }
        except subprocess.TimeoutExpired:
            # Best-effort kill of child processes inside the container
            try:
                await _run_docker("exec", container_id, "kill", "-9", "-1", timeout=10)
            except Exception:
                pass
            return {
                "ok": True,
                "returncode": -1,
                "output": f"Command timed out after {timeout}s",
            }

    async def diff(self, container_id: str, cwd: str = DEFAULT_CWD) -> str:
        r = await _run_docker(
            "exec", "-w", cwd, container_id,
            "bash", "-lc", "git add -A && git diff --cached",
            timeout=60,
        )
        if r.returncode != 0:
            logger.warning("[docker] diff returncode=%s: %s", r.returncode, r.stderr[:200])
        return r.stdout

    async def destroy(self, container_id: str) -> None:
        await _run_docker("rm", "-f", container_id, timeout=30)
        async with self._lock:
            self._active.pop(container_id, None)
        logger.info("[docker] destroyed %s", container_id[:12])

    async def evaluate(
        self,
        container_id: str,
        patch: str,
        eval_script: str,
        cwd: str = DEFAULT_CWD,
        timeout: int = 3600,
    ) -> dict[str, Any]:
        """Apply patch to a fresh HEAD then run eval_script. Mirrors
        swe_exec_server:/container/evaluate semantics."""
        delimiter = f"PATCH_{uuid.uuid4().hex}"
        apply_cmd = (
            f"git reset --hard HEAD && git clean -fd && "
            f"git apply <<'{delimiter}'\n{patch}\n{delimiter}"
        )
        r_apply = await _run_docker(
            "exec", "-w", cwd, container_id,
            "bash", "-lc", apply_cmd,
            timeout=60,
        )
        if r_apply.returncode != 0:
            return {
                "ok": True,
                "resolved": False,
                "returncode": -1,
                "apply_returncode": r_apply.returncode,
                "apply_output": r_apply.stdout + r_apply.stderr,
                "output": "",
                "error": f"git apply failed: {r_apply.stderr[:2000]}",
            }

        eval_delim = f"EVAL_{uuid.uuid4().hex}"
        eval_cmd = f"bash <<'{eval_delim}'\n{eval_script}\n{eval_delim}"
        try:
            r_eval = await _run_docker(
                "exec", "-w", cwd, container_id,
                "bash", "-lc", eval_cmd,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return {
                "ok": True,
                "resolved": False,
                "returncode": -1,
                "apply_returncode": r_apply.returncode,
                "apply_output": r_apply.stdout + r_apply.stderr,
                "output": f"Eval timed out after {timeout}s",
                "error": "eval_timeout",
            }

        eval_output, truncated = _clip_eval_output(r_eval.stdout + r_eval.stderr)
        return {
            "ok": True,
            "resolved": r_eval.returncode == 0,
            "returncode": r_eval.returncode,
            "apply_returncode": r_apply.returncode,
            "apply_output": r_apply.stdout + r_apply.stderr,
            "output": eval_output,
            "output_truncated": truncated,
        }

    async def setup_r2e_container(self, container_id: str, cwd: str = DEFAULT_CWD) -> None:
        """Run R2E-Gym-specific container setup (mirrors ``r2egym setup_env()``).

        R2E images bake test files at ``/r2e_tests`` and ``run_tests.sh`` at
        ``/testbed/run_tests.sh``. The official runtime moves them to ``/root/``
        and creates symlinks so the agent can't cheat. We replicate the
        essential parts: make ``r2e_tests`` visible from ``/testbed`` and
        protect ``run_tests.sh`` from accidental deletion.
        """
        setup_cmd = (
            "set -e; "
            # Symlink /r2e_tests into /testbed so `pytest r2e_tests` works from cwd
            "if [ -d /r2e_tests ] && [ ! -e /testbed/r2e_tests ]; then "
            "  ln -s /r2e_tests /testbed/r2e_tests; fi; "
            # Backup run_tests.sh to /root in case git clean later
            "if [ -f /testbed/run_tests.sh ]; then "
            "  cp /testbed/run_tests.sh /root/run_tests.sh.bak; fi; "
            "echo r2e_setup_done"
        )
        await _run_docker(
            "exec", "-w", cwd, container_id,
            "bash", "-lc", setup_cmd,
            timeout=30,
        )
        logger.info("[docker] R2E container setup done for %s", container_id[:12])

    async def evaluate_r2e(
        self,
        container_id: str,
        patch: str,
        cwd: str = DEFAULT_CWD,
        timeout: int = 300,
        run_tests_cmd: str = "bash /run_tests.sh",
    ) -> dict[str, Any]:
        """R2E-Gym eval: apply patch + run image's built-in ``/run_tests.sh``.

        R2E-Gym docker images ship with ``run_tests.sh`` pre-baked (usually
        at ``/testbed/run_tests.sh``). The official runtime moves it to
        ``/root/`` during ``setup_env()``, but we keep it in place and
        protect it from ``git clean`` via exclusion flags.

        Call :meth:`setup_r2e_container` on the eval container first if it
        hasn't been set up yet (the agent container should already have been
        set up before the agent loop).

        Returns: same shape as :meth:`evaluate`.
        """
        # ★ R2E images have non-git-tracked files (run_tests.sh, r2e_tests/)
        # that must survive the reset. Exclude them from git clean.
        delimiter = f"PATCH_{uuid.uuid4().hex}"
        apply_cmd = (
            f"git reset --hard HEAD && "
            f"git clean -fd -e run_tests.sh -e r2e_tests -e r2e_tests/ && "
            f"git apply <<'{delimiter}'\n{patch}\n{delimiter}"
        )
        r_apply = await _run_docker(
            "exec", "-w", cwd, container_id,
            "bash", "-lc", apply_cmd,
            timeout=60,
        )
        if r_apply.returncode != 0:
            return {
                "ok": True,
                "resolved": False,
                "returncode": -1,
                "apply_returncode": r_apply.returncode,
                "apply_output": r_apply.stdout + r_apply.stderr,
                "output": "",
                "error": f"git apply failed: {r_apply.stderr[:2000]}",
            }
        # Some R2E images place run_tests.sh at different paths. Mirror r2egym's
        # runtime fallback behavior before executing tests.
        detect_cmd = (
            "for p in /run_tests.sh /testbed/run_tests.sh /root/run_tests.sh; do "
            "[ -f \"$p\" ] && echo \"$p\" && exit 0; "
            "done; exit 1"
        )
        r_detect = await _run_docker(
            "exec", "-w", cwd, container_id,
            "bash", "-lc", detect_cmd,
            timeout=30,
        )
        if r_detect.returncode != 0 or not r_detect.stdout.strip():
            return {
                "ok": True,
                "resolved": False,
                "returncode": -1,
                "apply_returncode": r_apply.returncode,
                "apply_output": r_apply.stdout + r_apply.stderr,
                "output": "",
                "error": (
                    "run_tests.sh not found in known paths: "
                    "/run_tests.sh, /testbed/run_tests.sh, /root/run_tests.sh"
                ),
            }
        run_tests_path = r_detect.stdout.strip().splitlines()[-1]
        eval_cmd = f"bash {run_tests_path}" if run_tests_cmd == "bash /run_tests.sh" else run_tests_cmd
        try:
            r_eval = await _run_docker(
                "exec", "-w", cwd, container_id,
                "bash", "-lc", eval_cmd,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return {
                "ok": True,
                "resolved": False,
                "returncode": -1,
                "apply_returncode": r_apply.returncode,
                "apply_output": r_apply.stdout + r_apply.stderr,
                "output": f"Eval timed out after {timeout}s",
                "error": "eval_timeout",
            }

        eval_output, truncated = _clip_eval_output(r_eval.stdout + r_eval.stderr)
        return {
            "ok": True,
            # NOTE: returncode==0 only means script finished; real grading
            # happens in patch_utils.grade_eval_output_r2e by parsing `output`
            "resolved": r_eval.returncode == 0,
            "returncode": r_eval.returncode,
            "apply_returncode": r_apply.returncode,
            "apply_output": r_apply.stdout + r_apply.stderr,
            "output": eval_output,
            "output_truncated": truncated,
        }

    async def cleanup_all(self) -> None:
        async with self._lock:
            containers = list(self._active.keys())
            self._active.clear()
        for cid in containers:
            try:
                await _run_docker("rm", "-f", cid, timeout=30)
            except Exception:
                pass
