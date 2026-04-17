"""FastAPI agent server — runs on CPU pod (``launch_swe_agent_workers.yaml``).

Exposes one rollout endpoint ``/generate_trajectory`` that:
  1. Allocates a fresh SWE-Bench Docker container for the given instance
  2. Runs the full mini-swe-agent loop in-process (tokenizer + SGLang + docker)
  3. Runs the SWE-Bench eval harness on the resulting patch (fresh container)
  4. Returns per-turn token ids / loss_mask / logprobs to the trainer

Starts dockerd + preloads tokenizer at startup. Trainer talks HTTP to this
server, and SGLang router is reached via ``sglang_router_url`` supplied per
request.

Launch::

    uvicorn swe_agent_server:app --host 0.0.0.0 --port 5000

Env vars honored:
  MODEL_PATH              — HF path used to load the tokenizer
  SWE_CONFIG_PATH         — override for swebench.yaml (default: aws_kiro_infra/swebench.yaml)
  SWE_DOCKER_IMAGES_PATH  — directory of {instance_id}.tar.gz for offline load
  SWE_MAX_CONCURRENT      — concurrency cap for /generate_trajectory
  SWE_CONTAINER_MEMORY, SWE_CONTAINER_PIDS_LIMIT — per-container limits
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import mini_swe_agent, patch_utils
from .docker_ops import DockerManager, ensure_docker_daemon, load_image_from_tar
from .sglang_client import SGLangClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("swe_agent_server")


# =============================================================================
# Request / response schema
# =============================================================================

class GenerateRequest(BaseModel):
    instance_id: str
    problem_statement: str
    instance: dict[str, Any]
    data_source: str = "swe-gym"
    sampling_params: dict[str, Any] = Field(default_factory=dict)
    sglang_router_url: str
    max_context_len: int = 32768
    max_new_tokens: int = 4096
    step_limit: int = 30
    skip_eval: bool = False
    eval_timeout: int = 300
    exec_timeout: int = 180
    strict_no_test: bool | None = None
    strict_no_config: bool | None = None
    test_patch_policy_scope: str | None = None
    request_id: str | None = None


class GenerateResponse(BaseModel):
    status: str  # "success" | "aborted" | "error"
    error: str | None = None
    messages: list[dict[str, Any]] = Field(default_factory=list)
    step_debug: list[dict[str, Any]] = Field(default_factory=list)
    git_patch: str | None = None
    patch_source: str | None = None
    exit_status: str | None = None
    n_steps: int = 0
    reward: int = 0
    eval_result: dict[str, Any] | None = None
    policy: dict[str, Any] | None = None
    elapsed_seconds: float = 0.0


# =============================================================================
# Global state
# =============================================================================

_docker = DockerManager()
_semaphore: asyncio.Semaphore | None = None
_tokenizer = None
_sweagent_config: dict[str, Any] | None = None
_grading_executor = concurrent.futures.ThreadPoolExecutor(max_workers=16)
_swe_docker_images_dir: str | None = None


def _load_tokenizer():
    """Load HF tokenizer from MODEL_PATH. Cached at process startup."""
    model_path = os.getenv("MODEL_PATH") or os.getenv("DEFAULT_MODEL_PATH")
    if not model_path:
        raise RuntimeError("MODEL_PATH (or DEFAULT_MODEL_PATH) env var must be set")
    from transformers import AutoTokenizer
    logger.info("Loading tokenizer from %s", model_path)
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    logger.info("Tokenizer ready (vocab=%d, eos=%s)", tok.vocab_size, tok.eos_token_id)
    return tok


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _semaphore, _tokenizer, _sweagent_config, _swe_docker_images_dir

    max_concurrent = int(os.getenv("SWE_MAX_CONCURRENT", "16"))
    _semaphore = asyncio.Semaphore(max_concurrent)
    logger.info("Concurrency cap: %d", max_concurrent)

    _swe_docker_images_dir = os.getenv("SWE_DOCKER_IMAGES_PATH", "") or None

    if not ensure_docker_daemon():
        raise RuntimeError("Docker daemon failed to start; SWE tasks cannot run")

    _tokenizer = _load_tokenizer()
    _sweagent_config = mini_swe_agent.get_sweagent_config()
    logger.info("swebench.yaml loaded (agent=%s)", bool(_sweagent_config.get("agent")))

    yield

    logger.info("Shutting down; cleaning up containers...")
    await _docker.cleanup_all()


app = FastAPI(lifespan=lifespan)


# =============================================================================
# Endpoints
# =============================================================================

@app.get("/healthz")
async def healthz():
    docker_ok = True
    try:
        import subprocess
        r = subprocess.run(
            ["docker", "info", "--format", "{{.ContainersRunning}}"],
            capture_output=True, text=True, timeout=5,
        )
        docker_ok = r.returncode == 0
        running = r.stdout.strip() if docker_ok else "?"
    except Exception:
        docker_ok = False
        running = "?"
    return {
        "ok": True,
        "docker_ok": docker_ok,
        "containers_running": running,
        "tokenizer_loaded": _tokenizer is not None,
        "concurrent_slots_free": _semaphore._value if _semaphore else 0,
    }


@app.post("/generate_trajectory", response_model=GenerateResponse)
async def generate_trajectory(req: GenerateRequest) -> GenerateResponse:
    assert _semaphore is not None and _tokenizer is not None and _sweagent_config is not None
    t_start = time.time()
    iid = req.instance_id
    log_tag = f"[{iid}]"
    if req.request_id:
        log_tag = f"[{iid}|{req.request_id[:8]}]"
    logger.info("%s ENTER data_source=%s", log_tag, req.data_source)

    # Image resolution + (optional) offline tarball load
    try:
        image_name = mini_swe_agent.get_docker_image_name(req.instance, req.data_source)
    except Exception as e:
        return GenerateResponse(status="error", error=f"image resolution failed: {e}")
    if _swe_docker_images_dir:
        tar_path = str(Path(_swe_docker_images_dir) / f"{iid}.tar.gz")
        if os.path.isfile(tar_path):
            load_image_from_tar(tar_path)

    # Detect eval strategy early — needed for container setup (R2E needs symlinks)
    strategy = patch_utils.detect_eval_strategy(req.instance, req.data_source)
    logger.info("%s eval strategy: %s", log_tag, strategy)

    async with _semaphore:
        agent_container_id: str | None = None
        eval_container_id: str | None = None
        run_info: dict[str, Any] = {
            "messages": [], "step_debug": [], "git_patch": None,
            "patch_source": None, "exit_status": None, "n_steps": 0,
            "error": None, "eval_result": None, "policy": None,
        }
        try:
            # --- Step 1: allocate agent container ---
            logger.info("%s Step 1/4: docker create %s", log_tag, image_name)
            agent_c = await _docker.create(image=image_name)
            agent_container_id = agent_c.id

            # R2E-Gym images need extra setup (symlink r2e_tests, etc.)
            if strategy == "r2e":
                await _docker.setup_r2e_container(agent_container_id)

            # --- Step 2: agent loop ---
            logger.info("%s Step 2/4: agent loop (step_limit=%d)", log_tag, req.step_limit)
            sglang = SGLangClient(router_url=req.sglang_router_url)
            try:
                loop_result = await mini_swe_agent.run_agent_loop(
                    docker=_docker,
                    container_id=agent_container_id,
                    sglang=sglang,
                    tokenizer=_tokenizer,
                    instance=req.instance,
                    sweagent_config=_sweagent_config,
                    sampling_params=req.sampling_params,
                    step_limit=req.step_limit,
                    max_context_len=req.max_context_len,
                    max_new_tokens=req.max_new_tokens,
                    exec_timeout=req.exec_timeout,
                )
            finally:
                await sglang.close()
            run_info.update(loop_result)
            git_patch = run_info.get("git_patch")

            # --- Step 3: policy gate ---
            if git_patch:
                policy = patch_utils.analyze_patch_policy(
                    git_patch,
                    instance=req.instance,
                    strict_no_test=req.strict_no_test,
                    strict_no_config=req.strict_no_config,
                    scope=req.test_patch_policy_scope,
                )
                run_info["policy"] = policy
                if policy.get("violated"):
                    logger.warning(
                        "%s Step 3/4: policy blocked (%s)",
                        log_tag, ",".join(policy.get("reasons", [])),
                    )
                    run_info["eval_result"] = {
                        "ok": True,
                        "resolved": False,
                        "resolved_by": "policy_blocked",
                        "policy_blocked": True,
                        "policy": policy,
                    }
                    git_patch = None  # don't run eval

            # --- Step 4: eval harness (dispatch by strategy detected earlier) ---
            if git_patch and not req.skip_eval:

                if strategy == "swe_harness":
                    # SWE-Gym / SWE-Bench: pre-rendered eval_script + harness grading
                    eval_script = await asyncio.get_event_loop().run_in_executor(
                        _grading_executor, patch_utils.resolve_eval_script, req.instance,
                    )
                    if not eval_script:
                        run_info["error"] = "eval_script unavailable"
                    else:
                        # Release tainted agent container, use fresh one for eval
                        if agent_container_id is not None:
                            try:
                                await _docker.destroy(agent_container_id)
                            except Exception:
                                logger.exception("%s destroy agent container failed", log_tag)
                            agent_container_id = None
                        eval_c = await _docker.create(image=image_name)
                        eval_container_id = eval_c.id
                        eval_raw = await _docker.evaluate(
                            container_id=eval_container_id,
                            patch=git_patch,
                            eval_script=eval_script,
                            timeout=req.eval_timeout,
                        )
                        resolved_by_returncode = bool(eval_raw.get("resolved", False))
                        resolved = resolved_by_returncode
                        grading_error = None
                        grading_report = None
                        try:
                            graded = await asyncio.get_event_loop().run_in_executor(
                                _grading_executor,
                                patch_utils.grade_eval_output,
                                req.instance,
                                git_patch,
                                str(eval_raw.get("apply_output", "")),
                                str(eval_raw.get("output", "")),
                            )
                            grading_report = graded.get("report")
                            resolved = bool(graded.get("resolved", False))
                        except Exception as ge:
                            grading_error = str(ge)
                            logger.warning("%s harness grading failed: %s", log_tag, ge)

                        run_info["eval_result"] = {
                            **eval_raw,
                            "strategy": "swe_harness",
                            "resolved_by_returncode": resolved_by_returncode,
                            "resolved": resolved,
                            "resolved_by": (
                                "harness_get_eval_report" if grading_report is not None
                                else "returncode_fallback"
                            ),
                            "grading_report": grading_report,
                            "grading_error": grading_error,
                        }
                        logger.info(
                            "%s Step 4/4: [swe_harness] resolved=%s",
                            log_tag, resolved,
                        )

                elif strategy == "r2e":
                    # R2E-Gym native: image has /run_tests.sh; compare against expected_output_json
                    if not req.instance.get("expected_output_json"):
                        run_info["error"] = "expected_output_json missing (required for r2e strategy)"
                    else:
                        # Release tainted agent container, use fresh one for eval
                        if agent_container_id is not None:
                            try:
                                await _docker.destroy(agent_container_id)
                            except Exception:
                                logger.exception("%s destroy agent container failed", log_tag)
                            agent_container_id = None
                        eval_c = await _docker.create(image=image_name)
                        eval_container_id = eval_c.id
                        await _docker.setup_r2e_container(eval_container_id)
                        eval_raw = await _docker.evaluate_r2e(
                            container_id=eval_container_id,
                            patch=git_patch,
                            timeout=req.eval_timeout,
                        )
                        grading_error = None
                        grading_report = None
                        resolved = False
                        try:
                            graded = await asyncio.get_event_loop().run_in_executor(
                                _grading_executor,
                                patch_utils.grade_eval_output_r2e,
                                req.instance,
                                str(eval_raw.get("output", "")),
                            )
                            grading_report = graded.get("report")
                            resolved = bool(graded.get("resolved", False))
                        except Exception as ge:
                            grading_error = str(ge)
                            logger.warning("%s r2e grading failed: %s", log_tag, ge)

                        run_info["eval_result"] = {
                            **eval_raw,
                            "strategy": "r2e",
                            "resolved": resolved,
                            "resolved_by": (
                                "r2e_expected_output_json" if grading_report is not None
                                else "returncode_fallback"
                            ),
                            "grading_report": grading_report,
                            "grading_error": grading_error,
                        }
                        logger.info(
                            "%s Step 4/4: [r2e] resolved=%s n_tests=%s",
                            log_tag, resolved,
                            (grading_report or {}).get("n_tests"),
                        )
                else:
                    run_info["error"] = f"unknown eval strategy: {strategy}"

        except Exception as e:
            run_info["error"] = str(e)
            logger.exception("%s ERROR: %s", log_tag, e)
        finally:
            # Ensure containers released (including eval if something raised)
            for cid in (agent_container_id, eval_container_id):
                if cid is not None:
                    try:
                        await _docker.destroy(cid)
                    except Exception:
                        logger.warning("%s destroy container %s failed", log_tag, cid[:12])

    elapsed = time.time() - t_start
    eval_result = run_info.get("eval_result") or {}
    resolved = bool(eval_result.get("resolved", False))
    reward = 1 if resolved else 0

    status = "success"
    if run_info.get("error") and not run_info.get("messages"):
        status = "error"
    elif not run_info.get("messages"):
        status = "aborted"

    logger.info(
        "%s DONE status=%s exit=%s steps=%d reward=%d elapsed=%.1fs",
        log_tag, status, run_info.get("exit_status"), run_info.get("n_steps", 0),
        reward, elapsed,
    )

    return GenerateResponse(
        status=status,
        error=run_info.get("error"),
        messages=run_info.get("messages", []),
        step_debug=run_info.get("step_debug", []),
        git_patch=run_info.get("git_patch"),
        patch_source=run_info.get("patch_source"),
        exit_status=run_info.get("exit_status"),
        n_steps=run_info.get("n_steps", 0),
        reward=reward,
        eval_result=run_info.get("eval_result"),
        policy=run_info.get("policy"),
        elapsed_seconds=elapsed,
    )
