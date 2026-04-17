#!/usr/bin/env python3
"""Local end-to-end test for R2E-Gym eval strategy.

Exercises: ``docker load`` → ``docker run`` → (optional) apply gold patch →
``bash /run_tests.sh`` → parse via ``patch_utils.parse_r2e_pytest_log`` →
grade via ``patch_utils.grade_eval_output_r2e``.

**Sidesteps local docker exec stdio quirks** (seen on rootless docker inside
nested containers) by redirecting test output to a file inside the container
and pulling it out with ``docker cp`` — no reliance on ``docker exec`` stdout.

Prerequisites
-------------
1. An R2E-Gym parquet produced by ``preprocess_r2egym.py`` (or a row of one).
2. The ``<iid>.tar.gz`` for that instance in ``--tar-dir``.

Usage
-----
Test first R2E instance in the parquet (3 scenarios: no-patch / gold-patch
/ bad-patch)::

    python3 data/test_r2e_eval_local.py \\
        --parquet /data/r2egym_smoke.parquet \\
        --tar-dir /data/r2egym_smoke/images

Test a specific instance::

    python3 data/test_r2e_eval_local.py \\
        --parquet /data/r2egym.parquet \\
        --tar-dir /data/r2egym/images \\
        --instance-id <iid>

Exit code
---------
0 — all scenarios matched expected outcome
1 — setup error (missing tarball, docker fail, …)
2 — eval result unexpected (e.g. gold patch did NOT yield reward=1)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent  # aws_kiro_infra/
sys.path.insert(0, str(REPO_ROOT))

from server.patch_utils import (           # noqa: E402
    parse_r2e_pytest_log,
    grade_eval_output_r2e,
    detect_eval_strategy,
)

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("test_r2e_eval_local")

CONTAINER_NAME = "r2e_local_test"
TEST_OUTPUT_PATH_IN_CONTAINER = "/tmp/r2e_eval_output.log"
TEST_OUTPUT_PATH_LOCAL = "/tmp/r2e_eval_output.log"
PATCH_PATH_IN_CONTAINER = "/tmp/r2e_apply.patch"


def _run(cmd: list[str], *, check: bool = True, timeout: int = 60) -> subprocess.CompletedProcess:
    logger.info(" $ %s", " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if check and r.returncode != 0:
        logger.error("command failed (exit=%d): %s", r.returncode, r.stderr[:500])
        raise RuntimeError(f"cmd failed: {' '.join(cmd)}")
    return r


# ---------------------------------------------------------------------------
# Workarounds for docker exec stdio issues: write to file + docker cp out
# ---------------------------------------------------------------------------

def run_tests_via_docker_cp(container: str, timeout: int = 600) -> str:
    """Run ``run_tests.sh`` inside container, pull output via ``docker cp``."""
    # Make sure old output file doesn't linger
    subprocess.run(
        ["docker", "exec", container, "rm", "-f", TEST_OUTPUT_PATH_IN_CONTAINER],
        capture_output=True,
    )
    # Resolve script path (R2E images can place it under /testbed or /root).
    detect = subprocess.run(
        [
            "docker", "exec", container, "bash", "-lc",
            "for p in /run_tests.sh /testbed/run_tests.sh /root/run_tests.sh; do "
            "[ -f \"$p\" ] && echo \"$p\" && exit 0; done; exit 1",
        ],
        capture_output=True, text=True, timeout=30,
    )
    if detect.returncode != 0 or not detect.stdout.strip():
        raise RuntimeError(
            "run_tests.sh not found in known paths: "
            "/run_tests.sh, /testbed/run_tests.sh, /root/run_tests.sh"
        )
    run_tests_path = detect.stdout.strip().splitlines()[-1]
    # Run tests, redirect stdout + stderr to file inside container.
    # || true ensures docker exec returns 0 even if pytest has failures —
    # we want the log regardless, the grader decides reward.
    cmd = (
        f"bash {run_tests_path} > {TEST_OUTPUT_PATH_IN_CONTAINER} 2>&1 || true; "
        f"echo '{run_tests_path} finished'"
    )
    r = subprocess.run(
        ["docker", "exec", container, "bash", "-c", cmd],
        capture_output=True, text=True, timeout=timeout,
    )
    logger.info("  run_tests.sh exec stdout: %r", (r.stdout or "")[:100])
    logger.info("  run_tests.sh exec stderr: %r", (r.stderr or "")[:100])
    # Pull output file out
    _run(["docker", "cp", f"{container}:{TEST_OUTPUT_PATH_IN_CONTAINER}", TEST_OUTPUT_PATH_LOCAL])
    with open(TEST_OUTPUT_PATH_LOCAL) as f:
        content = f.read()
    logger.info("  captured output: %d chars, %d lines", len(content), content.count("\n"))
    return content


def apply_patch_via_docker_cp(container: str, patch: str, cwd: str = "/testbed") -> tuple[bool, str]:
    """Apply patch inside container via ``docker cp`` then ``git apply``."""
    with tempfile.NamedTemporaryFile("w", suffix=".patch", delete=False) as f:
        f.write(patch)
        local_patch_path = f.name
    try:
        _run(["docker", "cp", local_patch_path, f"{container}:{PATCH_PATH_IN_CONTAINER}"])
        # Reset first (we may have applied something previously) then apply
        cmd = (
            f"cd {cwd} && git reset --hard HEAD && git clean -fd && "
            f"git apply {PATCH_PATH_IN_CONTAINER} > /tmp/apply.log 2>&1"
        )
        r = subprocess.run(
            ["docker", "exec", container, "bash", "-c", cmd],
            capture_output=True, text=True, timeout=60,
        )
        if r.returncode != 0:
            # Pull apply.log out to see what went wrong
            try:
                _run(["docker", "cp", f"{container}:/tmp/apply.log", "/tmp/r2e_apply.log"])
                with open("/tmp/r2e_apply.log") as f:
                    err_log = f.read()
            except Exception:
                err_log = "(could not retrieve apply.log)"
            return False, f"git apply failed:\n{err_log[:2000]}"
        return True, "applied OK"
    finally:
        os.unlink(local_patch_path)


def reset_container_state(container: str, cwd: str = "/testbed") -> None:
    """Revert container to HEAD (undo previous patches)."""
    subprocess.run(
        ["docker", "exec", container, "bash", "-c",
         f"cd {cwd} && git reset --hard HEAD && git clean -fd"],
        capture_output=True, timeout=60,
    )


# ---------------------------------------------------------------------------
# Main test
# ---------------------------------------------------------------------------

def pick_instance(parquet_path: Path, instance_id: str | None) -> dict:
    df = pd.read_parquet(parquet_path)
    r2e_mask = df["data_source"].str.contains("r2e", case=False, na=False)
    r2e_df = df[r2e_mask]
    if len(r2e_df) == 0:
        raise RuntimeError(f"No r2e-gym rows in {parquet_path}")
    if instance_id:
        match = r2e_df[r2e_df["instance"].apply(lambda x: x["instance_id"]) == instance_id]
        if len(match) == 0:
            raise RuntimeError(
                f"instance_id={instance_id!r} not found. "
                f"Available (first 5): "
                f"{r2e_df['instance'].apply(lambda x: x['instance_id']).head().tolist()}"
            )
        return {k: match.iloc[0][k] for k in ("data_source", "instance")}
    row = r2e_df.iloc[0]
    return {k: row[k] for k in ("data_source", "instance")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True, help="R2E-Gym parquet (产自 preprocess_r2egym.py)")
    ap.add_argument("--tar-dir", required=True, help="Tarball directory (<iid>.tar.gz)")
    ap.add_argument("--instance-id", default=None, help="Specific instance_id to test (default: first r2e row)")
    ap.add_argument("--keep-container", action="store_true", help="Don't remove container at end")
    ap.add_argument("--skip-load", action="store_true", help="Skip docker load (assume image already loaded)")
    ap.add_argument("--tests-timeout", type=int, default=600, help="Timeout for /run_tests.sh")
    args = ap.parse_args()

    parquet = Path(args.parquet)
    tar_dir = Path(args.tar_dir)
    if not parquet.exists():
        logger.error("parquet not found: %s", parquet)
        return 1

    logger.info("=" * 70)
    logger.info("Picking R2E-Gym instance from %s", parquet)
    row = pick_instance(parquet, args.instance_id)
    inst = row["instance"]
    ds = row["data_source"]

    iid = inst["instance_id"]
    docker_image = inst.get("docker_image")
    expected_json = inst.get("expected_output_json", "")
    gold_patch = inst.get("parsed_commit_content", "") or inst.get("patch", "")

    logger.info("  instance_id        : %s", iid)
    logger.info("  data_source        : %s", ds)
    logger.info("  docker_image       : %s", docker_image)
    logger.info("  expected_output_json: %d chars", len(expected_json))
    logger.info("  gold_patch         : %d chars", len(gold_patch))

    strategy = detect_eval_strategy(inst, ds)
    logger.info("  detect_eval_strategy → %s", strategy)
    if strategy != "r2e":
        logger.error("Expected strategy=r2e, got %s. Check your parquet's data_source.", strategy)
        return 1

    if not docker_image:
        logger.error("instance missing docker_image field")
        return 1

    if not expected_json:
        logger.error("instance missing expected_output_json field")
        return 1

    # Validate expected dict
    try:
        expected_dict = json.loads(expected_json)
        logger.info("  expected_output_json parses OK: %d tests", len(expected_dict))
    except json.JSONDecodeError as e:
        logger.error("expected_output_json is not valid JSON: %s", e)
        return 1

    # ── Step 1: docker load ────────────────────────────────────────────────
    tarball = tar_dir / f"{iid}.tar.gz"
    if not args.skip_load:
        if not tarball.exists():
            logger.error(
                "Tarball not found: %s\nRun this first:\n  bash data/download_swe_images.sh "
                "--prompt-data %s --output-dir %s --max 1",
                tarball, parquet, tar_dir,
            )
            return 1
        logger.info("")
        logger.info("=== Step 1/5: docker load ===")
        _run(["docker", "load", "-i", str(tarball)], timeout=300)

    # Pre-cleanup old container if any
    subprocess.run(["docker", "rm", "-f", CONTAINER_NAME], capture_output=True)

    # ── Step 2: docker run ─────────────────────────────────────────────────
    logger.info("")
    logger.info("=== Step 2/5: docker run ===")
    r = _run([
        "docker", "run", "-d",
        "--name", CONTAINER_NAME,
        "--pull", "never",
        docker_image, "sleep", "3600",
    ])
    container_id = r.stdout.strip()
    logger.info("  container: %s", container_id[:12])

    exit_code = 0
    try:
        # ── Step 3: Scenario A — no patch (reward should be 0) ────────────
        logger.info("")
        logger.info("=== Step 3/5: Scenario A — no patch (expect reward=0) ===")
        try:
            out_A = run_tests_via_docker_cp(CONTAINER_NAME, timeout=args.tests_timeout)
        except Exception as e:
            logger.error("run_tests.sh (no patch) failed: %s", e)
            return 1
        parsed_A = parse_r2e_pytest_log(out_A)
        result_A = grade_eval_output_r2e(inst, out_A)
        logger.info("  parsed         : %d tests", len(parsed_A))
        logger.info("  resolved       : %s", result_A["resolved"])
        logger.info("  reason         : %s", result_A["report"].get("reason"))
        logger.info("  first 3 parsed : %s", dict(list(parsed_A.items())[:3]))
        if result_A["resolved"]:
            logger.warning("⚠️  no-patch yielded resolved=True — "
                           "this may be fine if the gold patch was already applied "
                           "or if tests are state-independent. Investigate.")

        # ── Step 4: Scenario B — apply gold patch (reward should be 1) ────
        logger.info("")
        logger.info("=== Step 4/5: Scenario B — gold patch (expect reward=1) ===")
        if not gold_patch:
            logger.warning("  No gold patch in parsed_commit_content; skipping")
            result_B = None
        else:
            ok, msg = apply_patch_via_docker_cp(CONTAINER_NAME, gold_patch)
            logger.info("  apply: %s", msg)
            if not ok:
                logger.error("Gold patch apply failed. Skipping scenario B.")
                result_B = None
            else:
                try:
                    out_B = run_tests_via_docker_cp(CONTAINER_NAME, timeout=args.tests_timeout)
                except Exception as e:
                    logger.error("run_tests.sh (gold) failed: %s", e)
                    out_B = ""
                parsed_B = parse_r2e_pytest_log(out_B)
                result_B = grade_eval_output_r2e(inst, out_B)
                logger.info("  parsed   : %d tests", len(parsed_B))
                logger.info("  resolved : %s", result_B["resolved"])
                logger.info("  reason   : %s", result_B["report"].get("reason"))
                if not result_B["resolved"]:
                    mismatches = result_B["report"].get("mismatches", [])[:5]
                    logger.warning("  first mismatches: %s", mismatches)

        # ── Step 5: Scenario C — empty/bad patch (reward should be 0) ─────
        logger.info("")
        logger.info("=== Step 5/5: Scenario C — empty patch (expect reward=0) ===")
        reset_container_state(CONTAINER_NAME)
        # Intentionally break by touching a .py file (append garbage)
        bad_patch_cmd = (
            "cd /testbed && "
            "F=$(find . -name '*.py' -not -path '*/tests/*' | head -1); "
            "echo 'xxxxxx syntax error xxxxxxxx' >> \"$F\""
        )
        subprocess.run(
            ["docker", "exec", CONTAINER_NAME, "bash", "-c", bad_patch_cmd],
            capture_output=True, timeout=30,
        )
        try:
            out_C = run_tests_via_docker_cp(CONTAINER_NAME, timeout=args.tests_timeout)
        except Exception as e:
            logger.warning("run_tests.sh (bad) failed (expected-ish): %s", e)
            out_C = ""
        parsed_C = parse_r2e_pytest_log(out_C)
        result_C = grade_eval_output_r2e(inst, out_C)
        logger.info("  parsed   : %d tests", len(parsed_C))
        logger.info("  resolved : %s", result_C["resolved"])
        if result_C["resolved"]:
            logger.error("⚠️  bad patch still yielded resolved=True — parser/grader problem.")

        # ── Summary ───────────────────────────────────────────────────────
        logger.info("")
        logger.info("=" * 70)
        logger.info("SUMMARY for %s", iid)
        logger.info("  Scenario A (no patch)   : resolved=%s (expected False)",
                    result_A["resolved"])
        if result_B is not None:
            logger.info("  Scenario B (gold patch) : resolved=%s (expected True)",
                        result_B["resolved"])
        logger.info("  Scenario C (bad patch)  : resolved=%s (expected False)",
                    result_C["resolved"])

        # Decide pass/fail
        pass_a = not result_A["resolved"]
        pass_b = result_B is None or result_B["resolved"]
        pass_c = not result_C["resolved"]
        if pass_a and pass_b and pass_c:
            logger.info("  ★ ALL SCENARIOS PASSED ✓")
            exit_code = 0
        else:
            logger.error("  ✗ Some scenarios unexpected — see details above")
            exit_code = 2

    finally:
        if not args.keep_container:
            logger.info("")
            logger.info("Cleanup: docker rm -f %s", CONTAINER_NAME)
            subprocess.run(["docker", "rm", "-f", CONTAINER_NAME], capture_output=True)
        else:
            logger.info("--keep-container: leaving %s running for inspection", CONTAINER_NAME)

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
