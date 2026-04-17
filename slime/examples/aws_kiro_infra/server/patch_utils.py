"""Patch parsing, policy gate, and eval-harness helpers.

Moved verbatim from ``aws_kiro_infra/generate_with_swe.py`` (trainer-side)
so everything SWE-Bench-harness-related lives on the CPU agent pod.

Supports **two eval strategies** dispatched via ``detect_eval_strategy``:
  - ``"swe_harness"``: SWE-Gym / SWE-Bench / SWE-Bench-Verified
      * uses ``FAIL_TO_PASS`` / ``PASS_TO_PASS`` + pre-rendered ``eval_script``
      * grading via ``grade_eval_output`` (official swebench / swegym harness)
  - ``"r2e"``: R2E-Gym native datasets
      * no FAIL_TO_PASS / no eval_script; image has built-in ``/run_tests.sh``
      * grading via ``grade_eval_output_r2e`` (compare test-status dict
        against instance's ``expected_output_json``)

Mirrors rllm's ``_calculate_reward_*`` dispatcher (``r2egym/runtime/docker.py``).
"""

from __future__ import annotations

import copy
import json
import logging
import os
import re
import tempfile
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger("swe_agent_server.patch_utils")

EvalStrategy = Literal["swe_harness", "r2e"]


# =============================================================================
# Eval strategy detection
# =============================================================================

def detect_eval_strategy(instance: dict, data_source: str) -> EvalStrategy:
    """Decide which reward computation path to take.

    Prefer explicit ``data_source`` tag; fall back to field-presence heuristic.
    """
    ds = (data_source or "").lower()
    # Explicit tag wins
    if "r2e-gym" in ds or "r2e_gym" in ds or ds == "r2e":
        return "r2e"
    if "swe-gym" in ds or "swe-bench" in ds:
        return "swe_harness"
    # Heuristic: R2E-Gym instances have expected_output_json but no FAIL_TO_PASS
    if isinstance(instance, dict):
        has_expected = bool(instance.get("expected_output_json"))
        has_fail_to_pass = bool(instance.get("FAIL_TO_PASS"))
        if has_expected and not has_fail_to_pass:
            return "r2e"
    return "swe_harness"


# =============================================================================
# Patch parsing & validation
# =============================================================================

def parse_bash_action(response_text: str) -> str | None:
    pattern = r"```bash\s*\n(.*?)```"
    match = re.search(pattern, response_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def extract_patch_from_submission(output: str) -> str:
    if not isinstance(output, str):
        return ""
    text = output.lstrip("\n")
    sentinel = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
    if text.startswith(sentinel):
        text = text[len(sentinel):].lstrip("\n")
    return text


def is_valid_git_patch(patch_text: str) -> bool:
    if not isinstance(patch_text, str):
        return False
    text = patch_text.strip()
    if not text:
        return False
    if "diff --git " not in text:
        return False
    has_old = ("--- a/" in text) or ("--- /dev/null" in text)
    has_new = "+++ b/" in text
    return has_old and has_new


def changed_files_from_patch(patch_text: str) -> list[str]:
    if not isinstance(patch_text, str) or not patch_text:
        return []
    files = []
    for m in re.finditer(r"^diff --git a/(.+?) b/(.+?)$", patch_text, flags=re.M):
        files.append(m.group(2))
    return files


# =============================================================================
# Policy gate
# =============================================================================

def is_test_like_path(path: str) -> bool:
    if not isinstance(path, str):
        return False
    return bool(
        re.search(r"(^|/)tests?/", path)
        or re.search(r"(^|/)test_.*", path)
        or re.search(r".*_test\.[^/]+$", path)
        or path.endswith("conftest.py")
    )


def is_config_like_path(path: str) -> bool:
    if not isinstance(path, str):
        return False
    normalized = path.strip().lower()
    basename = normalized.rsplit("/", 1)[-1]
    config_basenames = {
        "pyproject.toml", "setup.cfg", "setup.py", "tox.ini", "pytest.ini",
        ".coveragerc", ".flake8", "mypy.ini", "ruff.toml",
        "requirements.txt", "requirements-dev.txt",
    }
    if basename in config_basenames:
        return True
    if normalized.startswith(".github/workflows/"):
        return True
    if normalized.startswith(".gitlab/"):
        return True
    if normalized.startswith(".circleci/"):
        return True
    return False


def extract_eval_test_files(instance: dict) -> list[str]:
    if not isinstance(instance, dict):
        return []
    files: set[str] = set()
    for key in ("FAIL_TO_PASS", "PASS_TO_PASS"):
        value = instance.get(key)
        if value is None:
            continue
        if isinstance(value, str):
            items = [value]
        else:
            try:
                items = list(value)
            except TypeError:
                items = []
        for item in items:
            if not isinstance(item, str):
                continue
            test_name = item.split("::", 1)[0].strip()
            if test_name:
                files.add(test_name)
    return sorted(files)


def analyze_patch_policy(
    patch_text: str,
    instance: dict | None = None,
    *,
    strict_no_test: bool | None = None,
    strict_no_config: bool | None = None,
    scope: str | None = None,
) -> dict:
    """Flag patches that touch tests or project config files.

    If ``strict_no_test`` / ``strict_no_config`` / ``scope`` are ``None`` we
    fall back to the env-var convention inherited from swe-rl
    (``SWE_STRICT_NO_TEST_PATCH``, ``SWE_STRICT_NO_CONFIG_PATCH``,
    ``SWE_TEST_PATCH_POLICY_SCOPE``). Callers pass explicit args to override.
    """
    changed_files = changed_files_from_patch(patch_text)
    test_files = [f for f in changed_files if is_test_like_path(f)]
    config_files = [f for f in changed_files if is_config_like_path(f)]
    if strict_no_test is None:
        strict_no_test = os.getenv("SWE_STRICT_NO_TEST_PATCH", "1").strip() != "0"
    if strict_no_config is None:
        strict_no_config = os.getenv("SWE_STRICT_NO_CONFIG_PATCH", "1").strip() != "0"
    if scope is None:
        scope = os.getenv("SWE_TEST_PATCH_POLICY_SCOPE", "eval_tests_only").strip().lower()
    if scope not in {"all_tests", "eval_tests_only"}:
        scope = "eval_tests_only"
    eval_test_files = extract_eval_test_files(instance or {})
    eval_test_file_set = set(eval_test_files)
    matched_eval_test_files = [f for f in test_files if f in eval_test_file_set]

    reasons: list[str] = []
    if strict_no_test:
        if scope == "all_tests" and test_files:
            reasons.append("test_file_modified")
        elif scope == "eval_tests_only" and matched_eval_test_files:
            reasons.append("eval_test_file_modified")
    if strict_no_config and config_files:
        reasons.append("config_file_modified")

    return {
        "changed_files": changed_files,
        "test_files": test_files,
        "config_files": config_files,
        "eval_test_files": eval_test_files,
        "matched_eval_test_files": matched_eval_test_files,
        "test_policy_scope": scope,
        "strict_no_test": strict_no_test,
        "strict_no_config": strict_no_config,
        "violated": len(reasons) > 0,
        "reasons": reasons,
    }


# =============================================================================
# Eval script resolution & grading (SWE-Bench / SWE-Gym harness)
# =============================================================================

_eval_script_cache: dict[str, str] = {}
_eval_script_cache_lock = threading.Lock()


def _infer_instance_type(instance: dict) -> str:
    if not isinstance(instance, dict):
        return "swebench"
    data_kind = instance.get("data_kind")
    if isinstance(data_kind, str) and data_kind:
        return data_kind
    if "image_assets" in instance and instance.get("image_assets") is not None:
        return "swebench_multimodal"
    return "swebench"


def resolve_eval_script(instance: dict) -> str:
    direct = instance.get("eval_script", "")
    if isinstance(direct, str) and direct.strip():
        return direct

    iid = str(instance.get("instance_id", ""))
    if iid:
        with _eval_script_cache_lock:
            cached = _eval_script_cache.get(iid)
        if cached is not None:
            return cached

    inst = copy.deepcopy(instance)
    iid = inst.get("instance_id")
    kind = _infer_instance_type(inst)
    if isinstance(iid, str):
        inst["instance_id"] = iid.lower()
    if "version" not in inst and "base_commit" in inst:
        inst["version"] = inst["base_commit"]

    make_test_spec = None
    if kind == "swebench_multimodal":
        try:
            from swebench.harness.test_spec.test_spec import make_test_spec as _make_test_spec  # type: ignore
            make_test_spec = _make_test_spec
        except ModuleNotFoundError:
            logger.error("[eval] %s: swebench make_test_spec unavailable", iid or "unknown")
            return ""
    else:
        try:
            from swegym.harness.test_spec import make_test_spec as _make_test_spec  # type: ignore
            make_test_spec = _make_test_spec
        except ModuleNotFoundError:
            try:
                from swebench.harness.test_spec.test_spec import make_test_spec as _make_test_spec  # type: ignore
                make_test_spec = _make_test_spec
            except ModuleNotFoundError:
                logger.error("[eval] %s: neither swegym nor swebench make_test_spec found", iid or "unknown")
                return ""

    try:
        test_spec = make_test_spec(inst)
        script = getattr(test_spec, "eval_script", "")
        if isinstance(script, str) and script.strip():
            if iid:
                with _eval_script_cache_lock:
                    _eval_script_cache[str(iid)] = script
            return script
        logger.error("[eval] %s: make_test_spec returned empty eval_script", iid or "unknown")
        return ""
    except Exception as e:
        logger.exception("[eval] %s: make_test_spec failed: %s", iid or "unknown", e)
        return ""


def _get_harness_tools(instance: dict):
    kind = _infer_instance_type(instance)
    if kind == "swebench_multimodal":
        from swebench.harness.grading import get_eval_report  # type: ignore
        from swebench.harness.run_evaluation import APPLY_PATCH_PASS  # type: ignore
        from swebench.harness.test_spec.test_spec import make_test_spec  # type: ignore
        return make_test_spec, get_eval_report, APPLY_PATCH_PASS
    try:
        from swegym.harness.grading import get_eval_report  # type: ignore
        from swegym.harness.run_evaluation import APPLY_PATCH_PASS  # type: ignore
        from swegym.harness.test_spec import make_test_spec  # type: ignore
        return make_test_spec, get_eval_report, APPLY_PATCH_PASS
    except ModuleNotFoundError:
        from swebench.harness.grading import get_eval_report  # type: ignore
        from swebench.harness.run_evaluation import APPLY_PATCH_PASS  # type: ignore
        from swebench.harness.test_spec.test_spec import make_test_spec  # type: ignore
        return make_test_spec, get_eval_report, APPLY_PATCH_PASS


def grade_eval_output(
    instance: dict,
    git_patch: str,
    apply_output: str,
    eval_output: str,
) -> dict[str, Any]:
    """Run the official swebench/swegym grader on raw eval stdout.

    Runs in a thread-pool (caller's responsibility) because it does sync disk
    I/O in a temp dir.
    """
    inst = copy.deepcopy(instance)
    iid = str(inst.get("instance_id", "unknown")).lower()
    inst["instance_id"] = iid
    if "version" not in inst and "base_commit" in inst:
        inst["version"] = inst["base_commit"]

    make_test_spec, get_eval_report, apply_patch_pass = _get_harness_tools(inst)
    test_spec = make_test_spec(inst)
    pass_string = f"[{iid}] {apply_patch_pass}:\n{apply_output}"
    test_output = (
        pass_string + "\n"
        + ">>>>> Start Test Output\n"
        + (eval_output or "") + "\n"
        + ">>>>> End Test Output\n"
    )
    with tempfile.TemporaryDirectory() as tmp_dir:
        logs_dir = Path(tmp_dir) / "logs" / iid
        logs_dir.mkdir(parents=True, exist_ok=True)
        test_out_path = logs_dir / "test_output.txt"
        test_out_path.write_text(test_output)
        try:
            grading_report = get_eval_report(
                test_spec=test_spec,
                prediction={"model_patch": git_patch, "instance_id": iid},
                log_path=str(test_out_path),
                include_tests_status=True,
            )
        except Exception as e:
            if "got an unexpected keyword argument" in str(e):
                grading_report = get_eval_report(
                    test_spec=test_spec,
                    prediction={"model_patch": git_patch, "instance_id": iid},
                    test_log_path=str(test_out_path),
                    include_tests_status=True,
                )
            else:
                raise
    report = grading_report[iid]
    return {"resolved": bool(report.get("resolved", False)), "report": report}


# =============================================================================
# R2E-Gym native grader (no FAIL_TO_PASS; compare to expected_output_json)
# =============================================================================

def _decolor(s: str) -> str:
    """Strip ANSI color codes (copied from r2egym.repo_analysis.execution_log_parser)."""
    return re.sub(r"\x1b\[\d+m", "", s)


def _decolor_dict_keys(d: dict) -> dict:
    return {_decolor(k): v for k, v in d.items()}


def parse_r2e_pytest_log(log: str | None) -> dict[str, str]:
    """Parse R2E-Gym pytest output → ``{test_name: "PASSED"|"FAILED"|"ERROR"}``.

    Ported verbatim from ``r2egym.repo_analysis.execution_log_parser.parse_log_pytest``.
    Reads the "short test summary info" tail section.
    """
    if log is None:
        return {}
    out: dict[str, str] = {}
    if "short test summary info" not in log:
        return out
    tail = log.split("short test summary info", 1)[1].strip()
    for line in tail.split("\n"):
        if "PASSED" in line:
            name = ".".join(line.split("::")[1:])
            out[name] = "PASSED"
        elif "FAILED" in line:
            name = ".".join(line.split("::")[1:]).split(" - ")[0]
            out[name] = "FAILED"
        elif "ERROR" in line:
            try:
                name = ".".join(line.split("::")[1:])
            except IndexError:
                name = line
            name = name.split(" - ")[0]
            out[name] = "ERROR"
    return out


def grade_eval_output_r2e(instance: dict, eval_output: str) -> dict[str, Any]:
    """R2E-Gym native reward: compare parsed test-status dict against the
    instance's ``expected_output_json`` (gold dict).

    Mirrors ``r2egym.agenthub.runtime.docker.DockerRuntime._calculate_reward_r2e``.

    Args:
        instance: the parquet row's ``instance`` dict. Must contain
            ``expected_output_json`` (str holding a JSON dict).
        eval_output: raw pytest stdout+stderr from running ``/run_tests.sh``.

    Returns:
        ``{"resolved": bool, "report": {...}}``
    """
    expected_raw = instance.get("expected_output_json")
    if not expected_raw:
        return {
            "resolved": False,
            "report": {"error": "instance missing expected_output_json"},
        }
    try:
        expected = json.loads(expected_raw) if isinstance(expected_raw, str) else dict(expected_raw)
    except Exception as e:
        return {
            "resolved": False,
            "report": {"error": f"expected_output_json JSON parse failed: {e}"},
        }

    parsed = parse_r2e_pytest_log(eval_output)
    parsed = _decolor_dict_keys(parsed)
    expected = _decolor_dict_keys(expected)

    # Normalize keys: strip trailing " - <status>" if present (r2egym convention)
    parsed   = {k.split(" - ")[0]: v for k, v in parsed.items()   if k}
    expected = {k.split(" - ")[0]: v for k, v in expected.items() if k}

    if len(parsed) != len(expected):
        return {
            "resolved": False,
            "report": {
                "reason": "test_count_mismatch",
                "n_parsed": len(parsed),
                "n_expected": len(expected),
                "parsed": parsed,
                "expected": expected,
            },
        }

    mismatches = []
    for k in sorted(parsed.keys()):
        if k not in expected:
            mismatches.append({"test": k, "parsed": parsed[k], "expected": "<missing>"})
        elif parsed[k] != expected[k]:
            mismatches.append({"test": k, "parsed": parsed[k], "expected": expected[k]})

    resolved = len(mismatches) == 0
    return {
        "resolved": resolved,
        "report": {
            "reason": "ok" if resolved else "status_mismatch",
            "n_tests": len(parsed),
            "n_mismatches": len(mismatches),
            "mismatches": mismatches[:20],  # 只留前 20 条避免 payload 爆炸
            "parsed": parsed if resolved else None,
        },
    }
