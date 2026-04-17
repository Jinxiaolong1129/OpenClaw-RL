"""Trainer-side custom generate — delegates to remote CPU agent servers.

The agent loop lives on the CPU worker pod (``server/swe_agent_server.py``).
This module:
  1. Picks a worker URL from ``SWE_AGENT_URLS`` (least-pending load balance)
  2. POSTs ``/generate_trajectory`` with the task
  3. Receives per-turn token_ids / loss_mask / logprobs + patch + eval_result
  4. Reconstructs a slime ``Sample`` (trajectory mode, byte-perfect)
  5. Runs the trajectory-level LLM judge (aux reward) on the trainer side
  6. Saves artifacts to ``SWE_SAVE_TRAJ_DIR`` (optional)

Exposes the slime-facing entrypoints:
  * ``generate_trajectory(args, sample, sampling_params) -> Sample`` — async, with watchdog
  * ``_build_trajectory_sample(...)`` — moved unchanged from the old
    ``generate_with_swe.py``
"""

from __future__ import annotations

import asyncio
import copy
import json
import os
import time
from pathlib import Path

import httpx
from loguru import logger

from slime.utils.types import Sample

from swe_traj_judge import TrajectoryJudgeAgent


# =============================================================================
# Worker URLs + least-pending load balancing
# =============================================================================

_raw_urls = os.environ.get("SWE_AGENT_URLS", "") or os.environ.get("SWE_REMOTE_URLS", "")
SWE_AGENT_URLS: list[str] = [u.strip().rstrip("/") for u in _raw_urls.split(",") if u.strip()]
if not SWE_AGENT_URLS:
    SWE_AGENT_URLS = ["http://127.0.0.1:5000"]
    logger.warning("[SWE-KIRO] SWE_AGENT_URLS unset; defaulting to {}", SWE_AGENT_URLS)
else:
    logger.info("[SWE-KIRO] {} agent servers: {}", len(SWE_AGENT_URLS), SWE_AGENT_URLS)

_server_pending: dict[str, int] = {url: 0 for url in SWE_AGENT_URLS}
_pending_lock = asyncio.Lock()

_http_client: httpx.AsyncClient | None = None
_http_client_lock = asyncio.Lock()


async def _get_http_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is not None:
        return _http_client
    async with _http_client_lock:
        if _http_client is None:
            timeout = float(os.getenv("SWE_REMOTE_HTTP_TIMEOUT", "3600"))
            n = max(len(SWE_AGENT_URLS), 1)
            _http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(timeout, connect=30.0),
                limits=httpx.Limits(
                    max_connections=64 * n,
                    max_keepalive_connections=32 * n,
                ),
            )
    return _http_client


async def _pick_server() -> str:
    async with _pending_lock:
        url = min(_server_pending.items(), key=lambda kv: kv[1])[0]
        _server_pending[url] += 1
        return url


async def _release_server(url: str) -> None:
    async with _pending_lock:
        _server_pending[url] = max(0, _server_pending.get(url, 0) - 1)


# =============================================================================
# Artifact saving (unchanged from old generate_with_swe.py)
# =============================================================================

def _sanitize_filename(value: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in value)


def _get_save_dir() -> Path | None:
    save_dir = os.getenv("SWE_SAVE_TRAJ_DIR", "").strip()
    if not save_dir:
        return None
    path = Path(save_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save_rollout_artifacts(*, sample: Sample, iid: str, sampling_params: dict, run_info: dict) -> None:
    try:
        save_dir = _get_save_dir()
        if save_dir is None:
            return
        ts_ns = time.time_ns()
        stem = (
            f"{_sanitize_filename(iid)}"
            f"__g{sample.group_index if sample.group_index is not None else 'na'}"
            f"__i{sample.index if sample.index is not None else 'na'}"
            f"__{ts_ns}"
        )
        run_dir = save_dir / stem
        run_dir.mkdir(parents=True, exist_ok=True)
        msgs_light = [
            {k: v for k, v in m.items() if k in ("role", "content")}
            for m in run_info.get("messages", [])
        ]
        traj_payload = {
            "messages": msgs_light,
            "step_debug": run_info.get("step_debug", []),
            "aux_judge": run_info.get("aux_judge", None),
            "info": {
                "instance_id": iid,
                "exit_status": run_info.get("exit_status"),
                "error": run_info.get("error"),
                "steps": run_info.get("n_steps"),
                "patch_source": run_info.get("patch_source"),
                "reward": run_info.get("reward"),
                "eval_result": run_info.get("eval_result"),
                "policy": run_info.get("policy"),
                "group_index": sample.group_index,
                "index": sample.index,
            },
            "trajectory_format": "swe-kiro-agent-remote-1",
        }
        (run_dir / "traj.json").write_text(
            json.dumps(traj_payload, ensure_ascii=True, indent=2, default=str)
        )
        git_patch = run_info.get("git_patch")
        if isinstance(git_patch, str):
            (run_dir / "patch.diff").write_text(git_patch)
        meta_payload = {
            "instance_id": iid,
            "sampling_params": sampling_params,
            "sample_metadata": sample.metadata,
            "sample_prompt": sample.prompt,
            "group_index": sample.group_index,
            "index": sample.index,
        }
        (run_dir / "meta.json").write_text(
            json.dumps(meta_payload, ensure_ascii=True, indent=2, default=str)
        )
        logger.info("[SWE-KIRO] [{}] Saved artifacts to {}", iid, run_dir)
    except Exception as e:
        logger.warning("[SWE-KIRO] [{}] Failed to save artifacts: {}", iid, e)


# =============================================================================
# Trainer-side Sample construction (unchanged from old generate_with_swe.py)
# =============================================================================

def _build_trajectory_sample(
    sample: Sample,
    messages: list[dict],
    outcome_reward: float,
    resolved: bool,
    run_info: dict,
    iid: str,
) -> Sample:
    """Flatten per-message token_ids into a single Sample (trajectory mode)."""
    sample = copy.deepcopy(sample)

    if not messages:
        sample.status = Sample.Status.ABORTED
        sample.reward = {"score": 0.0, "acc": 0.0}
        sample.remove_sample = True
        return sample

    first_asst_idx: int | None = None
    for i, m in enumerate(messages):
        if m.get("role") == "assistant":
            first_asst_idx = i
            break
    if first_asst_idx is None:
        sample.status = Sample.Status.ABORTED
        sample.reward = {"score": 0.0, "acc": 0.0}
        sample.remove_sample = True
        return sample

    prompt_tokens: list[int] = []
    for m in messages[:first_asst_idx]:
        prompt_tokens.extend(m["token_ids"])

    response_tokens: list[int] = []
    loss_mask: list[int] = []
    rollout_log_probs: list[float] = []
    assistant_texts: list[str] = []
    assistant_turn_counter = 0

    for m in messages[first_asst_idx:]:
        if m.get("role") == "assistant":
            assistant_turn_counter += 1
            assistant_texts.append(m.get("content", ""))
        response_tokens.extend(m["token_ids"])
        loss_mask.extend(m["token_mask"])
        rollout_log_probs.extend(m["token_logprobs"])

    if not response_tokens:
        sample.status = Sample.Status.ABORTED
        sample.reward = {"score": 0.0, "acc": 0.0}
        sample.remove_sample = True
        return sample

    assert len(response_tokens) == len(loss_mask) == len(rollout_log_probs), (
        f"length mismatch: tokens={len(response_tokens)}, mask={len(loss_mask)}, "
        f"logprobs={len(rollout_log_probs)}"
    )

    sample.tokens = prompt_tokens + response_tokens
    sample.response = "".join(assistant_texts)
    sample.response_length = len(response_tokens)
    sample.loss_mask = loss_mask
    sample.rollout_log_probs = rollout_log_probs
    sample.status = Sample.Status.COMPLETED
    sample.reward = {"score": float(outcome_reward), "acc": float(bool(resolved))}

    sample.metadata = copy.deepcopy(sample.metadata or {})
    sample.metadata["exit_status"] = run_info.get("exit_status", "")
    sample.metadata["n_turns"] = assistant_turn_counter
    sample.metadata["n_steps"] = run_info.get("n_steps", 0)
    sample.metadata["patch_source"] = run_info.get("patch_source")
    sample.metadata["outcome_reward"] = float(outcome_reward)

    aux_judge = run_info.get("aux_judge")
    if isinstance(aux_judge, dict):
        sample.metadata["aux_judge"] = {
            "score": float(aux_judge.get("score", 0.0)),
            "model": aux_judge.get("model", ""),
            "num_votes": int(aux_judge.get("num_votes", 1)),
            "status": aux_judge.get("status", "ok"),
        }

    return sample


# =============================================================================
# Remote call: POST /generate_trajectory
# =============================================================================

def _build_payload(args, sample: Sample, sampling_params: dict, sglang_router_url: str) -> dict:
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    instance = metadata.get("instance", metadata)
    if not isinstance(instance, dict):
        instance = {}
    data_source = metadata.get("data_source", instance.get("data_source", "swe-gym"))
    iid = instance.get("instance_id", f"instance_{sample.index}")

    if isinstance(sample.prompt, str):
        problem_statement = sample.prompt
    elif isinstance(sample.prompt, list):
        user_msgs = [m for m in sample.prompt if m.get("role") == "user"]
        problem_statement = user_msgs[-1]["content"] if user_msgs else str(sample.prompt)
    else:
        problem_statement = str(sample.prompt)
    # Prefer instance.problem_statement if provided (SWE-Bench canonical)
    problem_statement = instance.get("problem_statement", problem_statement) or problem_statement

    return {
        "instance_id": iid,
        "problem_statement": problem_statement,
        "instance": instance,
        "data_source": data_source,
        "sampling_params": {
            "temperature": sampling_params.get("temperature", 1.0),
            "top_p": sampling_params.get("top_p", 0.95),
            "max_new_tokens": int(
                getattr(args, "rollout_max_response_len", 0) or sampling_params.get("max_new_tokens", 4096)
            ),
        },
        "sglang_router_url": sglang_router_url,
        "max_context_len": int(getattr(args, "rollout_max_context_len", 0) or 32768),
        "max_new_tokens": int(getattr(args, "rollout_max_response_len", 0) or 4096),
        "step_limit": int(os.getenv("SWE_STEP_LIMIT", "30")),
        "skip_eval": os.getenv("SWE_SKIP_EVAL", "0").strip() not in ("0", "", "false", "no"),
        "eval_timeout": int(os.getenv("SWE_EVAL_TIMEOUT", "300")),
        "exec_timeout": int(os.getenv("SWE_EXEC_TIMEOUT", "180")),
        "strict_no_test": os.getenv("SWE_STRICT_NO_TEST_PATCH", "1").strip() != "0",
        "strict_no_config": os.getenv("SWE_STRICT_NO_CONFIG_PATCH", "1").strip() != "0",
        "test_patch_policy_scope": os.getenv("SWE_TEST_PATCH_POLICY_SCOPE", "eval_tests_only"),
        "request_id": f"{iid}__g{sample.group_index}__i{sample.index}__{time.time_ns()}",
    }


async def _post_remote(url: str, payload: dict) -> dict:
    """POST /generate_trajectory with retries."""
    client = await _get_http_client()
    max_retries = int(os.getenv("SWE_REMOTE_MAX_RETRIES", "3"))
    retry_delay = float(os.getenv("SWE_REMOTE_RETRY_DELAY", "5.0"))
    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            r = await client.post(f"{url}/generate_trajectory", json=payload)
            r.raise_for_status()
            return r.json()
        except (httpx.HTTPStatusError, httpx.RequestError, httpx.TimeoutException) as e:
            last_exc = e
            logger.warning(
                "[SWE-KIRO] remote POST failed attempt {}/{}: {}: {}",
                attempt, max_retries, type(e).__name__, e,
            )
            if attempt < max_retries:
                await asyncio.sleep(retry_delay * attempt)
    raise last_exc or RuntimeError("remote POST failed with no specific exception")


# =============================================================================
# Trainer-side orchestration
# =============================================================================

async def _generate_impl(args, sample: Sample, sampling_params: dict) -> Sample:
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    instance = metadata.get("instance", metadata)
    if not isinstance(instance, dict):
        instance = {}
    iid = instance.get("instance_id", "unknown")

    router_ip = getattr(args, "sglang_router_ip", None)
    router_port = getattr(args, "sglang_router_port", None)
    if not (router_ip and router_port):
        raise RuntimeError(
            "--sglang-router-ip / --sglang-router-port required (rollout-external mode)"
        )
    sglang_router_url = f"http://{router_ip}:{router_port}"

    payload = _build_payload(args, sample, sampling_params, sglang_router_url)
    server_url = await _pick_server()
    logger.info(
        "[SWE-KIRO] [{}] → {} (n_urls={}, group={}, idx={})",
        iid, server_url, len(SWE_AGENT_URLS), sample.group_index, sample.index,
    )

    t_start = time.time()
    try:
        remote_result = await _post_remote(server_url, payload)
    except Exception as e:
        logger.error("[SWE-KIRO] [{}] remote call failed: {}", iid, e)
        sample = copy.deepcopy(sample)
        sample.status = Sample.Status.ABORTED
        sample.reward = {"score": 0.0, "acc": 0.0}
        sample.remove_sample = True
        return sample
    finally:
        await _release_server(server_url)

    messages = remote_result.get("messages", []) or []
    reward = int(remote_result.get("reward", 0))
    resolved = bool((remote_result.get("eval_result") or {}).get("resolved", False))

    run_info = {
        "messages": messages,
        "step_debug": remote_result.get("step_debug", []),
        "git_patch": remote_result.get("git_patch"),
        "patch_source": remote_result.get("patch_source"),
        "exit_status": remote_result.get("exit_status"),
        "n_steps": remote_result.get("n_steps", 0),
        "eval_result": remote_result.get("eval_result"),
        "policy": remote_result.get("policy"),
        "error": remote_result.get("error"),
        "reward": reward,
        "aux_judge": None,
    }

    if not messages:
        _save_rollout_artifacts(sample=sample, iid=iid, sampling_params=sampling_params, run_info=run_info)
        sample = copy.deepcopy(sample)
        sample.status = Sample.Status.ABORTED
        sample.reward = {"score": 0.0, "acc": 0.0}
        sample.remove_sample = True
        return sample

    outcome_reward = 1.0 if reward else -1.0

    # --- Trajectory-level aux judge (runs on trainer, one litellm call per traj) ---
    if bool(getattr(args, "aux_reward_enable", False)):
        try:
            aux = TrajectoryJudgeAgent(
                max_problem_len=int(getattr(args, "aux_judge_max_problem_len", 8000)),
                max_steps_shown=int(getattr(args, "aux_judge_max_steps_shown", 30)),
                max_output_per_step=int(getattr(args, "aux_judge_max_output_per_step", 1200)),
                max_patch_chars=int(getattr(args, "aux_judge_max_patch_chars", 4000)),
            )
            aux_result = await aux.judge_trajectory(
                problem_statement=instance.get("problem_statement", ""),
                step_debug=run_info.get("step_debug", []),
                final_patch=run_info.get("git_patch"),
                resolved=resolved,
            )
            run_info["aux_judge"] = aux_result
            logger.info(
                "[SWE-KIRO] [{}] aux_judge: score={:.3f}",
                iid, float(aux_result.get("score", 0.0)),
            )
        except Exception as e:
            logger.warning("[SWE-KIRO] [{}] aux_judge failed: {}", iid, e)
            run_info["aux_judge"] = {"status": "error", "score": 0.0, "error": str(e)}

    _save_rollout_artifacts(sample=sample, iid=iid, sampling_params=sampling_params, run_info=run_info)

    trajectory_sample = _build_trajectory_sample(
        sample=sample, messages=messages,
        outcome_reward=outcome_reward, resolved=resolved,
        run_info=run_info, iid=iid,
    )

    aux_score = 0.0
    aux_info = run_info.get("aux_judge")
    if isinstance(aux_info, dict):
        aux_score = float(aux_info.get("score", 0.0))
    logger.info(
        "[SWE-KIRO] [{}] DONE status={}, n_turns={}, response_length={}, "
        "outcome={}, aux={:.3f}, exit={}, elapsed={:.1f}s",
        iid,
        trajectory_sample.status.name,
        trajectory_sample.metadata.get("n_turns", 0) if isinstance(trajectory_sample.metadata, dict) else 0,
        trajectory_sample.response_length,
        outcome_reward, aux_score,
        run_info.get("exit_status", ""),
        time.time() - t_start,
    )
    return trajectory_sample


async def generate_trajectory(args, sample: Sample, sampling_params: dict) -> Sample:
    """Entrypoint wired via ``--custom-generate-function-path generate_kiro.generate``."""
    rollout_timeout = float(os.getenv("SWE_ROLLOUT_TIMEOUT", "1800"))
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    instance = metadata.get("instance", metadata)
    if not isinstance(instance, dict):
        instance = {}
    iid = instance.get("instance_id", "unknown")
    try:
        return await asyncio.wait_for(
            _generate_impl(args, sample, sampling_params),
            timeout=rollout_timeout,
        )
    except (asyncio.TimeoutError, TimeoutError):
        logger.error("[SWE-KIRO] [{}] TOTAL ROLLOUT TIMEOUT ({}s)", iid, rollout_timeout)
        sample = copy.deepcopy(sample)
        sample.status = Sample.Status.ABORTED
        sample.reward = {"score": 0.0, "acc": 0.0}
        sample.remove_sample = True
        return sample
