"""
Custom Generate Function for Kiro-on-Strands (KoS).

Supports two modes of operation:

1. Direct SGLang mode (default):
   Uses the SGLang server launched by launch_external_sglang.py (or SLIME's
   built-in rollout engine) as the LLM backend. Runs the multi-turn agent
   loop locally. Implementation lives in kiro_generate_direct.py.

2. Remote KoS mode (KOS_REMOTE_URLS is set):
   Delegates the entire agent workflow to distributed remote
   sglang_remote_rollout.py FastAPI servers via their /generate_single
   endpoint. Each FastAPI server handles Docker setup, Strands agent
   execution, and trajectory formatting locally, while sharing the same
   SGLang router as the LLM inference backend.

   Architecture:
     Several requests -> distributed to different FastAPI servers (round-robin
     with least-pending) -> each FastAPI server uses the same SGLang router
     IP:port for LLM inference, but tool execution (Docker containers) runs
     locally on each server's machine -> SGLang router distributes LLM
     requests to engines automatically (with sticky routing for KV cache).

   This avoids the bottleneck of a single FastAPI server handling all
   container invocations, while still leveraging centralized SGLang routing
   for KV-cache-aware LLM inference.

Usage:
    --custom-generate-function-path examples.kiro_agent.kiro_generate_with_kos:generate
    --custom-rm-path examples.kiro_agent.kiro_generate_with_kos:reward_func

    # With custom data source (recommended):
    --data-source-path examples.kiro_agent.custom_data_source:AgentDataSource

Environment Variables:
    KOS_REMOTE_URLS: Comma-separated list of remote KoS FastAPI server URLs.
                     e.g. "http://node0:5000,http://node1:5000,http://node2:5000"
                     Also supports the legacy KOS_REMOTE_URL (single URL).
    KOS_TIMEOUT: Request timeout in seconds for remote mode (default: 3600)
    KOS_MAX_RETRIES: Max retries on transient failures (default: 3)
    KOS_RETRY_DELAY: Delay between retries in seconds (default: 5)
    TRAJECTORY_FOLDER: Path to save trajectory logs
"""

import asyncio
import json
import logging
import os
import time
from pathlib import Path

import httpx
import numpy as np

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.types import Sample

try:
    from examples.kiro_agent.kiro_generate_direct import _generate_direct
except ModuleNotFoundError:
    _generate_direct = None  # Only needed for local (non-KOS) mode

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

# Remote KoS mode: distributed FastAPI servers sharing one SGLang router.
_raw_urls = os.environ.get("KOS_REMOTE_URLS", "")
if not _raw_urls:
    _raw_urls = os.environ.get("KOS_REMOTE_URL", "")
KOS_REMOTE_URLS: list[str] = [
    u.strip().rstrip("/") for u in _raw_urls.split(",") if u.strip()
]

KOS_TIMEOUT = int(os.environ.get("KOS_TIMEOUT", "3600"))
KOS_MAX_RETRIES = int(os.environ.get("KOS_MAX_RETRIES", "3"))
KOS_RETRY_DELAY = float(os.environ.get("KOS_RETRY_DELAY", "5"))

TRAJECTORY_FOLDER = os.environ.get("TRAJECTORY_FOLDER")

# CGS task mode: when True, reward is extracted from CGS-specific fields
# (f1_score, judge_score) instead of SWE execution results.
KOS_IS_CGS_TASK = os.environ.get("KOS_IS_CGS_TASK", "").lower() in ("1", "true", "yes")

# Shared async HTTP client for remote mode (lazy-initialized)
_http_client: httpx.AsyncClient | None = None


def _get_http_client() -> httpx.AsyncClient:
    """Get or create the shared async HTTP client for remote KoS mode."""
    global _http_client
    if _http_client is None:
        _http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(KOS_TIMEOUT, connect=30.0),
            limits=httpx.Limits(
                max_connections=64 * max(len(KOS_REMOTE_URLS), 1),
                max_keepalive_connections=32 * max(len(KOS_REMOTE_URLS), 1),
            ),
        )
    return _http_client


# Shared async HTTP client for validate service (lazy-initialized).
_validate_client: httpx.AsyncClient | None = None


def _get_validate_client() -> httpx.AsyncClient:
    """Get or create the shared async HTTP client for the validate service."""
    global _validate_client
    if _validate_client is None:
        timeout = int(os.environ.get("VALIDATE_TIMEOUT", "900"))
        _validate_client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout, connect=30.0),
            limits=httpx.Limits(
                max_connections=256,
                max_keepalive_connections=128,
            ),
        )
    return _validate_client


# ============================================================================
# Distributed Server Selection (for remote KoS mode)
# ============================================================================

_server_pending: dict[str, int] = {url: 0 for url in KOS_REMOTE_URLS}
_server_pending_lock = asyncio.Lock()


async def _pick_server() -> str:
    """Pick the remote KoS server with the fewest in-flight requests."""
    async with _server_pending_lock:
        url = min(_server_pending, key=_server_pending.get)  # type: ignore[arg-type]
        _server_pending[url] += 1
        return url


async def _release_server(url: str) -> None:
    """Decrement the in-flight counter for a server after a request completes."""
    async with _server_pending_lock:
        _server_pending[url] = max(0, _server_pending.get(url, 1) - 1)


# ============================================================================
# Remote KoS Mode: delegate to sglang_remote_rollout.py /generate_single
# ============================================================================

def _build_remote_payload(args, sample: Sample, global_step: int = 0) -> dict:
    """
    Convert a SLIME Sample into a SingleInstanceRequest payload for
    the /generate_single endpoint of sglang_remote_rollout.py.
    """
    instance_id = sample.metadata.get("instance_id", f"instance_{sample.index}")
    rollout_idx = sample.metadata.get(
        "rollout_idx", sample.index % args.n_samples_per_prompt
    )

    if isinstance(sample.prompt, str):
        problem_statement = sample.prompt
    elif isinstance(sample.prompt, list):
        user_msgs = [m for m in sample.prompt if m.get("role") == "user"]
        problem_statement = user_msgs[-1]["content"] if user_msgs else str(sample.prompt)
    else:
        problem_statement = str(sample.prompt)

    metadata = dict(sample.metadata)
    if sample.label is not None:
        metadata["patch"] = sample.label

    # Sanitize metadata values for JSON serialization
    for k, v in metadata.items():
        if isinstance(v, np.ndarray):
            metadata[k] = v.tolist()
        elif isinstance(v, (np.integer, np.floating, np.bool_)):
            metadata[k] = v.item()

    validate = 1 if sample.metadata.get("evaluation", False) else 0

    return {
        "instance_id": instance_id,
        "problem_statement": problem_statement,
        "rollout_idx": rollout_idx,
        "global_step": global_step,
        "validate": validate,
        "metadata": metadata if metadata else None,
    }


async def _post_to_remote(url: str, payload: dict) -> dict:
    """POST JSON to the remote KoS server with retries."""
    client = _get_http_client()
    for attempt in range(1, KOS_MAX_RETRIES + 1):
        try:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPStatusError, httpx.RequestError, httpx.TimeoutException) as e:
            logger.warning(
                f"Remote request failed (attempt {attempt}/{KOS_MAX_RETRIES}): "
                f"{type(e).__name__}: {e}"
            )
            if attempt < KOS_MAX_RETRIES:
                await asyncio.sleep(KOS_RETRY_DELAY * attempt)
            else:
                raise


# ============================================================================
# TITO (Token-In, Token-Out) Trajectory Parsing
# ============================================================================


class TITODataError(Exception):
    """Raised when full TITO data is not available in the logprobs file."""
    pass


def _load_logprobs_file(trajectory_file: str) -> dict:
    """
    Load the TITO logprobs JSON file saved alongside the trajectory.

    Expects full TITO data (vLLM with return_token_ids):
    - full_token_ids: complete token sequence (prompt + all turns)
    - aligned_logprobs: per-token logprobs (0.0 for non-generated positions)
    - response_mask: binary mask (1 = model-generated, 0 = prompt/tool output)

    Returns dict with full TITO data.
    Raises TITODataError if the file is missing or does not contain full TITO data.
    """
    if not trajectory_file:
        raise TITODataError("No trajectory file provided")

    logprobs_file = trajectory_file.replace(".jsonl", "_logprobs.json")
    if not os.path.exists(logprobs_file):
        raise TITODataError(f"Logprobs file not found: {logprobs_file}")

    try:
        with open(logprobs_file, "r") as f:
            data = json.load(f)
    except Exception as e:
        raise TITODataError(f"Failed to load logprobs file {logprobs_file}: {e}") from e

    full_token_ids = data.get("full_token_ids", [])
    aligned_logprobs = data.get("aligned_logprobs", [])
    response_mask = data.get("response_mask", [])

    if not full_token_ids or not response_mask:
        raise TITODataError(
            f"Logprobs file does not contain full TITO data "
            f"(full_token_ids={len(full_token_ids)}, response_mask={len(response_mask)}): "
            f"{logprobs_file}"
        )

    prompt_len = 0
    for i, m in enumerate(response_mask):
        if m == 1:
            prompt_len = i
            break

    # Ensure response_mask and aligned_logprobs match full_token_ids length
    n_tokens = len(full_token_ids)
    if len(response_mask) != n_tokens:
        logger.warning(
            f"[TITO] response_mask length ({len(response_mask)}) != "
            f"full_token_ids length ({n_tokens}), fixing"
        )
        if len(response_mask) < n_tokens:
            response_mask = response_mask + [0] * (n_tokens - len(response_mask))
        else:
            response_mask = response_mask[:n_tokens]

    if len(aligned_logprobs) != n_tokens:
        logger.warning(
            f"[TITO] aligned_logprobs length ({len(aligned_logprobs)}) != "
            f"full_token_ids length ({n_tokens}), fixing"
        )
        if len(aligned_logprobs) < n_tokens:
            aligned_logprobs = aligned_logprobs + [0.0] * (n_tokens - len(aligned_logprobs))
        else:
            aligned_logprobs = aligned_logprobs[:n_tokens]

    logger.info(
        f"[TITO] Loaded full TITO: {len(full_token_ids)} tokens, "
        f"prompt_len={prompt_len}, "
        f"non-zero logprobs={sum(1 for lp in aligned_logprobs[prompt_len:] if lp != 0.0)}"
    )

    return {
        "full_token_ids": full_token_ids,
        "aligned_logprobs": aligned_logprobs,
        "response_mask": response_mask,
        "prompt_len": prompt_len,
    }


def _parse_trajectory_file(trajectory_file: str, tokenizer) -> dict | None:
    """
    Parse a trajectory JSONL file from the remote server to extract
    token-level data for training.

    Requires a companion _logprobs.json file with full TITO data
    (exact token IDs and logprobs from the inference engine).

    Raises TITODataError if full TITO data is not available.
    Returns None if the trajectory file itself is missing.
    """
    if not trajectory_file or not os.path.exists(trajectory_file):
        return None

    tito_data = _load_logprobs_file(trajectory_file)

    full_token_ids = tito_data["full_token_ids"]
    response_mask = tito_data["response_mask"]
    aligned_logprobs = tito_data["aligned_logprobs"]
    prompt_len = tito_data["prompt_len"]

    loss_mask = response_mask[prompt_len:]
    rollout_log_probs = aligned_logprobs[prompt_len:]
    response_length = len(full_token_ids) - prompt_len
    response_text = tokenizer.decode(
        full_token_ids[prompt_len:], skip_special_tokens=False
    )

    # Ensure loss_mask and rollout_log_probs match response_length
    if len(loss_mask) != response_length:
        logger.warning(
            f"[TITO] loss_mask length ({len(loss_mask)}) != "
            f"response_length ({response_length}), fixing"
        )
        if len(loss_mask) < response_length:
            loss_mask = loss_mask + [0] * (response_length - len(loss_mask))
        else:
            loss_mask = loss_mask[:response_length]

    if len(rollout_log_probs) != response_length:
        logger.warning(
            f"[TITO] rollout_log_probs length ({len(rollout_log_probs)}) != "
            f"response_length ({response_length}), fixing"
        )
        if len(rollout_log_probs) < response_length:
            rollout_log_probs = rollout_log_probs + [0.0] * (response_length - len(rollout_log_probs))
        else:
            rollout_log_probs = rollout_log_probs[:response_length]

    logger.info(
        f"[TITO] Full mode: {len(full_token_ids)} tokens, "
        f"{response_length} response, {sum(loss_mask)} trainable"
    )

    return {
        "tokens": full_token_ids,
        "loss_mask": loss_mask,
        "rollout_log_probs": rollout_log_probs,
        "response": response_text,
        "response_length": response_length,
    }


# ============================================================================
# Remote KoS Mode: _generate_remote
# ============================================================================

async def _generate_remote(args, sample: Sample, sampling_params: dict) -> Sample:
    """
    Delegate generation to one of the distributed remote KoS FastAPI servers.

    Requests are distributed across servers using least-pending load balancing.
    Each server runs the full Strands agent workflow locally (including Docker
    container execution), while sharing the same SGLang router for LLM inference.
    """
    state = GenerateState(args)
    instance_id = sample.metadata.get("instance_id", f"instance_{sample.index}")
    global_step = sample.metadata.get("global_step", 0)

    if "instance_id" not in sample.metadata:
        logger.warning(
            f"[KOS-remote] 'instance_id' not in sample.metadata, using fallback '{instance_id}'. "
            f"sample.index={sample.index}, metadata_keys={list(sample.metadata.keys())}"
        )

    # Pick the least-loaded server
    server_url = await _pick_server()
    logger.info(
        f"[KOS-remote] Sending {instance_id} to {server_url} "
        f"(pending: {_server_pending})"
    )
    start_time = time.time()

    payload = _build_remote_payload(args, sample, global_step=global_step)
    url = f"{server_url}/generate_single"

    try:
        result = await _post_to_remote(url, payload)
    except Exception as e:
        logger.error(f"[KOS-remote] Request failed for {instance_id} on {server_url}: {e}")
        sample.status = Sample.Status.FAILED
        sample.response = f"Error: Remote rollout failed on {server_url}: {e}"
        sample.tokens = []
        sample.response_length = 0
        sample.loss_mask = []
        sample.rollout_log_probs = []
        return sample
    finally:
        await _release_server(server_url)

    duration = time.time() - start_time
    status = result.get("status", "error")
    server_hostname = result.get("server_hostname", "unknown")

    if status != "success":
        error_msg = result.get("message", "Unknown error")
        logger.error(f"[KOS-remote] Error for {instance_id}: {error_msg}")
        sample.status = Sample.Status.FAILED
        sample.response = f"Error: {error_msg}"
        sample.tokens = []
        sample.response_length = 0
        sample.loss_mask = []
        sample.rollout_log_probs = []
        sample.metadata["kos_error"] = error_msg
        return sample

    server_result = result.get("result", {})
    trajectory_file = result.get("trajectory_file", "")
    patch = server_result.get("test_result", {}).get("git_patch", "")
    agent_duration = server_result.get("agent_duration", duration)
    agent_status = server_result.get("status", "success")

    # Log server-side errors for debugging
    if agent_status not in ("success", "completed"):
        server_error = (
            server_result.get("error_message")
            or server_result.get("error")
            or server_result.get("message")
            or ""
        )
        logger.warning(
            f"[KOS-remote] Agent failed on {server_url} (hostname={server_hostname}): "
            f"agent_status={agent_status}, error={server_error}, "
            f"duration={agent_duration:.1f}s, "
            f"server_result_keys={list(server_result.keys())}"
        )
        logger.debug(f"[KOS-remote] Full server result for {instance_id}: {server_result}")

    # Parse trajectory for token-level data (requires full TITO)
    try:
        traj_data = _parse_trajectory_file(trajectory_file, state.tokenizer)
    except TITODataError as e:
        logger.warning(f"[KOS-remote] TITO data unavailable for {instance_id}: {e}")
        traj_data = None

    if traj_data:
        sample.tokens = traj_data["tokens"]
        sample.loss_mask = traj_data["loss_mask"]
        sample.rollout_log_probs = traj_data["rollout_log_probs"]
        sample.response = traj_data["response"]
        sample.response_length = traj_data["response_length"]
    else:
        # Fallback: tokenize minimal response
        logger.warning(f"[KOS-remote] No trajectory data for {instance_id}, using fallback")
        fallback = patch if patch else "Task completed."
        prompt_text = sample.prompt if isinstance(sample.prompt, str) else str(sample.prompt)
        prompt_tokens = state.tokenizer.encode(prompt_text, add_special_tokens=False)
        resp_tokens = state.tokenizer.encode(fallback, add_special_tokens=False)
        sample.tokens = prompt_tokens + resp_tokens
        sample.response = fallback
        sample.response_length = len(resp_tokens)
        sample.loss_mask = [1] * len(resp_tokens)
        sample.rollout_log_probs = [0.0] * len(resp_tokens)

    # Map status
    if agent_status in ("success", "completed"):
        sample.status = Sample.Status.COMPLETED
    elif agent_status in ("timeout", "truncated"):
        sample.status = Sample.Status.TRUNCATED
    else:
        sample.status = Sample.Status.FAILED

    sample.metadata["patch"] = patch
    sample.metadata["kos_duration"] = agent_duration
    sample.metadata["kos_server"] = server_hostname
    sample.metadata["kos_trajectory_file"] = trajectory_file
    sample.metadata["kos_status"] = agent_status

    # =========================================================================
    # CGS Task: Extract reward from CGS-specific fields
    # =========================================================================
    if KOS_IS_CGS_TASK:
        # CGS reward comes from KoS server's F1 or judge score
        cgs_reward = server_result.get("cgs_reward", 0.0)
        reward_source = server_result.get("reward_source", "f1")
        judge_score = server_result.get("judge_score")
        f1_score = server_result.get("f1_score", 0.0)

        if cgs_reward is not None:
            sample.reward = float(cgs_reward)
        else:
            sample.reward = 0.0

        # Store CGS metrics in metadata for reward shaping
        sample.metadata["reward_source"] = reward_source
        sample.metadata["judge_score"] = judge_score
        sample.metadata["f1_score"] = f1_score
        sample.metadata["cgs_reward"] = cgs_reward
        sample.metadata["num_turns"] = server_result.get("num_turns", 0)
        sample.metadata["avg_tool_calls_per_turn"] = server_result.get("avg_tool_calls_per_turn", 0.0)
        sample.metadata["max_tool_calls_per_turn"] = server_result.get("max_tool_calls_per_turn", 0)
        sample.metadata["found_files"] = server_result.get("found_files", [])

        logger.info(
            f"[KOS-remote-CGS] {instance_id}: reward={sample.reward:.3f}, "
            f"source={reward_source}, judge={judge_score}, f1={f1_score:.3f}, "
            f"turns={sample.metadata['num_turns']}, "
            f"avg_tc={sample.metadata['avg_tool_calls_per_turn']:.1f}, "
            f"status={sample.status.name}, "
            f"response_len={sample.response_length}, "
            f"duration={agent_duration:.1f}s, server={server_hostname}"
        )
        return sample

    # =========================================================================
    # SWE Task: Inline execution reward from KoS server test outcomes
    # =========================================================================
    # Uses execution_result status + predicted_patch_length (aligned with v8 reference)
    execution_result = server_result.get("execution_result", "")
    predicted_patch_length = server_result.get("predicted_patch_length", len(patch) if patch else 0)
    outcome_results = server_result.get("outcome_results", {})

    if predicted_patch_length == 0:
        # empty_patch: no patch generated
        sample.reward = 0.0
        exec_status = "empty_patch"
    elif execution_result == "empty" or (not execution_result and not outcome_results):
        # empty: no test results available
        sample.reward = 0.0
        exec_status = "empty"
    elif execution_result == "success":
        # success: all tests passed
        sample.reward = 1.0
        exec_status = "success"
    elif execution_result == "partial_success":
        # partial_success: calculate from test results
        f2p_success = outcome_results.get("fail_to_pass_success", 0)
        f2p_total = outcome_results.get("total_fail_to_pass", 0)
        p2p_success = outcome_results.get("pass_to_pass_success", 0)
        p2p_total = outcome_results.get("total_pass_to_pass", 0)
        total_tests = f2p_total + p2p_total
        if total_tests > 0:
            sample.reward = (f2p_success + p2p_success) / total_tests
        else:
            sample.reward = 0.0
        exec_status = "partial_success"
    elif execution_result == "fail":
        # fail: all tests failed
        sample.reward = 0.0
        exec_status = "fail"
    elif outcome_results:
        # Fallback: execution_result not set but outcome_results available
        # (backward compat with servers that don't return execution_result yet)
        try:
            f2p_success = outcome_results.get("fail_to_pass_success", 0)
            f2p_total = outcome_results.get("total_fail_to_pass", 0)
            p2p_success = outcome_results.get("pass_to_pass_success", 0)
            p2p_total = outcome_results.get("total_pass_to_pass", 0)
            total_tests = f2p_total + p2p_total
            total_passed = f2p_success + p2p_success

            if total_tests > 0 and total_passed == total_tests:
                sample.reward = 1.0
                exec_status = "success"
            elif total_tests == 0:
                sample.reward = 0.0
                exec_status = "empty"
            elif total_passed > 0:
                sample.reward = total_passed / total_tests
                exec_status = "partial_success"
            else:
                sample.reward = 0.0
                exec_status = "fail"
        except Exception as e:
            logger.warning(
                f"[KOS-remote] Failed to compute reward from outcome_results "
                f"for {instance_id}: {e}"
            )
            sample.reward = 0.0
            exec_status = "error"
    elif VALIDATE_SHARD_MAP_DIR and patch and instance_id:
        # Fallback: use validate service if no outcome_results from KoS server
        exec_status = "validate_fallback"
        try:
            sample.reward = await asyncio.wait_for(
                _compute_execution_reward(instance_id, patch, sample),
                timeout=900.0,
            )
            logger.info(
                f"[KOS-remote] Inline reward (validate fallback) for {instance_id}: "
                f"{sample.reward:.3f}"
            )
        except Exception as e:
            logger.warning(
                f"[KOS-remote] Inline reward failed for {instance_id}, "
                f"will fall back to async_rm: {e}"
            )
    else:
        exec_status = "empty_patch" if not patch else "unknown"
        if not patch:
            sample.reward = 0.0

    logger.info(
        f"[KOS-remote] {instance_id}: exec_result={exec_status}, "
        f"reward={sample.reward if sample.reward is not None else 'N/A'}, "
        f"status={sample.status.name}, "
        f"response_len={sample.response_length}, "
        f"patch_len={predicted_patch_length}, "
        f"patch_lines={len(patch.splitlines()) if patch else 0}, "
        f"duration={agent_duration:.1f}s, server={server_hostname}, "
        f"kos_url={server_url}"
    )

    return sample


# ============================================================================
# Unified Entry Points
# ============================================================================

async def generate(args, sample: Sample, sampling_params: dict) -> Sample:
    """
    Custom generate function for Kiro-on-Strands.

    Automatically selects the mode based on configuration:
    - If KOS_REMOTE_URLS is set: distributes requests across remote
      sglang_remote_rollout.py FastAPI servers (least-pending balancing)
    - Otherwise: runs multi-turn agent loop locally against the SGLang server
    """
    assert not args.partial_rollout, "Partial rollout not supported"

    if KOS_REMOTE_URLS:
        return await _generate_remote(args, sample, sampling_params)
    else:
        if _generate_direct is None:
            raise ImportError(
                "Local mode requires 'examples.kiro_agent.kiro_generate_direct' "
                "but the module was not found. Set KOS_REMOTE_URLS for remote mode."
            )
        return await _generate_direct(args, sample, sampling_params)


def _parse_patch_changes(patch: str) -> set[str]:
    """Extract the set of changed lines (additions/deletions) from a unified diff."""
    changes = set()
    for line in patch.splitlines():
        if line.startswith(("diff ", "index ", "--- ", "+++ ", "@@")):
            continue
        if line.startswith(("+", "-")):
            stripped = line[1:].strip()
            if stripped:
                changes.add(line[0] + stripped)
    return changes


def _patch_similarity(generated: str, reference: str) -> float:
    """Compute Jaccard similarity between two patches based on changed lines."""
    gen_changes = _parse_patch_changes(generated)
    ref_changes = _parse_patch_changes(reference)
    if not ref_changes:
        return 0.0
    intersection = gen_changes & ref_changes
    union = gen_changes | ref_changes
    if not union:
        return 0.0
    return len(intersection) / len(union)


async def reward_func(args, sample: Sample, **kwargs) -> float:
    """
    Reward function for KoS generation.

    Combines patch presence with similarity to the ground-truth patch:
      - No patch generated:                    0.0
      - Patch generated, no ground truth:      0.3  (presence bonus only)
      - Patch generated, with ground truth:    0.3 + 0.7 * jaccard_similarity
    """
    patch = sample.metadata.get("patch", "")
    if not patch:
        return 0.0

    PRESENCE_REWARD = 0.3

    ground_truth = sample.label or ""
    if not ground_truth:
        return PRESENCE_REWARD

    similarity = _patch_similarity(patch, ground_truth)
    reward = PRESENCE_REWARD + 0.7 * similarity

    logger.debug(
        f"[KOS-reward] instance={sample.metadata.get('instance_id', '?')}, "
        f"similarity={similarity:.3f}, reward={reward:.3f}"
    )
    return reward


async def cgs_reward_func(args, sample: Sample, **kwargs) -> float:
    """
    Reward function for CGS tasks.

    In practice this is a no-op safety net: _generate_remote() always sets
    sample.reward inline, so Slime's sglang_rollout.py skips the RM call
    (it checks `if sample.reward is None` before calling async_rm).
    """
    if sample.reward is not None:
        return sample.reward
    return 0.0


# ============================================================================
# Execution-based reward
# ============================================================================

VALIDATE_SHARD_MAP_DIR = os.environ.get("VALIDATE_SHARD_MAP_DIR", "")
VALIDATE_SERVICE_PORT = int(os.environ.get("VALIDATE_SERVICE_PORT", "51429"))
VALIDATE_TIMEOUT = int(os.environ.get("VALIDATE_TIMEOUT", "900"))
VALIDATE_SERVICE_NAMESPACE = os.environ.get("VALIDATE_SERVICE_NAMESPACE", "hyperpod-ns-aladdin")

# Global mapping: instance_id -> list of validate service base URLs.
_instance_to_servers: dict[str, list[str]] = {}
_shard_map_loaded = False

_validate_server_pending: dict[str, int] = {}
_validate_pending_lock = asyncio.Lock()


def _build_server_url(hostname: str) -> str:
    """Build the full server URL from a shard map hostname."""
    if VALIDATE_SERVICE_NAMESPACE:
        return f"http://{hostname}.{VALIDATE_SERVICE_NAMESPACE}:{VALIDATE_SERVICE_PORT}"
    return f"http://{hostname}:{VALIDATE_SERVICE_PORT}"


def _load_shard_map() -> None:
    """Load shard map JSONs and build the instance_id -> [server URLs] table."""
    global _instance_to_servers, _shard_map_loaded, _validate_server_pending
    if _shard_map_loaded:
        return

    if not VALIDATE_SHARD_MAP_DIR:
        logger.warning("[KOS-exec-reward] VALIDATE_SHARD_MAP_DIR not set")
        _shard_map_loaded = True
        return

    shard_dir = Path(VALIDATE_SHARD_MAP_DIR)
    if not shard_dir.is_dir():
        logger.warning(f"[KOS-exec-reward] Shard map dir not found: {shard_dir}")
        _shard_map_loaded = True
        return

    count = 0
    all_servers: set[str] = set()
    for shard_file in sorted(shard_dir.glob("*.json")):
        try:
            with open(shard_file) as f:
                shard = json.load(f)
            hostname = shard["hostname"]
            server_url = _build_server_url(hostname)
            all_servers.add(server_url)
            for iid in shard.get("instance_ids", []):
                if iid not in _instance_to_servers:
                    _instance_to_servers[iid] = []
                if server_url not in _instance_to_servers[iid]:
                    _instance_to_servers[iid].append(server_url)
                count += 1
            logger.info(
                f"[KOS-exec-reward] Loaded {len(shard.get('instance_ids', []))} "
                f"instances from {shard_file.name} -> {server_url}"
            )
        except Exception as e:
            logger.warning(f"[KOS-exec-reward] Failed to load {shard_file}: {e}")

    _validate_server_pending = {url: 0 for url in all_servers}
    _shard_map_loaded = True

    replica_counts = [len(urls) for urls in _instance_to_servers.values()]
    avg_replicas = sum(replica_counts) / len(replica_counts) if replica_counts else 0
    logger.info(
        f"[KOS-exec-reward] Shard map loaded: {len(_instance_to_servers)} instances across "
        f"{len(all_servers)} servers, avg {avg_replicas:.1f} replicas/instance "
        f"(total mappings: {count})"
    )


async def _pick_validate_server(instance_id: str) -> str:
    """Pick the least-loaded validate server that hosts this instance."""
    servers = _instance_to_servers.get(instance_id)
    if not servers:
        raise KeyError(
            f"instance_id {instance_id!r} not found in shard map "
            f"({len(_instance_to_servers)} instances loaded)"
        )

    async with _validate_pending_lock:
        best = min(servers, key=lambda url: _validate_server_pending.get(url, 0))
        _validate_server_pending[best] = _validate_server_pending.get(best, 0) + 1
        return best


async def _release_validate_server(url: str) -> None:
    """Decrement the in-flight counter for a validate server."""
    async with _validate_pending_lock:
        _validate_server_pending[url] = max(0, _validate_server_pending.get(url, 1) - 1)


async def _validate_patch(instance_id: str, patch: str) -> dict:
    """Submit a patch to the least-loaded validate service replica."""
    _load_shard_map()

    server_url = await _pick_validate_server(instance_id)
    validate_url = f"{server_url}/validate"
    logger.info(
        f"[KOS-exec-reward] Sending validation request: "
        f"instance={instance_id}, url={validate_url}"
    )

    client = _get_validate_client()
    try:
        resp = await client.post(
            validate_url,
            json={"instance_id": instance_id, "patch": patch},
        )
        resp.raise_for_status()
        return resp.json()
    except (httpx.HTTPStatusError, httpx.RequestError) as e:
        await _release_validate_server(server_url)
        server_url = None

        all_servers = _instance_to_servers.get(instance_id, [])
        fallbacks = [s for s in all_servers if s != validate_url.rsplit("/validate", 1)[0]]
        for fallback_url in fallbacks:
            logger.warning(
                f"[KOS-exec-reward] Primary failed for {instance_id} "
                f"({type(e).__name__}), trying fallback {fallback_url}"
            )
            fb_validate_url = f"{fallback_url}/validate"
            async with _validate_pending_lock:
                _validate_server_pending[fallback_url] = _validate_server_pending.get(fallback_url, 0) + 1
            try:
                resp = await client.post(
                    fb_validate_url,
                    json={"instance_id": instance_id, "patch": patch},
                )
                resp.raise_for_status()
                return resp.json()
            except Exception:
                await _release_validate_server(fallback_url)
                continue

        raise
    finally:
        if server_url is not None:
            await _release_validate_server(server_url)


async def execution_reward_func(args, sample: Sample, **kwargs) -> float:
    """
    Execution-based reward function for KoS generation.

    If reward was already computed inline during _generate_remote(),
    returns it directly. Otherwise submits the patch to the validate service.
    """
    if sample.reward is not None:
        return sample.reward

    patch = sample.metadata.get("patch", "")
    instance_id = sample.metadata.get("instance_id", "")
    return await _compute_execution_reward(instance_id, patch, sample)


async def _compute_execution_reward(
    instance_id: str, patch: str, sample: Sample | None = None
) -> float:
    """Core reward computation: submit patch to validate service and score."""
    if not patch:
        return 0.0

    if not VALIDATE_SHARD_MAP_DIR:
        logger.warning("[KOS-exec-reward] VALIDATE_SHARD_MAP_DIR not set, falling back to 0.0")
        return 0.0

    if not instance_id:
        logger.warning("[KOS-exec-reward] No instance_id, cannot validate")
        return 0.0

    try:
        t0 = time.monotonic()
        result = await _validate_patch(instance_id, patch)
        elapsed = time.monotonic() - t0
        if sample is not None:
            sample.metadata["reward_execution_time"] = elapsed
        verdict = result.get("verdict", False)
        exit_code = result.get("test_exit_code", -1)
        error = result.get("error")

        if error:
            logger.warning(
                f"[KOS-exec-reward] instance={instance_id}, "
                f"error={error}, reward=0.0, elapsed={elapsed:.1f}s"
            )
            return 0.0

        # Determine execution result status (aligned with v8 reference reward)
        f2p = result.get("fail_to_pass", {})
        p2p = result.get("pass_to_pass", {})

        f2p_passed = len(f2p.get("passed", []))
        f2p_total = (
            f2p_passed
            + len(f2p.get("failed", []))
            + len(f2p.get("missing", []))
        )
        p2p_passed = len(p2p.get("passed", []))
        p2p_total = (
            p2p_passed
            + len(p2p.get("failed", []))
            + len(p2p.get("missing", []))
        )

        if verdict:
            execution_result_status = "success"
            reward = 1.0
        elif f2p_total + p2p_total == 0:
            # No test results at all
            execution_result_status = "empty"
            reward = 0.0
        elif f2p_passed > 0 or p2p_passed > 0:
            # Partial success — at least some tests passed
            execution_result_status = "partial_success"
            total_tests = f2p_total + p2p_total
            reward = (f2p_passed + p2p_passed) / total_tests
        else:
            # All tests failed
            execution_result_status = "fail"
            reward = 0.0

        logger.info(
            f"[KOS-exec-reward] instance={instance_id}, "
            f"status={execution_result_status}, "
            f"verdict={'PASS' if verdict else 'FAIL'}, "
            f"F2P {f2p_passed}/{f2p_total}, P2P {p2p_passed}/{p2p_total}, "
            f"exit_code={exit_code}, reward={reward:.3f}, "
            f"elapsed={elapsed:.1f}s"
        )
        return reward

    except KeyError as e:
        logger.warning(f"[KOS-exec-reward] {e}")
        return 0.0
    except httpx.TimeoutException:
        logger.warning(f"[KOS-exec-reward] Validation timed out for {instance_id}")
        return 0.0
    except Exception as e:
        logger.warning(f"[KOS-exec-reward] Validation failed for {instance_id}: {e}")
        return 0.0
