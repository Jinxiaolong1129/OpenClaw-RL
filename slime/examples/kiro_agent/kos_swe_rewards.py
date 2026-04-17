"""
Reward functions for Kiro-on-Strands (KoS) SWE-bench training.

All reward computation lives here so that kiro_generate_with_kos.py can
import whichever reward function it needs via configuration.

Available reward functions (for --custom-rm-path):
    - reward_func:            patch-similarity reward (Jaccard vs ground truth)
    - execution_reward_func:  execution-based reward (validate service)

Helper used inside _generate_remote():
    - compute_inline_execution_reward:  scores a sample from KoS server
                                        outcome fields without a separate
                                        validate call.
"""

import asyncio
import json
import logging
import os
import time
from pathlib import Path

import httpx
import numpy as np

from slime.utils.types import Sample

logger = logging.getLogger(__name__)


# ============================================================================
# SWE-bench GRPO reward (f2p / p2p based)
# ============================================================================

def compute_reward(f2p_passed, f2p_total, p2p_passed, p2p_total,
                   beta=3.0, bonus=0.2):
    """
    Reward design for SWE-bench GRPO training.

    Returns reward in [-1, 1] range.
    Null patch -> 0 (by construction).

    Args:
        beta: harshness of regression penalty (higher = harsher).
              Recommend 2-5.
        bonus: additional bonus for achieving FULL fix with zero regression.
               Helps maintain signal toward the binary-success objective.
    """
    if f2p_total == 0:
        fix_rate = 1.0
    else:
        fix_rate = f2p_passed / f2p_total

    if p2p_total == 0:
        keep_rate = 1.0
    else:
        keep_rate = p2p_passed / p2p_total

    regression_rate = 1.0 - keep_rate

    if regression_rate == 0:
        reward = fix_rate
    else:
        regression_gate = keep_rate ** beta
        reward = fix_rate * regression_gate - (1 - regression_gate) * (1 - fix_rate)

    if fix_rate == 1.0 and keep_rate == 1.0:
        reward += bonus

    reward = np.clip(reward, -1.0, 1.0 + bonus)
    reward = reward / (1.0 + bonus)

    return reward


# ============================================================================
# Inline execution reward (called from _generate_remote)
# ============================================================================

def compute_inline_execution_reward(
    server_result: dict,
    patch: str,
    instance_id: str,
    is_binary=True,
) -> tuple[float | None, str]:
    """
    Compute reward from KoS server outcome fields without a validate call.

    Returns:
        (reward, exec_status) — reward may be None when no determination
        could be made (caller should fall back to validate service or
        async RM).
    """
    execution_result = server_result.get("execution_result", "")
    predicted_patch_length = server_result.get(
        "predicted_patch_length", len(patch) if patch else 0
    )
    outcome_results = server_result.get("outcome_results", {})

    if predicted_patch_length == 0:
        return 0.0, "empty_patch"

    if execution_result == "empty" or (not execution_result and not outcome_results):
        return 0.0, "empty"

    if execution_result == "success":
        return 1.0, "success"

    # elif execution_result == "partial_success":
    #     f2p_success = outcome_results.get("fail_to_pass_success", 0)
    #     f2p_total = outcome_results.get("total_fail_to_pass", 0)
    #     p2p_success = outcome_results.get("pass_to_pass_success", 0)
    #     p2p_total = outcome_results.get("total_pass_to_pass", 0)
    #     total_tests = f2p_total + p2p_total
    #     if total_tests > 0:
    #         return (f2p_success + p2p_success) / total_tests, "partial_success"
    #     else:
    #         return 0.0, "partial_success"

    if execution_result == "fail":
        return 0.0, "fail"

    if outcome_results and execution_result == "partial_success":
        try:
            f2p_success = outcome_results.get("fail_to_pass_success", 0)
            f2p_total = outcome_results.get("total_fail_to_pass", 0)
            p2p_success = outcome_results.get("pass_to_pass_success", 0)
            p2p_total = outcome_results.get("total_pass_to_pass", 0)
            total_tests = f2p_total + p2p_total
            total_passed = f2p_success + p2p_success
            if is_binary:
                if total_tests > 0 and total_passed == total_tests:
                    return 1.0, "success"
                elif total_tests == 0:
                    return 0.0, "empty"
                # elif total_passed > 0:
                #     return total_passed / total_tests, "partial_success"
                else:
                    return 0.0, "fail"
            else:
                reward = compute_reward(f2p_success, f2p_total, p2p_success, p2p_total)
                return reward, "partial success"
        except Exception as e:
            logger.warning(
                f"[KOS-reward] Failed to compute reward from outcome_results "
                f"for {instance_id}: {e}"
            )
            return 0.0, "error"

    # Could not determine — caller should fall back
    return None, "empty_patch" if not patch else "unknown"


# ============================================================================
# Patch-similarity reward
# ============================================================================

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


# ============================================================================
# Execution-based reward (validate service)
# ============================================================================

VALIDATE_SHARD_MAP_DIR = os.environ.get("VALIDATE_SHARD_MAP_DIR", "")
VALIDATE_SERVICE_PORT = int(os.environ.get("VALIDATE_SERVICE_PORT", "51429"))
VALIDATE_TIMEOUT = int(os.environ.get("VALIDATE_TIMEOUT", "900"))
VALIDATE_SERVICE_NAMESPACE = os.environ.get(
    "VALIDATE_SERVICE_NAMESPACE", "hyperpod-ns-aladdin"
)

# Global mapping: instance_id -> list of validate service base URLs.
_instance_to_servers: dict[str, list[str]] = {}
_shard_map_loaded = False

_validate_server_pending: dict[str, int] = {}
_validate_pending_lock = asyncio.Lock()

# Shared async HTTP client for validate service (lazy-initialized)
_validate_client: httpx.AsyncClient | None = None


def _get_validate_client() -> httpx.AsyncClient:
    """Return (or create) the shared async HTTP client for validate calls."""
    global _validate_client
    if _validate_client is None:
        _validate_client = httpx.AsyncClient(
            timeout=httpx.Timeout(VALIDATE_TIMEOUT, connect=30.0),
            limits=httpx.Limits(
                max_connections=200,
                max_keepalive_connections=50,
            ),
        )
    return _validate_client


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
        _validate_server_pending[url] = max(
            0, _validate_server_pending.get(url, 1) - 1
        )


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
        fallbacks = [
            s for s in all_servers
            if s != validate_url.rsplit("/validate", 1)[0]
        ]
        for fallback_url in fallbacks:
            logger.warning(
                f"[KOS-exec-reward] Primary failed for {instance_id} "
                f"({type(e).__name__}), trying fallback {fallback_url}"
            )
            fb_validate_url = f"{fallback_url}/validate"
            async with _validate_pending_lock:
                _validate_server_pending[fallback_url] = (
                    _validate_server_pending.get(fallback_url, 0) + 1
                )
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


async def _compute_execution_reward(
    instance_id: str, patch: str, sample: Sample | None = None
) -> float:
    """Core reward computation: submit patch to validate service and score."""
    if not patch:
        return 0.0

    if not VALIDATE_SHARD_MAP_DIR:
        logger.warning(
            "[KOS-exec-reward] VALIDATE_SHARD_MAP_DIR not set, falling back to 0.0"
        )
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
            execution_result_status = "empty"
            reward = 0.0
        elif f2p_passed > 0 or p2p_passed > 0:
            execution_result_status = "partial_success"
            total_tests = f2p_total + p2p_total
            reward = (f2p_passed + p2p_passed) / total_tests
        else:
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
        logger.warning(
            f"[KOS-exec-reward] Validation timed out for {instance_id}"
        )
        return 0.0
    except Exception as e:
        logger.warning(
            f"[KOS-exec-reward] Validation failed for {instance_id}: {e}"
        )
        return 0.0


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
