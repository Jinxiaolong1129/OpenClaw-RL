"""
BIPOD (on-policy distillation) generate function for Kiro agent.

Wraps the existing KoS generate function, then queries an external teacher
SGLang server for top-K logprobs on the student's generated token sequence.
Teacher top-K data is attached to the Sample so it flows through to the
custom distillation loss.

Usage:
    --custom-generate-function-path examples.kiro_bipod.bipod_generate.generate
    --custom-rm-path examples.kiro_bipod.bipod_generate.reward_func

Environment variables:
    TEACHER_URL:  SGLang /generate endpoint for the teacher model.
    TEACHER_TOPK: Number of top-K logprobs (default: 50, overridden by --distill-topk).
    TEACHER_MAX_CONCURRENCY: Max concurrent teacher requests (default: 8).
    KOS_REMOTE_URLS, KOS_TIMEOUT, etc.: Inherited from kiro_generate_with_kos.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import torch

from examples.kiro_agent.kiro_generate_with_kos import (
    cgs_reward_func as _kos_cgs_reward_func,
    generate as _kos_generate
)
from examples.kiro_bipod.teacher_logprobs import query_teacher_topk
from slime.utils.types import Sample

logger = logging.getLogger(__name__)

TEACHER_TOPK = int(os.environ.get("TEACHER_TOPK", "50"))

# Debug dump directory. Set BIPOD_DEBUG_DIR env var to enable saving.
# Each sample is written as a separate JSON file for post-hoc analysis.
BIPOD_DEBUG_DIR = os.environ.get("BIPOD_DEBUG_DIR", "")

_dump_counter = 0


def _dump_sample_debug(sample: Sample, K: int, teacher_duration: float | None = None) -> None:
    """Serialize key Sample fields and debug metrics to a JSON file.

    Writes tokens, loss_mask, rollout_log_probs, teacher top-k data,
    and derived metrics that help verify the distillation loss is being
    applied correctly on multi-turn trajectories.
    """
    if not BIPOD_DEBUG_DIR:
        return

    global _dump_counter
    _dump_counter += 1

    debug_dir = Path(BIPOD_DEBUG_DIR)
    debug_dir.mkdir(parents=True, exist_ok=True)

    instance_id = (sample.metadata.get("instance_id", "unknown") if sample.metadata else "unknown")
    filename = f"sample_{_dump_counter:06d}_{instance_id}.json"

    # --- Core fields ---
    data: dict = {
        "dump_index": _dump_counter,
        "instance_id": instance_id,
        "status": sample.status.name,
        "response_length": sample.response_length,
        "total_token_count": len(sample.tokens) if sample.tokens else 0,
        "prompt_length": (len(sample.tokens) - sample.response_length) if sample.tokens else 0,
        "tokens": sample.tokens,
        "loss_mask": sample.loss_mask,
        "rollout_log_probs": sample.rollout_log_probs,
    }

    # --- Teacher top-k data ---
    teacher_topk_lp = getattr(sample, "teacher_topk_log_probs", None)
    teacher_topk_idx = getattr(sample, "teacher_topk_indices", None)
    if teacher_topk_lp is not None:
        t = teacher_topk_lp if isinstance(teacher_topk_lp, torch.Tensor) else torch.tensor(teacher_topk_lp)
        data["teacher_topk_log_probs"] = t.tolist()
        data["teacher_topk_log_probs_shape"] = list(t.shape)
    if teacher_topk_idx is not None:
        t = teacher_topk_idx if isinstance(teacher_topk_idx, torch.Tensor) else torch.tensor(teacher_topk_idx)
        data["teacher_topk_indices"] = t.tolist()
        data["teacher_topk_indices_shape"] = list(t.shape)

    # --- Derived debug metrics for verifying distillation correctness ---
    loss_mask = sample.loss_mask
    if loss_mask is not None:
        trainable_positions = sum(loss_mask)
        data["trainable_token_count"] = trainable_positions
        data["trainable_fraction"] = trainable_positions / max(len(loss_mask), 1)

        # Show contiguous masked/unmasked spans to verify multi-turn masking
        spans = []
        if loss_mask:
            current_val = loss_mask[0]
            start = 0
            for i in range(1, len(loss_mask)):
                if loss_mask[i] != current_val:
                    spans.append({"start": start, "end": i, "mask_value": current_val, "length": i - start})
                    current_val = loss_mask[i]
                    start = i
            spans.append({"start": start, "end": len(loss_mask), "mask_value": current_val, "length": len(loss_mask) - start})
        data["loss_mask_spans"] = spans

    # Teacher log-prob statistics (for positions with loss_mask=1)
    if teacher_topk_lp is not None and loss_mask is not None:
        t_lp = teacher_topk_lp if isinstance(teacher_topk_lp, torch.Tensor) else torch.tensor(teacher_topk_lp)
        # Only response portion of loss_mask matters; teacher tensors are [response_length, K]
        resp_mask = loss_mask[-sample.response_length:] if sample.response_length > 0 else []
        mask_t = torch.tensor(resp_mask, dtype=torch.bool)
        if mask_t.any() and t_lp.shape[0] == len(resp_mask):
            masked_lp = t_lp[mask_t]  # [num_trainable, K]
            # Top-K coverage: sum of exp(logprob) across K tokens per position
            topk_probs = torch.exp(masked_lp)
            coverage = topk_probs.sum(dim=-1)  # [num_trainable]
            data["teacher_topk_coverage_mean"] = coverage.mean().item()
            data["teacher_topk_coverage_min"] = coverage.min().item()
            data["teacher_topk_coverage_max"] = coverage.max().item()
            # Mean teacher log-prob for the most likely token
            data["teacher_top1_logprob_mean"] = masked_lp[:, 0].mean().item()
            # Check for placeholder data (-1e10)
            data["has_placeholder_teacher_data"] = bool((masked_lp < -1e9).any())

    # Rollout log-prob statistics at trainable positions
    if sample.rollout_log_probs is not None and loss_mask is not None:
        rlp = torch.tensor(sample.rollout_log_probs, dtype=torch.float32)
        resp_mask = loss_mask[-sample.response_length:] if sample.response_length > 0 else []
        mask_t = torch.tensor(resp_mask, dtype=torch.bool)
        if mask_t.any() and rlp.shape[0] == len(resp_mask):
            masked_rlp = rlp[mask_t]
            data["student_rollout_logprob_mean"] = masked_rlp.mean().item()
            data["student_rollout_logprob_min"] = masked_rlp.min().item()
            data["student_rollout_logprob_max"] = masked_rlp.max().item()

    data["K"] = K
    if teacher_duration is not None:
        data["teacher_query_duration_s"] = teacher_duration

    # Reward info
    data["reward"] = sample.reward

    try:
        with open(debug_dir / filename, "w") as f:
            json.dump(data, f, indent=2, default=str)
        logger.info("[BIPOD] Debug dump written: %s", debug_dir / filename)
    except Exception as e:
        logger.warning("[BIPOD] Failed to write debug dump: %s", e)


def _set_placeholder_teacher_data(sample: Sample, K: int, reason: str) -> None:
    """Attach zero-contribution placeholder teacher tensors.

    Uses -1e10 logprobs so the tail trick puts all mass in the tail bin,
    contributing effectively zero to the KL loss.
    """
    T = sample.response_length
    sample.teacher_topk_log_probs = torch.full((T, K), -1e10, dtype=torch.float32)
    sample.teacher_topk_indices = torch.zeros((T, K), dtype=torch.long)
    sample.loss_mask = [0] * T
    instance_id = sample.metadata.get("instance_id", "?") if sample.metadata else "?"
    logger.warning(
        "[BIPOD] Placeholder teacher data for %s: reason=%s, status=%s, "
        "response_len=%d, has_tokens=%s",
        instance_id, reason, sample.status.name,
        sample.response_length, bool(sample.tokens),
    )


async def generate(args, sample: Sample, sampling_params: dict) -> Sample:
    """Run KoS agent, then query teacher for top-K logprobs.

    The KoS agent produces the full multi-turn trajectory with TITO data
    (exact token IDs, loss mask, rollout logprobs).  After generation,
    we send the student's complete token sequence to the teacher model
    and retrieve per-position top-K logprobs.  These are stored on the
    Sample for use by the distillation loss function.

    Every sample is guaranteed to have ``teacher_topk_log_probs`` and
    ``teacher_topk_indices``; failed/skipped samples get placeholder
    tensors that contribute zero to the KL loss.
    """
    K = getattr(args, "distill_topk", 0) or TEACHER_TOPK

    # Step 1: Run the KoS agent (produces tokens, loss_mask, rollout_log_probs)
    sample = await _kos_generate(args, sample, sampling_params)

    if sample.status not in (Sample.Status.COMPLETED, Sample.Status.TRUNCATED):
        _set_placeholder_teacher_data(sample, K, reason="agent_failed")
        _dump_sample_debug(sample, K)
        return sample
    if not sample.tokens or sample.response_length <= 0:
        _set_placeholder_teacher_data(sample, K, reason="empty_response")
        _dump_sample_debug(sample, K)
        return sample

    # Step 2: Query teacher for top-K logprobs
    if K <= 0:
        _dump_sample_debug(sample, K)
        return sample

    t0 = time.monotonic()
    teacher_data = await query_teacher_topk(
        input_ids=sample.tokens,
        response_length=sample.response_length,
        loss_mask=sample.loss_mask,
        K=K,
    )
    teacher_duration = time.monotonic() - t0

    if teacher_data is not None:
        sample.teacher_topk_log_probs = teacher_data["log_probs"]
        sample.teacher_topk_indices = teacher_data["indices"]

        instance_id = sample.metadata.get("instance_id", "?")
        logger.info(
            "[BIPOD] Teacher query OK for %s: K=%d, response_len=%d, "
            "trainable=%d, teacher_duration=%.1fs",
            instance_id,
            K,
            sample.response_length,
            sum(sample.loss_mask) if sample.loss_mask else 0,
            teacher_duration,
        )
    else:
        _set_placeholder_teacher_data(sample, K, reason="teacher_query_failed")

    _dump_sample_debug(sample, K, teacher_duration=teacher_duration)
    return sample


async def reward_func(args, sample: Sample, **kwargs) -> float:
    """Reward function for BIPOD.

    In pure distillation mode (no GRPO), reward is not used for advantage
    computation.  We delegate to the KoS CGS reward func so that metrics
    (F1, judge score) are still tracked for monitoring.
    """
    return await _kos_cgs_reward_func(args, sample, **kwargs)
