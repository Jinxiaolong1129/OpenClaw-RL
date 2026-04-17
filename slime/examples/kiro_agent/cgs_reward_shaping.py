"""
CGS Reward Shaping for Slime Training.

Custom reward post-processor that implements VeRL-equivalent reward shaping
for CGS (Context Gathering Sub-agent) tasks:

1. Zero-group on error: zero all rewards for instances with errors
2. Length penalty: per-group length normalization (Kimi k1.5 paper)
3. Outcome gamma decay: reward *= gamma^num_turns (efficiency incentive)
4. Peaked tool call curve: log-growth reward for parallel tool calls

After shaping, applies standard GRPO group normalization (reward - mean).

Optionally, post-GRPO advantage scaling by avg tool calls per turn
(matches VeRL's scale_advantage_by_avg_tool_calls in ray_trainer.py).

Usage:
    --custom-reward-post-process-path examples.kiro_agent.cgs_reward_shaping.cgs_reward_post_process

Environment Variables:
    CGS_GAMMA_DECAY_ENABLE: Enable gamma decay (default: false)
    CGS_GAMMA_DECAY_COEF: Gamma coefficient (default: 0.99)
    CGS_TOOL_CALL_INCENTIVE_ENABLE: Enable peaked tool call curve (default: false)
    CGS_SOFT_TOOL_CALL_CAP: Peak of the tool call curve (default: 8)
    CGS_TOOL_CALL_LOG_ALPHA: Alpha for log growth curve (default: 1.0)
    CGS_LENGTH_PENALTY_ENABLE: Enable length penalty (default: false)
    CGS_LENGTH_PENALTY_COEF: Length penalty coefficient (default: 1.0)
    CGS_ZERO_GROUP_ON_ERROR: Zero all rewards for groups with errors (default: true)
    CGS_GRPO_STD_NORMALIZATION: Divide by std in GRPO normalization (default: false)
    CGS_SCALE_ADV_BY_AVG_TC: Scale advantages by avg_tool_calls_per_turn (default: false)
    CGS_MAX_TC_PER_TURN_LIMIT: Hard limit on max tool calls per turn (default: 0 = disabled)
"""

import logging
import math
import os
from collections import defaultdict

import torch

from slime.utils.types import Sample

logger = logging.getLogger(__name__)

# ============================================================================
# Configuration from environment variables
# ============================================================================

GAMMA_DECAY_ENABLE = os.environ.get("CGS_GAMMA_DECAY_ENABLE", "").lower() in ("1", "true", "yes")
GAMMA_DECAY_COEF = float(os.environ.get("CGS_GAMMA_DECAY_COEF", "0.99"))

TOOL_CALL_INCENTIVE_ENABLE = os.environ.get("CGS_TOOL_CALL_INCENTIVE_ENABLE", "").lower() in ("1", "true", "yes")
SOFT_TOOL_CALL_CAP = float(os.environ.get("CGS_SOFT_TOOL_CALL_CAP", "8"))
TOOL_CALL_LOG_ALPHA = float(os.environ.get("CGS_TOOL_CALL_LOG_ALPHA", "1.0"))

LENGTH_PENALTY_ENABLE = os.environ.get("CGS_LENGTH_PENALTY_ENABLE", "").lower() in ("1", "true", "yes")
LENGTH_PENALTY_COEF = float(os.environ.get("CGS_LENGTH_PENALTY_COEF", "1.0"))

ZERO_GROUP_ON_ERROR = os.environ.get("CGS_ZERO_GROUP_ON_ERROR", "true").lower() in ("1", "true", "yes")

GRPO_STD_NORMALIZATION = os.environ.get("CGS_GRPO_STD_NORMALIZATION", "").lower() in ("1", "true", "yes")

SCALE_ADV_BY_AVG_TC = os.environ.get("CGS_SCALE_ADV_BY_AVG_TC", "").lower() in ("1", "true", "yes")
MAX_TC_PER_TURN_LIMIT = float(os.environ.get("CGS_MAX_TC_PER_TURN_LIMIT", "0"))


def _compute_peaked_tool_call_reward(reward: float, avg_tool_calls: float) -> float:
    """
    Peaked reward curve for parallel tool calls.

    Matches VeRL's q.py implementation:
    - avg_tc <= 0:          reward = 0
    - 0 < avg_tc <= 1:      reward *= avg_tc  (linear ramp-up)
    - 1 < avg_tc <= peak:   reward *= (1 + alpha * log(avg_tc))  (log growth)
    - peak < avg_tc <= 2*peak: linear decay from peaked_reward to -1
    - avg_tc > 2*peak:      reward = -1  (hard cap)
    """
    peak = SOFT_TOOL_CALL_CAP
    alpha = TOOL_CALL_LOG_ALPHA
    hard_cap = peak * 2

    if avg_tool_calls <= 0:
        return 0.0
    elif avg_tool_calls <= 1:
        return reward * avg_tool_calls
    elif avg_tool_calls <= peak:
        return reward * (1 + alpha * math.log(avg_tool_calls))
    elif avg_tool_calls <= hard_cap:
        peaked_reward = reward * (1 + alpha * math.log(peak))
        t = (avg_tool_calls - peak) / (hard_cap - peak)
        return peaked_reward * (1 - t) + (-1.0) * t
    else:
        return -1.0


def cgs_reward_post_process(
    args, samples: list[Sample]
) -> tuple[list[float], list[float]]:
    """
    Custom reward post-processor for CGS tasks.

    Receives a flat list of Samples (already flattened from groups).
    Applies CGS-specific reward shaping, then GRPO group normalization,
    then optional advantage scaling by avg tool calls.

    Operation order matches VeRL's q.py:
      1. Zero-group on error
      2. Length penalty (Kimi k1.5)
      3. Gamma decay
      4. Peaked tool call curve
      5. GRPO group normalization
      6. Advantage scaling by avg_tool_calls_per_turn (post-GRPO)

    Returns:
        (raw_rewards, processed_rewards): flat lists of floats, one per sample.
    """
    n = args.n_samples_per_prompt

    # Extract raw rewards
    raw_rewards = [s.reward if s.reward is not None else 0.0 for s in samples]

    # Work on a copy for shaping
    shaped = list(raw_rewards)

    # Group samples by prompt (n_samples_per_prompt consecutive samples per group)
    num_groups = len(samples) // n if n > 0 else len(samples)

    for g in range(num_groups):
        start = g * n
        end = start + n
        group_samples = samples[start:end]

        # Step 1: Zero-group on error
        if ZERO_GROUP_ON_ERROR:
            has_error = any(
                s.status == Sample.Status.FAILED or s.metadata.get("kos_status") == "error"
                for s in group_samples
            )
            if has_error:
                instance_id = group_samples[0].metadata.get("instance_id", "?")
                logger.info(f"[CGS_REWARD] Zeroing group {instance_id}: error_in_group")
                for i in range(start, end):
                    shaped[i] = 0.0
                continue

        # Step 2: Length penalty (per-group) — applied first, matching VeRL q.py order
        if LENGTH_PENALTY_ENABLE:
            lengths = [samples[i].response_length or 0 for i in range(start, end)]
            min_len = min(lengths)
            max_len = max(lengths)
            len_range = max_len - min_len
            if len_range > 0:
                for idx, i in enumerate(range(start, end)):
                    lam = 0.5 - (lengths[idx] - min_len) / len_range
                    # VeRL: full penalty for correct (reward == 1), clamped for incorrect
                    penalty = lam if raw_rewards[i] == 1 else min(0.0, lam)
                    shaped[i] += LENGTH_PENALTY_COEF * penalty

        # Step 3: Outcome gamma decay — uses raw reward to gate, matches VeRL
        if GAMMA_DECAY_ENABLE:
            for i in range(start, end):
                num_turns = samples[i].metadata.get("num_turns", 0)
                if raw_rewards[i] > 0 and num_turns > 0:
                    shaped[i] *= GAMMA_DECAY_COEF ** num_turns

        # Step 4: Peaked tool call curve — applied last in shaping, matching VeRL
        if TOOL_CALL_INCENTIVE_ENABLE:
            for i in range(start, end):
                avg_tc = samples[i].metadata.get("avg_tool_calls_per_turn", 0.0)
                shaped[i] = _compute_peaked_tool_call_reward(shaped[i], avg_tc)

    # Step 5: GRPO group normalization (reward - mean, optionally / std)
    rewards_tensor = torch.tensor(shaped, dtype=torch.float)
    if n > 1 and len(samples) >= n:
        rewards_tensor = rewards_tensor.reshape(-1, n)
        mean = rewards_tensor.mean(dim=-1, keepdim=True)
        rewards_tensor = rewards_tensor - mean
        if GRPO_STD_NORMALIZATION:
            std = rewards_tensor.std(dim=-1, keepdim=True)
            rewards_tensor = rewards_tensor / (std + 1e-6)
        rewards_tensor = rewards_tensor.flatten()

    # Step 6: Scale advantages by avg_tool_calls_per_turn (post-GRPO).
    # Matches VeRL ray_trainer.py scale_advantage_by_avg_tool_calls.
    # Since GRPO advantages ≈ processed rewards, scaling here is equivalent
    # to scaling advantages in the trainer.
    if SCALE_ADV_BY_AVG_TC and n > 0:
        for g in range(num_groups):
            start = g * n
            end = start + n

            for i in range(start, end):
                avg_tc = samples[i].metadata.get("avg_tool_calls_per_turn", 0.0)
                rewards_tensor[i] = rewards_tensor[i] * avg_tc

            # Hard limit: if max_tool_calls_per_turn exceeds limit, set
            # advantages to -max(abs(adv)) within the group (actively discourage)
            if MAX_TC_PER_TURN_LIMIT > 0:
                group_adv = rewards_tensor[start:end]
                max_abs_adv = group_adv.abs().max().item()
                if max_abs_adv == 0:
                    max_abs_adv = 1.0
                for i in range(start, end):
                    max_tc = samples[i].metadata.get("max_tool_calls_per_turn", 0)
                    if max_tc > MAX_TC_PER_TURN_LIMIT:
                        rewards_tensor[i] = -max_abs_adv
                        logger.info(
                            f"[CGS_REWARD] max_tc={max_tc} > limit={MAX_TC_PER_TURN_LIMIT}, "
                            f"setting adv to -{max_abs_adv:.4f} for sample {i}"
                        )

    processed = rewards_tensor.tolist()

    # Log summary with reward source breakdown
    if raw_rewards:
        source_counts = defaultdict(int)
        for s in samples:
            source_counts[s.metadata.get("reward_source", "unknown")] += 1
        source_str = ", ".join(f"{k}={v}" for k, v in sorted(source_counts.items()))
        logger.info(
            f"[CGS_REWARD] {num_groups} groups, {len(raw_rewards)} samples: "
            f"raw_mean={sum(raw_rewards)/len(raw_rewards):.3f}, "
            f"shaped_mean={sum(shaped)/len(shaped):.3f}, "
            f"processed_mean={sum(processed)/len(processed):.3f}, "
            f"reward_sources=[{source_str}]"
        )

    return raw_rewards, processed
