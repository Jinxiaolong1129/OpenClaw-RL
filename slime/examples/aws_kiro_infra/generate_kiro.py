"""Kiro-style algorithm wrapper for the AWS Kiro infra variant (KoS-style + traj judge).

Algorithm profile (Kiro recipe):
  * Sample organization: trajectory mode (per-message token_ids, byte-perfect)
  * Docker:              **per-worker-pod** swe_exec_server (HTTP), KoS-style colocate
                         (replaces parent's remote ECS pool; replaces earlier in-process DinD)
  * Advantage estimator: GRPO (with std normalization)
  * KL loss:             coef=0 (nominally on, effectively off)
  * Clip ratio:          0.2 / 0.28 (DAPO asymmetric)
  * Entropy:             0
  * Auxiliary reward:    trajectory-level LLM-as-judge via TrajectoryJudgeAgent
                         ONE judge call per trajectory, score in [-1, +1]

Reward combination (when ``args.aux_reward_enable=True``):

    final_score = outcome_reward + aux_reward_coef * aux_judge_score

The ``aux_reward_coef`` (default 0.5) controls how much the judge adjusts the
sparse binary outcome — the outcome stays dominant, the judge smoothens it.

Usage:
    --custom-generate-function-path generate_kiro.generate
    --custom-rm-path                 generate_kiro.reward_func
"""

from __future__ import annotations

from slime.utils.types import Sample

from generate_with_swe_remote import generate_trajectory


async def generate(args, sample: Sample, sampling_params: dict) -> Sample:
    return await generate_trajectory(args, sample, sampling_params)


def _get_reward(s: Sample, args) -> dict:
    """Compute the reward dict for one sample, combining outcome + aux_judge
    when the judge is enabled.
    """
    aux_enable = bool(getattr(args, "aux_reward_enable", False))
    aux_coef = float(getattr(args, "aux_reward_coef", 0.5))
    meta = s.metadata if isinstance(s.metadata, dict) else {}

    # outcome_reward lives in metadata (set by _build_trajectory_sample)
    # and in sample.reward["score"] (as the default raw signal).
    outcome_reward = 0.0
    if isinstance(s.reward, dict):
        outcome_reward = float(s.reward.get("score", 0.0))
    elif s.reward is not None:
        outcome_reward = float(s.reward)
    outcome_reward = float(meta.get("outcome_reward", outcome_reward))

    acc = 1.0 if outcome_reward > 0 else 0.0

    if aux_enable:
        aux_info = meta.get("aux_judge") if isinstance(meta, dict) else None
        aux_score = 0.0
        aux_status = "missing"
        if isinstance(aux_info, dict):
            aux_score = float(aux_info.get("score", 0.0))
            aux_status = aux_info.get("status", "missing")
        final_score = outcome_reward + aux_coef * aux_score
        return {
            "score": final_score,
            "acc": acc,
            "outcome_reward": outcome_reward,
            "aux_score": aux_score,
            "aux_coef": aux_coef,
            "aux_status": aux_status,
        }

    # No aux judge: return outcome-only reward.
    if isinstance(s.reward, dict):
        return s.reward
    return {"score": outcome_reward, "acc": acc}


async def reward_func(args, sample, **kwargs):
    if isinstance(sample, list):
        return [_get_reward(s, args) for s in sample]
    return _get_reward(sample, args)
