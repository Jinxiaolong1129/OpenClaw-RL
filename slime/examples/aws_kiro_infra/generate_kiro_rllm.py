"""rllm / DeepSWE-style entrypoint for aws_kiro_infra (K8s 3-pool variant).

Differs from ``generate_kiro`` (Kiro recipe with aux LLM judge) by:
  * **Compact filter** — only trajectories that voluntarily submitted via
    ``COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`` contribute training signal;
    all others (max_steps / context_overflow / timeout / error) get their
    ``loss_mask`` zeroed. Central DeepSWE blog §2.3 mechanism preventing
    reward-collapse from "lucky-patch reinforcement".
  * **No aux LLM judge** — strict 0/1 (or -1/+1) ORM reward. Matches rllm.

Underlying scaffold + HTTP path to CPU agent pod is identical to
``generate_kiro`` — we just post-process the returned Sample.

Usage:
    --custom-generate-function-path generate_kiro_rllm.generate
    --custom-rm-path                 generate_kiro_rllm.reward_func
"""

from __future__ import annotations

from loguru import logger

from slime.utils.types import Sample

from generate_with_swe_remote import generate_trajectory


# Only "submitted" = LLM explicitly emitted the COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT
# sentinel and we extracted a patch from it. Anything else (max_steps, git_diff
# fallback, context_overflow, error, timeout) is masked.
_ACTIVE_EXIT_STATUSES = {"submitted"}


async def generate(args, sample: Sample, sampling_params: dict) -> Sample:
    sample = await generate_trajectory(args, sample, sampling_params)

    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    exit_status = metadata.get("exit_status", "") if isinstance(metadata, dict) else ""
    status = getattr(sample, "status", None)

    # Apply compact filter only to COMPLETED samples with a real response.
    # ABORTED samples already have loss_mask consistent with their state.
    if (
        sample.loss_mask
        and exit_status not in _ACTIVE_EXIT_STATUSES
        and status == Sample.Status.COMPLETED
    ):
        sample.loss_mask = [0] * len(sample.loss_mask)
        sample.metadata["compact_filter_masked"] = True
        iid = "unknown"
        inst = metadata.get("instance") if isinstance(metadata, dict) else None
        if isinstance(inst, dict):
            iid = inst.get("instance_id", iid)
        logger.info(
            f"[RLLM] [{iid}] compact filter masked trajectory "
            f"(exit_status={exit_status or 'unknown'}, "
            f"n_turns={metadata.get('n_turns', 0) if isinstance(metadata, dict) else 0})"
        )

    return sample


async def reward_func(args, sample, **kwargs):
    """Pass-through: ``generate_trajectory`` already set sample.reward.

    Note: unlike ``generate_kiro.reward_func``, this does NOT add an aux LLM
    judge component. The reward is strict 0/1 (or -1/+1 depending on
    ``generate_with_swe_remote._generate_impl``'s outcome_reward convention)
    matching DeepSWE blog §2.2 sparse ORM.
    """
    if isinstance(sample, list):
        return [s.reward for s in sample]
    return sample.reward
