"""
CGS Sample Filter for Slime Training.

Custom dynamic sampling filter that implements VeRL-equivalent group filtering
for CGS (Context Gathering Sub-agent) tasks:

1. Filters groups where all rewards are identical (no learning signal)
2. Filters groups where max reward < min_reward_threshold

Usage:
    --dynamic-sampling-filter-path examples.kiro_agent.cgs_sample_filter.cgs_filter

Environment Variables:
    CGS_MIN_REWARD_THRESHOLD: Minimum reward threshold (default: 0.0)
"""

import logging
import os

from slime.rollout.filter_hub.base_types import DynamicFilterOutput
from slime.utils.types import Sample

logger = logging.getLogger(__name__)

MIN_REWARD_THRESHOLD = float(os.environ.get("CGS_MIN_REWARD_THRESHOLD", "0.0"))


def cgs_filter(args, samples: list[Sample], **kwargs) -> DynamicFilterOutput:
    """
    Dynamic sampling filter for CGS tasks.

    Filters a single group of samples (n_samples_per_prompt copies).

    Filtering criteria (matches VeRL ray_trainer.py filter_groups logic):
    1. Drop groups where all rewards are identical (std=0), unless single sample
    2. Drop groups where max(reward) < MIN_REWARD_THRESHOLD
    """
    rewards = [
        s.reward if s.reward is not None else 0.0
        for s in samples
    ]

    max_r = max(rewards) if rewards else 0.0
    min_r = min(rewards) if rewards else 0.0

    # Criterion 1: filter zero-std groups
    if len(rewards) > 1 and max_r == min_r:
        return DynamicFilterOutput(
            keep=False,
            reason=f"zero_std_{round(rewards[0], 2)}",
        )

    # Criterion 2: filter below threshold
    if max_r < MIN_REWARD_THRESHOLD:
        return DynamicFilterOutput(
            keep=False,
            reason=f"below_threshold_{round(max_r, 2)}_lt_{MIN_REWARD_THRESHOLD}",
        )

    return DynamicFilterOutput(keep=True)
