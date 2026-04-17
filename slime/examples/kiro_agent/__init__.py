"""
Kiro-on-Strands SLIME Integration

This module provides custom generate and reward functions for integrating
Strands agent workflows with SLIME training.

Usage:
    # Set PYTHONPATH to include Kiro-on-Strands
    export KIRO_ON_STRANDS_PATH=/path/to/Kiro-on-Strands
    export PYTHONPATH=${KIRO_ON_STRANDS_PATH}:${PYTHONPATH}
    
    # Use in SLIME training
    --rollout-function-path examples.kiro_on_strands.slime_generate.generate
    --reward-function-path examples.kiro_on_strands.slime_generate.reward_func
"""
