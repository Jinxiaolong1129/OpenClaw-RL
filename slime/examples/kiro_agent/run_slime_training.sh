#!/bin/bash
# SLIME training script for Kiro-on-Strands agent workflows
#
# This script runs GRPO training with your custom Strands agent for
# multi-turn rollout with Docker environment interaction.
#
# Prerequisites:
# 1. Install SLIME: pip install -e . --no-deps
# 2. Install strands-sglang: pip install strands-sglang @ git+https://github.com/horizon-rl/strands-sglang.git
# 3. Convert model checkpoint (see below)
# 4. Prepare your dataset in CSV/parquet format

set -e

# ============================================================================
# Configuration - Adjust these for your setup
# ============================================================================

# Kiro-on-Strands path (REQUIRED - set this to your Kiro-on-Strands directory)
KIRO_ON_STRANDS_PATH="${KIRO_ON_STRANDS_PATH:-/path/to/Kiro-on-Strands}"

# Model paths
MODEL_NAME="Qwen3-8B"
HF_CHECKPOINT="/root/models/${MODEL_NAME}"
MCORE_CHECKPOINT="/root/models/${MODEL_NAME}_torch_dist"

# Dataset
TRAIN_DATA="/root/data/your_swe_dataset.csv"  # CSV with instance_id, problem_statement columns
EVAL_DATA="/root/data/your_eval_dataset.csv"

# Training parameters
NUM_GPUS=8
ACTOR_GPUS=4
ROLLOUT_GPUS=4
TP_SIZE=4
GLOBAL_BATCH_SIZE=32
ROLLOUT_BATCH_SIZE=8
N_SAMPLES_PER_PROMPT=4
MAX_RESPONSE_LEN=8192
LEARNING_RATE=1e-6

# Output
SAVE_PATH="/root/checkpoints/${MODEL_NAME}_kiro_grpo"
WANDB_PROJECT="kiro-slime-training"

# ============================================================================
# Setup PYTHONPATH to include Kiro-on-Strands
# ============================================================================

if [ ! -d "${KIRO_ON_STRANDS_PATH}" ]; then
    echo "ERROR: Kiro-on-Strands directory not found at ${KIRO_ON_STRANDS_PATH}"
    echo "Please set KIRO_ON_STRANDS_PATH environment variable:"
    echo "  export KIRO_ON_STRANDS_PATH=/path/to/Kiro-on-Strands"
    exit 1
fi

export PYTHONPATH="${KIRO_ON_STRANDS_PATH}:${PYTHONPATH}"
echo "Added Kiro-on-Strands to PYTHONPATH: ${KIRO_ON_STRANDS_PATH}"

# ============================================================================
# Model Conversion (run once)
# ============================================================================

convert_model() {
    echo "Converting HF checkpoint to Megatron format..."
    
    cd /root/slime
    source scripts/models/qwen3-8B.sh
    
    PYTHONPATH=/root/Megatron-LM:${PYTHONPATH} python tools/convert_hf_to_torch_dist.py \
        ${MODEL_ARGS[@]} \
        --hf-checkpoint ${HF_CHECKPOINT} \
        --save ${MCORE_CHECKPOINT}
    
    echo "Model conversion complete!"
}

# Uncomment to run conversion:
# convert_model

# ============================================================================
# Training
# ============================================================================

echo "Starting SLIME training with Kiro-on-Strands agent..."
echo "Model: ${MODEL_NAME}"
echo "GPUs: ${NUM_GPUS} (Actor: ${ACTOR_GPUS}, Rollout: ${ROLLOUT_GPUS})"
echo "Dataset: ${TRAIN_DATA}"
echo "Kiro-on-Strands: ${KIRO_ON_STRANDS_PATH}"

cd /root/slime

# Source model-specific arguments
source scripts/models/qwen3-8B.sh

# Checkpoint arguments
CKPT_ARGS=(
    --hf-checkpoint ${HF_CHECKPOINT}
    --ref-load ${MCORE_CHECKPOINT}
    --save ${SAVE_PATH}
    --save-interval 50
)

# Rollout arguments - using custom generate function
ROLLOUT_ARGS=(
    --prompt-data ${TRAIN_DATA}
    --input-key problem_statement
    --label-key patch
    # Custom rollout function for Strands agent
    --rollout-function-path examples.kiro_on_strands.slime_generate.generate
    --reward-function-path examples.kiro_on_strands.slime_generate.reward_func
    # Optionally use custom data source
    # --data-source-path examples.kiro_on_strands.slime_generate.KiroDataSource
    --rollout-batch-size ${ROLLOUT_BATCH_SIZE}
    --n-samples-per-prompt ${N_SAMPLES_PER_PROMPT}
    --rollout-max-response-len ${MAX_RESPONSE_LEN}
    --rollout-temperature 1.0
    --global-batch-size ${GLOBAL_BATCH_SIZE}
    --num-rollout 1000
)

# Evaluation arguments
EVAL_ARGS=(
    --eval-interval 20
    --eval-prompt-data kiro_eval ${EVAL_DATA}
    --n-samples-per-eval-prompt 1
    --eval-max-response-len ${MAX_RESPONSE_LEN}
    --eval-top-k 1
)

# GRPO arguments
GRPO_ARGS=(
    --advantage-estimator grpo
    --use-kl-loss
    --kl-loss-coef 0.01
    --kl-loss-type low_var_kl
    --entropy-coef 0.00
    --eps-clip 0.2
    --eps-clip-high 0.28
)

# Optimizer arguments
OPTIMIZER_ARGS=(
    --optimizer adam
    --lr ${LEARNING_RATE}
    --lr-decay-style constant
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.98
)

# Parallelism arguments
PARALLEL_ARGS=(
    --tensor-model-parallel-size ${TP_SIZE}
    --sequence-parallel
    --pipeline-model-parallel-size 1
    --context-parallel-size 1
    --expert-model-parallel-size 1
    --expert-tensor-parallel-size 1
    --use-dynamic-batch-size
    --max-tokens-per-gpu 8192
)

# SGLang arguments
SGLANG_ARGS=(
    --rollout-num-gpus-per-engine ${ROLLOUT_GPUS}
    --sglang-mem-fraction-static 0.7
)

# Actor/Rollout resource allocation
RESOURCE_ARGS=(
    --actor-num-nodes 1
    --actor-num-gpus-per-node ${ACTOR_GPUS}
    --colocate
)

# Misc arguments
MISC_ARGS=(
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    --attention-backend flash
    --seed 42
)

# WandB arguments
WANDB_ARGS=(
    --wandb-project ${WANDB_PROJECT}
    --wandb-name "${MODEL_NAME}-kiro-grpo"
)

# Run training
PYTHONPATH=/root/Megatron-LM:${KIRO_ON_STRANDS_PATH}:${PYTHONPATH} python train.py \
    ${MODEL_ARGS[@]} \
    ${CKPT_ARGS[@]} \
    ${ROLLOUT_ARGS[@]} \
    ${EVAL_ARGS[@]} \
    ${GRPO_ARGS[@]} \
    ${OPTIMIZER_ARGS[@]} \
    ${PARALLEL_ARGS[@]} \
    ${SGLANG_ARGS[@]} \
    ${RESOURCE_ARGS[@]} \
    ${MISC_ARGS[@]} \
    ${WANDB_ARGS[@]}

echo "Training complete!"
