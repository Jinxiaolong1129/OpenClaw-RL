#!/bin/bash
# =============================================================================
# SLIME Async Training for CGS (Context Gathering Sub-agent) Tasks
# =============================================================================
#
# CGS variant of run-qwen3-30b-kiro-kos-async-debug-run.sh.
# Uses KoS servers in CGS mode with agent-as-judge reward.
#
# Key differences from the SWE script:
#   - Sets KOS_IS_CGS_TASK=true for CGS reward extraction
#   - Uses cgs_reward_shaping for custom reward post-processing
#   - Uses cgs_sample_filter for dynamic sampling filter
#   - Uses cgs_reward_func instead of execution_reward_func
#   - CGS-specific parquet data and configs
#
# Prerequisites:
#   - SGLang GPU job running (launch_sglang_only_async.yaml)
#   - KoS CPU job running with --is_cgs_task flag (launch_kos_only_async.yaml)
#   - CGS training data parquet at PROMPT_DATA path
# =============================================================================

set -ex

export PYTHONBUFFERED=16

# =============================================================================
# External SGLang Server Configuration
# =============================================================================
RUN_NAME=${RUN_NAME:-"cgs_async"}
SGLANG_RUN_NAME=${SGLANG_RUN_NAME:-"sglang_server_cgs"}
SGLANG_ENV_BASE=${SGLANG_ENV_BASE:-"/mnt_out/yawenwuu/logs/slime"}
SGLANG_ENV_FILE=${SGLANG_ENV_FILE:-"${SGLANG_ENV_BASE}/${SGLANG_RUN_NAME}/sglang_external_rollout.env"}

if [ -f "${SGLANG_ENV_FILE}" ]; then
    echo "Loading SGLang config from ${SGLANG_ENV_FILE}"
    source "${SGLANG_ENV_FILE}"
else
    echo "WARNING: ${SGLANG_ENV_FILE} not found."
    echo "Set SGLANG_ROUTER_IP, SGLANG_ROUTER_PORT, SGLANG_ENGINE_ADDRS,"
    echo "SGLANG_TOTAL_GPUS, SGLANG_GPUS_PER_ENGINE manually, or run"
    echo "launch_external_sglang.py with --run-name ${SGLANG_RUN_NAME} first."
fi

SGLANG_ROUTER_IP=${SGLANG_ROUTER_IP:?"ERROR: SGLANG_ROUTER_IP not set"}
SGLANG_ROUTER_PORT=${SGLANG_ROUTER_PORT:?"ERROR: SGLANG_ROUTER_PORT not set"}
SGLANG_ENGINE_ADDRS=${SGLANG_ENGINE_ADDRS:?"ERROR: SGLANG_ENGINE_ADDRS not set"}
SGLANG_TOTAL_GPUS=${SGLANG_TOTAL_GPUS:?"ERROR: SGLANG_TOTAL_GPUS not set"}
SGLANG_GPUS_PER_ENGINE=${SGLANG_GPUS_PER_ENGINE:?"ERROR: SGLANG_GPUS_PER_ENGINE not set"}

# =============================================================================
# KoS Remote Server Configuration
# =============================================================================
KOS_PORT=${KOS_PORT:-5000}

KOS_ENV_FILE=${KOS_ENV_FILE:-"${SGLANG_ENV_BASE}/${KOS_RUN_NAME}/kos_servers.env"}

if [ -z "${KOS_REMOTE_URLS:-}" ]; then
    if [ -f "${KOS_ENV_FILE}" ]; then
        echo "Loading KoS server URLs from ${KOS_ENV_FILE}"
        source "${KOS_ENV_FILE}"
        export KOS_REMOTE_URLS="${KOS_REMOTE_URLS}"
    fi

    if [ -z "${KOS_REMOTE_URLS:-}" ]; then
        _SEEN_IPS=""
        _KOS_URLS=""
        for _ADDR in ${SGLANG_ENGINE_ADDRS}; do
            _IP=$(echo "${_ADDR}" | cut -d: -f1)
            if echo "${_SEEN_IPS}" | grep -qw "${_IP}"; then
                continue
            fi
            _SEEN_IPS="${_SEEN_IPS} ${_IP}"
            if [ -n "${_KOS_URLS}" ]; then
                _KOS_URLS="${_KOS_URLS},http://${_IP}:${KOS_PORT}"
            else
                _KOS_URLS="http://${_IP}:${KOS_PORT}"
            fi
        done
        export KOS_REMOTE_URLS="${_KOS_URLS}"
    fi
else
    export KOS_REMOTE_URLS="${KOS_REMOTE_URLS}"
fi

export KOS_TIMEOUT=${KOS_TIMEOUT:-3600}
export TRAJECTORY_FOLDER=${TRAJECTORY_FOLDER:-"/mnt_out/trajectories"}

# =============================================================================
# CGS Task Configuration
# =============================================================================
export KOS_IS_CGS_TASK=true

# CGS reward shaping (env vars for cgs_reward_shaping.py)
export CGS_ZERO_GROUP_ON_ERROR=${CGS_ZERO_GROUP_ON_ERROR:-true}
export CGS_TOOL_CALL_INCENTIVE_ENABLE=${CGS_TOOL_CALL_INCENTIVE_ENABLE:-false}
export CGS_SOFT_TOOL_CALL_CAP=${CGS_SOFT_TOOL_CALL_CAP:-8}
export CGS_TOOL_CALL_LOG_ALPHA=${CGS_TOOL_CALL_LOG_ALPHA:-1.0}
export CGS_GAMMA_DECAY_ENABLE=${CGS_GAMMA_DECAY_ENABLE:-false}
export CGS_GAMMA_DECAY_COEF=${CGS_GAMMA_DECAY_COEF:-0.99}
export CGS_LENGTH_PENALTY_ENABLE=${CGS_LENGTH_PENALTY_ENABLE:-false}
export CGS_LENGTH_PENALTY_COEF=${CGS_LENGTH_PENALTY_COEF:-1.0}
export CGS_GRPO_STD_NORMALIZATION=${CGS_GRPO_STD_NORMALIZATION:-false}

# Advantage scaling by avg tool calls (matches VeRL scale_advantage_by_avg_tool_calls)
export CGS_SCALE_ADV_BY_AVG_TC=${CGS_SCALE_ADV_BY_AVG_TC:-false}
export CGS_MAX_TC_PER_TURN_LIMIT=${CGS_MAX_TC_PER_TURN_LIMIT:-0}

# CGS sample filter
export CGS_MIN_REWARD_THRESHOLD=${CGS_MIN_REWARD_THRESHOLD:-0.2}

# =============================================================================
# Cluster Configuration
# =============================================================================
SLIME_DIR=${SLIME_DIR:-"/mnt_out/yawenwuu/packages/rl_test_repo/slime"}
SCRIPT_DIR=${SLIME_DIR}/scripts
export MODEL_ARGS_ROTARY_BASE=10000000
source "${SCRIPT_DIR}/models/qwen3-30B-A3B.sh"

export NUM_NODES=${NUM_NODES:-1}
export NUM_GPUS_PER_NODE=${NUM_GPUS_PER_NODE:-8}

if [ -n "$PET_NNODES" ]; then
    export NUM_NODES=${PET_NNODES}
    export MASTER_ADDR=${MASTER_ADDR:-$(hostname | sed -E 's/-worker-[0-9]+$/-worker-0/')}
    export MY_ADDR=$(hostname)
else
    export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
    export MY_ADDR=${MY_ADDR:-"127.0.0.1"}
fi

echo "=============================================="
echo "CGS Async Training Configuration:"
echo "  NUM_NODES: ${NUM_NODES}"
echo "  NUM_GPUS_PER_NODE: ${NUM_GPUS_PER_NODE}"
echo "  MASTER_ADDR: ${MASTER_ADDR}"
echo "  MY_ADDR: ${MY_ADDR}"
echo "External SGLang:"
echo "  SGLANG_ROUTER_IP: ${SGLANG_ROUTER_IP}"
echo "  SGLANG_ROUTER_PORT: ${SGLANG_ROUTER_PORT}"
echo "  SGLANG_ENGINE_ADDRS: ${SGLANG_ENGINE_ADDRS}"
echo "  SGLANG_TOTAL_GPUS: ${SGLANG_TOTAL_GPUS}"
echo "  SGLANG_GPUS_PER_ENGINE: ${SGLANG_GPUS_PER_ENGINE}"
echo "KoS Remote Servers (CGS mode):"
echo "  KOS_REMOTE_URLS: ${KOS_REMOTE_URLS}"
echo "  KOS_IS_CGS_TASK: ${KOS_IS_CGS_TASK}"
echo "CGS Reward Shaping:"
echo "  TOOL_CALL_INCENTIVE: ${CGS_TOOL_CALL_INCENTIVE_ENABLE}"
echo "  GAMMA_DECAY: ${CGS_GAMMA_DECAY_ENABLE}"
echo "  LENGTH_PENALTY: ${CGS_LENGTH_PENALTY_ENABLE}"
echo "  ZERO_GROUP_ON_ERROR: ${CGS_ZERO_GROUP_ON_ERROR}"
echo "=============================================="

# =============================================================================
# Model Checkpoints
# =============================================================================
CKPT_ARGS=(
   --hf-checkpoint /mnt_out/yawenwuu/models/qwen/Qwen3-Coder-30B-A3B-Instruct
   --ref-load /mnt_out/yawenwuu/models/qwen/Qwen3-Coder-30B-A3B-Instruct-mcore
   --load /mnt_out/yawenwuu/models/qwen/kiro_rl/Qwen3-30B_slime_kos_cgs/${RUN_NAME}
   --save /mnt_out/yawenwuu/models/qwen/kiro_rl/Qwen3-30B_slime_kos_cgs/${RUN_NAME}
   --save-interval 1
)

# =============================================================================
# Rollout Configuration (CGS data)
# =============================================================================
PROMPT_DATA=${PROMPT_DATA:-"/mnt_out/yawenwuu/cgs_sft_synthetic/consensus_benchmark_0211/cgs_train.parquet"}
ROLLOUT_ARGS=(
   --data-source-path examples.kiro_agent.custom_data_source.AgentDataSource
   --prompt-data ${PROMPT_DATA}
   --input-key problem_statement
   --label-key expected_groundtruth_files
   --apply-chat-template
   --rollout-shuffle
   --num-rollout 330
   --rollout-batch-size 16
   --n-samples-per-prompt 16
   --rollout-max-response-len 98304
   --rollout-temperature 1
   --num-steps-per-rollout 1
   --balance-data
)

# =============================================================================
# Evaluation
# =============================================================================
EVAL_DATA=${EVAL_DATA:-""}
EVAL_ARGS=()
if [ -n "${EVAL_DATA}" ]; then
    EVAL_ARGS=(
       --eval-interval 400
       --eval-prompt-data cgs_val ${EVAL_DATA}
       --n-samples-per-eval-prompt 16
       --eval-max-response-len 98304
       --eval-top-p 1
    )
fi

# =============================================================================
# Performance / Parallelism
# =============================================================================
TENSOR_MODEL_PARALLEL_SIZE=${TENSOR_MODEL_PARALLEL_SIZE:-4}
SEQUENCE_PARALLEL=${SEQUENCE_PARALLEL:-true}
PIPELINE_MODEL_PARALLEL_SIZE=${PIPELINE_MODEL_PARALLEL_SIZE:-2}
CONTEXT_PARALLEL_SIZE=${CONTEXT_PARALLEL_SIZE:-8}
EXPERT_MODEL_PARALLEL_SIZE=${EXPERT_MODEL_PARALLEL_SIZE:-8}
EXPERT_TENSOR_PARALLEL_SIZE=${EXPERT_TENSOR_PARALLEL_SIZE:-1}

RECOMPUTE_GRANULARITY=${RECOMPUTE_GRANULARITY:-full}
RECOMPUTE_METHOD=${RECOMPUTE_METHOD:-uniform}
RECOMPUTE_NUM_LAYERS=${RECOMPUTE_NUM_LAYERS:-1}

USE_DYNAMIC_BATCH_SIZE=${USE_DYNAMIC_BATCH_SIZE:-true}
MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU:-32768}
LOG_PROBS_MAX_TOKENS_PER_GPU=${LOG_PROBS_MAX_TOKENS_PER_GPU:-65536}
LOG_PROBS_CHUNK_SIZE=${LOG_PROBS_CHUNK_SIZE:-512}

PERF_ARGS=(
   --tensor-model-parallel-size ${TENSOR_MODEL_PARALLEL_SIZE}
   --pipeline-model-parallel-size ${PIPELINE_MODEL_PARALLEL_SIZE}
   --context-parallel-size ${CONTEXT_PARALLEL_SIZE}
   --expert-model-parallel-size ${EXPERT_MODEL_PARALLEL_SIZE}
   --expert-tensor-parallel-size ${EXPERT_TENSOR_PARALLEL_SIZE}

   --recompute-granularity ${RECOMPUTE_GRANULARITY}
   --recompute-method ${RECOMPUTE_METHOD}
   --recompute-num-layers ${RECOMPUTE_NUM_LAYERS}

   --max-tokens-per-gpu ${MAX_TOKENS_PER_GPU}
   --log-probs-max-tokens-per-gpu ${LOG_PROBS_MAX_TOKENS_PER_GPU}
   --log-probs-chunk-size ${LOG_PROBS_CHUNK_SIZE}
)

if [ "${SEQUENCE_PARALLEL}" = "true" ]; then
   PERF_ARGS+=(--sequence-parallel)
fi
if [ "${USE_DYNAMIC_BATCH_SIZE}" = "true" ]; then
   PERF_ARGS+=(--use-dynamic-batch-size)
fi

# =============================================================================
# GRPO with CGS reward shaping and sample filter
# =============================================================================
GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --use-tis
   --kl-loss-coef 0.0
   --kl-loss-type low_var_kl
   --entropy-coef 0.0
   --eps-clip 0.2
   --eps-clip-high 0.28
   --calculate-per-token-loss
   --dynamic-sampling-filter-path examples.kiro_agent.cgs_sample_filter.cgs_filter
)

# =============================================================================
# Optimizer
# =============================================================================
OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.01
   --adam-beta1 0.9
   --adam-beta2 0.98

   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)

# =============================================================================
# Logging
# =============================================================================
WANDB_ARGS=(
   #--use-wandb
   # --wandb-project slime-kos-cgs
   # --wandb-group qwen3-30B-A3B-kos-cgs
   # --wandb-key ${WANDB_KEY}
)

TENSORBOARD_DIR=${TENSORBOARD_DIR:-"${LOG_DIR}/tensorboard/${RUN_NAME}"}
export TENSORBOARD_DIR

TENSORBOARD_ARGS=(
   --use-tensorboard
   --tb-project-name slime-kos-cgs
   --tb-experiment-name qwen3-30b-rl-cgs-async
)

MLFLOW_ARGS=(
   --use-mlflow
   --mlflow-experiment-name slime-kos-cgs-async
   --log-memory-to-mlflow
   --mlflow-run-name "qwen3-30b-kos-cgs-${RUN_NAME}"
)

# =============================================================================
# SGLang Configuration (External Rollout)
# =============================================================================
SGLANG_ARGS=(
   --rollout-external
   --rollout-external-engine-addrs ${SGLANG_ENGINE_ADDRS}
   --sglang-router-ip ${SGLANG_ROUTER_IP}
   --sglang-router-port ${SGLANG_ROUTER_PORT}
   --rollout-num-gpus-per-engine ${SGLANG_GPUS_PER_ENGINE}
)

# =============================================================================
# Async Training Configuration
# =============================================================================
UPDATE_WEIGHTS_INTERVAL=${UPDATE_WEIGHTS_INTERVAL:-1}

ASYNC_ARGS=(
   --update-weights-interval ${UPDATE_WEIGHTS_INTERVAL}
)

# =============================================================================
# Misc
# =============================================================================
MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --no-check-for-nan-in-loss-and-grad
)

# =============================================================================
# Custom Functions: CGS generate + reward
# =============================================================================
CUSTOM_ARGS=(
   --custom-generate-function-path examples.kiro_agent.kiro_generate_with_kos.generate
   --custom-rm-path examples.kiro_agent.kiro_generate_with_kos.cgs_reward_func
   --custom-reward-post-process-path examples.kiro_agent.cgs_reward_shaping.cgs_reward_post_process
)

# =============================================================================
# Ray Cluster Setup
# =============================================================================
export no_proxy="127.0.0.1,${MASTER_ADDR}"
export RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE=1

STARTED_RAY=false

if ray status &>/dev/null; then
    echo "Existing Ray cluster detected"
    ray status
else
    STARTED_RAY=true
    echo "Starting new Ray cluster..."

    if [[ "${MY_ADDR}" == "${MASTER_ADDR}" ]] || [[ "${MASTER_ADDR}" == "127.0.0.1" ]]; then
        echo "Starting Ray head node on ${MASTER_ADDR}..."
        ray start --head \
            --node-ip-address=${MASTER_ADDR} \
            --num-gpus=${NUM_GPUS_PER_NODE} \
            --object-store-memory=200000000000 \
            --dashboard-host=0.0.0.0 \
            --dashboard-port=8265 \
            --disable-usage-stats

        if [ -n "$WORKER_IPS" ]; then
            echo "Starting worker nodes..."
            for WORKER_IP in $WORKER_IPS; do
                echo "Starting Ray worker on ${WORKER_IP}..."
                ssh root@"${WORKER_IP}" \
                    "pkill -9 sglang 2>/dev/null || true; \
                     ray stop --force 2>/dev/null || true; \
                     pkill -9 python 2>/dev/null || true; \
                     export RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE=1; \
                     ray start --address=${MASTER_ADDR}:6379 \
                         --num-gpus=${NUM_GPUS_PER_NODE} \
                         --node-ip-address=${WORKER_IP} \
                         --object-store-memory=200000000000 \
                         --disable-usage-stats" &
            done
            wait
            echo "All worker nodes started"
        fi

        if [ "${NUM_NODES}" -gt 1 ]; then
            echo "Waiting for all ${NUM_NODES} nodes to join the cluster..."
            MAX_WAIT=2400
            WAITED=0
            while true; do
                count=$(ray status 2>/dev/null | grep -Eo '^ *1 node_[0-9a-f]+' | wc -l)
                if [ "$count" -ge "$NUM_NODES" ]; then
                    echo "All $count / $NUM_NODES nodes have joined."
                    break
                fi
                if [ "$WAITED" -ge "$MAX_WAIT" ]; then
                    echo "ERROR: Timeout waiting for nodes. Only $count / $NUM_NODES joined after ${MAX_WAIT}s."
                    ray status
                    exit 1
                fi
                echo "Waiting for $NUM_NODES nodes... (currently $count, ${WAITED}s elapsed)"
                sleep 5
                WAITED=$((WAITED + 5))
            done
        fi
    else
        echo "Worker node — waiting for head at ${MASTER_ADDR}:6379..."
        until ray health-check --address="${MASTER_ADDR}:6379" 2>/dev/null; do
            echo "Head node not ready yet, retrying in 5s..."
            sleep 5
        done
        echo "Head node is ready, joining cluster..."

        ray start --address=${MASTER_ADDR}:6379 \
            --num-gpus=${NUM_GPUS_PER_NODE} \
            --node-ip-address=${MY_ADDR} \
            --object-store-memory=200000000000 \
            --disable-usage-stats

        echo "Worker node joined. Sleeping to keep Ray alive (head runs training)."
        sleep infinity
    fi
fi

ray status
TOTAL_GPUS=$((NUM_NODES * NUM_GPUS_PER_NODE))
echo "Ray cluster ready with ${NUM_NODES} nodes, ${TOTAL_GPUS} total GPUs"

# =============================================================================
# Runtime Environment
# =============================================================================
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/:${SLIME_DIR}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"PYTORCH_ALLOC_CONF\": \"expandable_segments:True\",
    \"PYTORCH_CUDA_ALLOC_CONF\": \"expandable_segments:True\",
    \"no_proxy\": \"127.0.0.1,${MASTER_ADDR}\",
    \"MASTER_ADDR\": \"${MASTER_ADDR}\",
    \"NCCL_SOCKET_IFNAME\": \"^lo,docker0,veth_def_agent\",
    \"NCCL_DEBUG\": \"DEBUG\",
    \"CUDA_LAUNCH_BLOCKING\": \"0\",
    \"KOS_REMOTE_URLS\": \"${KOS_REMOTE_URLS}\",
    \"KOS_TIMEOUT\": \"${KOS_TIMEOUT}\",
    \"KOS_IS_CGS_TASK\": \"${KOS_IS_CGS_TASK}\",
    \"TRAJECTORY_FOLDER\": \"${TRAJECTORY_FOLDER}\",
    \"TENSORBOARD_DIR\": \"${TENSORBOARD_DIR}\",
    \"MLFLOW_TRACKING_URI\": \"${MLFLOW_TRACKING_URI:-}\",
    \"CGS_ZERO_GROUP_ON_ERROR\": \"${CGS_ZERO_GROUP_ON_ERROR}\",
    \"CGS_TOOL_CALL_INCENTIVE_ENABLE\": \"${CGS_TOOL_CALL_INCENTIVE_ENABLE}\",
    \"CGS_SOFT_TOOL_CALL_CAP\": \"${CGS_SOFT_TOOL_CALL_CAP}\",
    \"CGS_TOOL_CALL_LOG_ALPHA\": \"${CGS_TOOL_CALL_LOG_ALPHA}\",
    \"CGS_GAMMA_DECAY_ENABLE\": \"${CGS_GAMMA_DECAY_ENABLE}\",
    \"CGS_GAMMA_DECAY_COEF\": \"${CGS_GAMMA_DECAY_COEF}\",
    \"CGS_LENGTH_PENALTY_ENABLE\": \"${CGS_LENGTH_PENALTY_ENABLE}\",
    \"CGS_LENGTH_PENALTY_COEF\": \"${CGS_LENGTH_PENALTY_COEF}\",
    \"CGS_GRPO_STD_NORMALIZATION\": \"${CGS_GRPO_STD_NORMALIZATION}\",
    \"CGS_SCALE_ADV_BY_AVG_TC\": \"${CGS_SCALE_ADV_BY_AVG_TC}\",
    \"CGS_MAX_TC_PER_TURN_LIMIT\": \"${CGS_MAX_TC_PER_TURN_LIMIT}\",
    \"CGS_MIN_REWARD_THRESHOLD\": \"${CGS_MIN_REWARD_THRESHOLD}\"
  }
}"

# =============================================================================
# Start Async Training
# =============================================================================
LOG_DIR=${LOG_DIR:-"/mnt_out/yawenwuu/logs/slime"}
LOG_FILE="${LOG_DIR}/train_async_kos_cgs_${NUM_NODES}nodes_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$(dirname "$LOG_FILE")"
echo "Training log: $LOG_FILE"

echo "=============================================="
echo "Starting CGS ASYNC Training with External SGLang + Remote KoS Rollout:"
echo "  Training Nodes: ${NUM_NODES}"
echo "  Training GPUs per Node: ${NUM_GPUS_PER_NODE}"
echo "  External SGLang: ${SGLANG_ROUTER_IP}:${SGLANG_ROUTER_PORT}"
echo "  SGLang Engines: ${SGLANG_ENGINE_ADDRS}"
echo "  KoS Servers (CGS): ${KOS_REMOTE_URLS}"
echo "  Update Weights Interval: ${UPDATE_WEIGHTS_INTERVAL}"
echo "=============================================="

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 ${SLIME_DIR}/train_async.py \
   --actor-num-nodes ${NUM_NODES} \
   --actor-num-gpus-per-node ${NUM_GPUS_PER_NODE} \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${TENSORBOARD_ARGS[@]} \
   ${MLFLOW_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${ASYNC_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${CUSTOM_ARGS[@]} \
   2>&1 | tee "$LOG_FILE"

# =============================================================================
# Cleanup
# =============================================================================
if [ "$STARTED_RAY" = true ]; then
    echo "Training completed. Cleaning up Ray cluster..."
    sleep 3

    if [ -n "$WORKER_IPS" ]; then
        for WORKER_IP in $WORKER_IPS; do
            ssh root@"${WORKER_IP}" "ray stop --force; pkill -9 ray; pkill -9 python" 2>/dev/null &
        done
        wait
    fi

    ray stop --force
    pkill -9 ray 2>/dev/null || true
    pkill -9 python 2>/dev/null || true
    sleep 3
    pkill -9 ray 2>/dev/null || true
    pkill -9 python 2>/dev/null || true

    echo "Cleanup complete."
else
    echo "Training completed. Skipping cleanup (Ray was started externally)."
fi
