#!/bin/bash
# =============================================================================
# BIPOD: On-Policy Distillation for Kiro Agent (Top-K Logits KL Loss)
# =============================================================================
#
# Runs the KoS agent for multi-turn trajectory generation, queries an external
# teacher SGLang server for top-K logprobs, then trains the student using
# reverse KL distillation (no GRPO / policy gradient).
#
# Architecture:
#   [KoS agent] -> [Student SGLang (external)] -> trajectories
#                -> [Teacher SGLang (external)] -> top-K logprobs
#                -> [Slime Megatron trainer]    -> KL distillation loss
#
# Prerequisites:
#   1. Student SGLang GPU job running (same as kiro_agent setup)
#   2. Teacher SGLang GPU job running (larger model, separate nodes)
#   3. KoS CPU job running (same as kiro_agent setup)
#   4. CGS training data parquet at PROMPT_DATA path
# =============================================================================

set -ex

export PYTHONBUFFERED=16

# =============================================================================
# External SGLang Server Configuration (Student)
# =============================================================================
RUN_NAME=${RUN_NAME:-"bipod_async"}
SGLANG_RUN_NAME=${SGLANG_RUN_NAME:-"sglang_server_bipod"}
SGLANG_ENV_BASE=${SGLANG_ENV_BASE:-"/mnt_out/yawenwuu/logs/slime"}
SGLANG_ENV_FILE=${SGLANG_ENV_FILE:-"${SGLANG_ENV_BASE}/${SGLANG_RUN_NAME}/sglang_external_rollout.env"}

if [ -f "${SGLANG_ENV_FILE}" ]; then
    echo "Loading SGLang config from ${SGLANG_ENV_FILE}"
    source "${SGLANG_ENV_FILE}"
else
    echo "WARNING: ${SGLANG_ENV_FILE} not found."
    echo "Set SGLANG_ROUTER_IP, SGLANG_ROUTER_PORT, SGLANG_ENGINE_ADDRS,"
    echo "SGLANG_TOTAL_GPUS, SGLANG_GPUS_PER_ENGINE manually."
fi

SGLANG_ROUTER_IP=${SGLANG_ROUTER_IP:?"ERROR: SGLANG_ROUTER_IP not set"}
SGLANG_ROUTER_PORT=${SGLANG_ROUTER_PORT:?"ERROR: SGLANG_ROUTER_PORT not set"}
SGLANG_ENGINE_ADDRS=${SGLANG_ENGINE_ADDRS:?"ERROR: SGLANG_ENGINE_ADDRS not set"}
SGLANG_TOTAL_GPUS=${SGLANG_TOTAL_GPUS:?"ERROR: SGLANG_TOTAL_GPUS not set"}
SGLANG_GPUS_PER_ENGINE=${SGLANG_GPUS_PER_ENGINE:?"ERROR: SGLANG_GPUS_PER_ENGINE not set"}

# =============================================================================
# External Teacher SGLang Server Configuration
# =============================================================================
# The teacher model is hosted on a separate SGLang server (typically larger model).
# Launch it independently, e.g.:
#   python -m sglang.launch_server --model-path Qwen/Qwen3-235B-A22B-Instruct \
#       --tp 8 --host 0.0.0.0 --port 31000 --mem-fraction-static 0.85
TEACHER_SGLANG_RUN_NAME=${TEACHER_SGLANG_RUN_NAME:-"sglang_teacher"}
TEACHER_SGLANG_ENV_FILE=${TEACHER_SGLANG_ENV_FILE:-"${SGLANG_ENV_BASE}/${TEACHER_SGLANG_RUN_NAME}/sglang_external_rollout.env"}

if [ -f "${TEACHER_SGLANG_ENV_FILE}" ]; then
    echo "Loading Teacher SGLang config from ${TEACHER_SGLANG_ENV_FILE}"
    # Source into a subshell-safe prefix to avoid clobbering student vars
    TEACHER_ROUTER_IP=$(source "${TEACHER_SGLANG_ENV_FILE}" && echo "${SGLANG_ROUTER_IP}")
    TEACHER_ROUTER_PORT=$(source "${TEACHER_SGLANG_ENV_FILE}" && echo "${SGLANG_ROUTER_PORT}")
    TEACHER_URL=${TEACHER_URL:-"http://${TEACHER_ROUTER_IP}:${TEACHER_ROUTER_PORT}/generate"}
else
    echo "WARNING: ${TEACHER_SGLANG_ENV_FILE} not found."
    echo "Set TEACHER_URL manually (e.g. http://teacher-host:31000/generate)"
fi
TEACHER_URL=${TEACHER_URL:?"ERROR: TEACHER_URL not set (e.g. http://teacher-host:31000/generate)"}
TEACHER_TOPK=${TEACHER_TOPK:-50}
TEACHER_MAX_CONCURRENCY=${TEACHER_MAX_CONCURRENCY:-8}

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

# CGS task mode (same as kiro_agent)
export KOS_IS_CGS_TASK=true

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
echo "BIPOD Async Training Configuration:"
echo "  NUM_NODES: ${NUM_NODES}"
echo "  NUM_GPUS_PER_NODE: ${NUM_GPUS_PER_NODE}"
echo "  MASTER_ADDR: ${MASTER_ADDR}"
echo "  MY_ADDR: ${MY_ADDR}"
echo "External SGLang (Student):"
echo "  SGLANG_ROUTER_IP: ${SGLANG_ROUTER_IP}"
echo "  SGLANG_ROUTER_PORT: ${SGLANG_ROUTER_PORT}"
echo "External Teacher:"
echo "  TEACHER_URL: ${TEACHER_URL}"
echo "  TEACHER_TOPK: ${TEACHER_TOPK}"
echo "KoS Remote Servers:"
echo "  KOS_REMOTE_URLS: ${KOS_REMOTE_URLS}"
echo "=============================================="

# =============================================================================
# Model Checkpoints
# =============================================================================
CKPT_ARGS=(
   --hf-checkpoint /mnt_out/yawenwuu/models/qwen/Qwen3-Coder-30B-A3B-Instruct
   --ref-load /mnt_out/yawenwuu/models/qwen/Qwen3-Coder-30B-A3B-Instruct-mcore
   --load /mnt_out/yawenwuu/models/qwen/kiro_rl/Qwen3-30B_slime_bipod/${RUN_NAME}
   --save /mnt_out/yawenwuu/models/qwen/kiro_rl/Qwen3-30B_slime_bipod/${RUN_NAME}
   --save-interval 1
)

# =============================================================================
# Rollout Configuration
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
   --n-samples-per-prompt 1
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
# BIPOD: Top-K Distillation Loss (no GRPO / policy gradient)
# =============================================================================
BIPOD_ARGS=(
   --loss-type custom_loss
   --custom-loss-function-path examples.kiro_bipod.topk_distillation_loss.topk_distillation_loss_function
   --distill-topk ${TEACHER_TOPK}
   --disable-compute-advantages-and-returns
   --entropy-coef 0.0
   --calculate-per-token-loss
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
TENSORBOARD_DIR=${TENSORBOARD_DIR:-"${LOG_DIR}/tensorboard/${RUN_NAME}"}
export TENSORBOARD_DIR

TENSORBOARD_ARGS=(
   --use-tensorboard
   --tb-project-name slime-kos-bipod
   --tb-experiment-name qwen3-30b-bipod-async
)

MLFLOW_ARGS=(
   --use-mlflow
   --mlflow-experiment-name slime-kos-bipod-async
   --log-memory-to-mlflow
   --mlflow-run-name "qwen3-30b-bipod-${RUN_NAME}"
)

# =============================================================================
# SGLang Configuration (External Rollout - Student)
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
# Custom Functions: BIPOD generate (KoS + teacher query) + reward
# =============================================================================
CUSTOM_ARGS=(
   --custom-generate-function-path examples.kiro_bipod.bipod_generate.generate
   --custom-rm-path examples.kiro_bipod.bipod_generate.reward_func
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
    \"TEACHER_URL\": \"${TEACHER_URL}\",
    \"TEACHER_TOPK\": \"${TEACHER_TOPK}\",
    \"TEACHER_MAX_CONCURRENCY\": \"${TEACHER_MAX_CONCURRENCY}\"
  }
}"

# =============================================================================
# Start Async Training
# =============================================================================
LOG_DIR=${LOG_DIR:-"/mnt_out/yawenwuu/logs/slime"}
LOG_FILE="${LOG_DIR}/train_bipod_${NUM_NODES}nodes_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$(dirname "$LOG_FILE")"
echo "Training log: $LOG_FILE"

echo "=============================================="
echo "Starting BIPOD ASYNC Training:"
echo "  Training Nodes: ${NUM_NODES}"
echo "  Training GPUs per Node: ${NUM_GPUS_PER_NODE}"
echo "  Student SGLang: ${SGLANG_ROUTER_IP}:${SGLANG_ROUTER_PORT}"
echo "  Teacher SGLang: ${TEACHER_URL}"
echo "  KoS Servers: ${KOS_REMOTE_URLS}"
echo "  Distill Top-K: ${TEACHER_TOPK}"
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
   ${BIPOD_ARGS[@]} \
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
