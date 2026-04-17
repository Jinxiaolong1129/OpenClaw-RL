#!/bin/bash
# =============================================================================
# SLIME Training with Remote Kiro-on-Strands (KoS) Rollout Server
# =============================================================================
#
# This script runs SLIME RL training using:
#   - An external SGLang server launched by launch_external_sglang.py
#     (for weight syncing and log-prob computation)
#   - A remote sglang_remote_rollout.py FastAPI server
#     (for agent rollout: Docker, Strands agent, trajectory formatting)
#
# Prerequisites:
#   1. Launch the external SGLang server:
#      python examples/kiro_agent/sglang_rollout_server/launch_external_sglang.py \
#          --model-path /path/to/Qwen3-Coder-30B-A3B-Instruct \
#          --tp-size 8
#      # This writes: /mnt_out/myshang/logs/slime/sglang_external_rollout.env
#
#   2. Start the KoS remote rollout server:
#      cd Kiro-on-Strands
#      python remote_rollout_server/sglang_remote_rollout.py \
#          --trajectory_folder /mnt_out/trajectories \
#          --sglang_log_folder /mnt_out/logs \
#          --workspace_path /mnt_out/workspace \
#          --model_config_name hostedsglang \
#          --is_swe_task
#
#   3. Run this script (KOS_REMOTE_URL defaults to http://<SGLANG_ROUTER_IP>:5000):
#      bash examples/kiro_agent/run-qwen3-30b-kiro-kos.sh
#
# =============================================================================

set -ex

export PYTHONBUFFERED=16

# =============================================================================
# External SGLang Server Configuration
# =============================================================================
# Source the env file written by launch_external_sglang.py
# Use RUN_NAME to isolate config files per job (matches --run-name in launch_external_sglang)
RUN_NAME=${RUN_NAME:-"smoke_test"}
SGLANG_ENV_BASE=${SGLANG_ENV_BASE:-"/mnt_out/myshang/logs/slime"}
SGLANG_ENV_FILE=${SGLANG_ENV_FILE:-"${SGLANG_ENV_BASE}/${RUN_NAME}/sglang_external_rollout.env"}

if [ -f "${SGLANG_ENV_FILE}" ]; then
    echo "Loading SGLang config from ${SGLANG_ENV_FILE}"
    source "${SGLANG_ENV_FILE}"
else
    echo "WARNING: ${SGLANG_ENV_FILE} not found."
    echo "Set SGLANG_ROUTER_IP, SGLANG_ROUTER_PORT, SGLANG_ENGINE_ADDRS,"
    echo "SGLANG_TOTAL_GPUS, SGLANG_GPUS_PER_ENGINE manually, or run"
    echo "launch_external_sglang.py with --run-name ${RUN_NAME} first."
fi

# These come from the env file (or can be overridden)
SGLANG_ROUTER_IP=${SGLANG_ROUTER_IP:?"ERROR: SGLANG_ROUTER_IP not set"}
SGLANG_ROUTER_PORT=${SGLANG_ROUTER_PORT:?"ERROR: SGLANG_ROUTER_PORT not set"}
SGLANG_ENGINE_ADDRS=${SGLANG_ENGINE_ADDRS:?"ERROR: SGLANG_ENGINE_ADDRS not set"}
SGLANG_TOTAL_GPUS=${SGLANG_TOTAL_GPUS:?"ERROR: SGLANG_TOTAL_GPUS not set"}
SGLANG_GPUS_PER_ENGINE=${SGLANG_GPUS_PER_ENGINE:?"ERROR: SGLANG_GPUS_PER_ENGINE not set"}

# =============================================================================
# KoS Remote Server Configuration
# =============================================================================
# KOS_REMOTE_URLS is a comma-separated list of distributed FastAPI server URLs.
# Each node running launch_sglang_and_kos.sh starts a KoS FastAPI server on
# KOS_PORT (default 5000). All servers share the same SGLang router for LLM
# inference, but handle Docker container execution locally.
#
# If not explicitly set, we auto-discover server URLs from SGLANG_ENGINE_ADDRS.
# SGLANG_ENGINE_ADDRS contains space-separated "IP:port" entries for each
# sglang engine. We extract unique IPs (a node may host multiple engines)
# and build KoS URLs from those IPs + KOS_PORT.
KOS_PORT=${KOS_PORT:-5000}

if [ -z "${KOS_REMOTE_URLS:-}" ]; then
    # Auto-build URL list from unique node IPs in SGLANG_ENGINE_ADDRS
    _SEEN_IPS=""
    _KOS_URLS=""
    for _ADDR in ${SGLANG_ENGINE_ADDRS}; do
        _IP=$(echo "${_ADDR}" | cut -d: -f1)
        # Deduplicate: skip if we've already seen this IP
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
else
    export KOS_REMOTE_URLS="${KOS_REMOTE_URLS}"
fi

export KOS_TIMEOUT=${KOS_TIMEOUT:-3600}
export TRAJECTORY_FOLDER=${TRAJECTORY_FOLDER:-"/mnt_out/trajectories"}

# =============================================================================
# Cluster Configuration
# =============================================================================
SLIME_DIR=${SLIME_DIR:-"/mnt_out/myshang/codebase/slime"}
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
echo "Cluster Configuration:"
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
echo "KoS Remote Servers:"
echo "  KOS_REMOTE_URLS: ${KOS_REMOTE_URLS}"
echo "=============================================="

# =============================================================================
# Model Checkpoints
# =============================================================================
CKPT_ARGS=(
   --hf-checkpoint /mnt_out/myshang/models/qwen/Qwen3-Coder-30B-A3B-Instruct
   --ref-load /mnt_out/myshang/models/qwen/Qwen3-Coder-30B-A3B-Instruct-mcore
   --load /mnt_out/myshang/models/qwen/kiro_rl/Qwen3-30B_slime_kos/
   --save /mnt_out/myshang/models/qwen/kiro_rl/Qwen3-30B_slime_kos/
   --save-interval 20
)

# =============================================================================
# Rollout Configuration
# =============================================================================
# The external SGLang server (launched by launch_external_sglang.py) handles
# inference. The custom generate function delegates agent rollout to the
# remote KoS server via HTTP. SLIME uses --rollout-external to connect to
# the pre-launched SGLang engines for weight syncing and log-prob computation.
ROLLOUT_ARGS=(
   --data-source-path examples.kiro_agent.custom_data_source.AgentDataSource
   --prompt-data /mnt_out/zhenghuj/data/preprocessed_sweap_582_v8.0_train.parquet
   --input-key problem_statement
   --label-key patch
   --apply-chat-template
   --rollout-shuffle
   --num-rollout 330
   --rollout-batch-size 16
   --n-samples-per-prompt 16
   --rollout-max-response-len 117888
   --rollout-temperature 1

   --global-batch-size 64
   --balance-data
)

# =============================================================================
# Evaluation
# =============================================================================
EVAL_ARGS=(
   --eval-interval 200
   --eval-prompt-data sweap_val /mnt_out/zhenghuj/data/preprocessed_sweap_12_v8.0_val.parquet
   --n-samples-per-eval-prompt 16
   --eval-max-response-len 117888
   --eval-top-p 1
)

# =============================================================================
# Performance / Parallelism
# =============================================================================
PERF_ARGS=(
   --tensor-model-parallel-size 4
   --sequence-parallel
   --pipeline-model-parallel-size 2
   --context-parallel-size 4
   --expert-model-parallel-size 8
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu 8192
)

# =============================================================================
# GRPO
# =============================================================================
GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

# =============================================================================
# Optimizer
# =============================================================================
OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98

   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)

# =============================================================================
# Wandb (uncomment to enable)
# =============================================================================
WANDB_ARGS=(
   #--use-wandb
   # --wandb-project slime-kos
   # --wandb-group qwen3-30B-A3B-kos
   # --wandb-key ${WANDB_KEY}
)

TENSORBOARD_DIR=${TENSORBOARD_DIR:-"${LOG_DIR}/tensorboard/${RUN_NAME}"}
export TENSORBOARD_DIR

TENSORBOARD_ARGS=(
   --use-tensorboard
   --tb-project-name slime-smoke-test
   --tb-experiment-name qwen3-30b-rl
)

# =============================================================================
# MLflow (uses MLFLOW_TRACKING_URI env var if --mlflow-tracking-uri not set)
# =============================================================================
MLFLOW_ARGS=(
   --use-mlflow
   --mlflow-experiment-name slime-kos-smoke-test
   --mlflow-run-name "qwen3-30b-kos-${RUN_NAME}"
)

# =============================================================================
# SGLang Configuration (External Rollout)
# =============================================================================
# Use --rollout-external to connect to the pre-launched SGLang engines from
# launch_external_sglang.py. SLIME won't launch its own SGLang servers —
# it connects to the external ones for weight syncing and log-prob computation.
#
# With --rollout-external (no --colocate), the placement group only allocates
# training GPUs. Lightweight rollout proxy actors share those GPUs for
# weight-update IPC. The actual SGLang engines run on separate nodes with
# their own GPUs, so no offload/onload cycle is needed.
SGLANG_ARGS=(
   --rollout-external
   --rollout-external-engine-addrs ${SGLANG_ENGINE_ADDRS}
   --sglang-router-ip ${SGLANG_ROUTER_IP}
   --sglang-router-port ${SGLANG_ROUTER_PORT}
   --rollout-num-gpus-per-engine ${SGLANG_GPUS_PER_ENGINE}
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
)

# =============================================================================
# Custom Functions: KoS generate + reward
# =============================================================================
CUSTOM_ARGS=(
   --custom-generate-function-path examples.kiro_agent.kiro_generate_with_kos.generate
   --custom-rm-path examples.kiro_agent.kiro_generate_with_kos.reward_func
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
            # Manual multi-node: start workers via SSH
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

        # Wait for all nodes to join the cluster (PyTorchJob or manual)
        if [ "${NUM_NODES}" -gt 1 ]; then
            echo "Waiting for all ${NUM_NODES} nodes to join the cluster..."
            MAX_WAIT=300
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
        # Worker node: wait for head, join cluster, then sleep forever
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
    \"no_proxy\": \"127.0.0.1,${MASTER_ADDR}\",
    \"MASTER_ADDR\": \"${MASTER_ADDR}\",
    \"NCCL_SOCKET_IFNAME\": \"^lo,docker0,veth_def_agent\",
    \"NCCL_DEBUG\": \"INFO\",
    \"CUDA_LAUNCH_BLOCKING\": \"0\",
    \"KOS_REMOTE_URLS\": \"${KOS_REMOTE_URLS}\",
    \"KOS_TIMEOUT\": \"${KOS_TIMEOUT}\",
    \"TRAJECTORY_FOLDER\": \"${TRAJECTORY_FOLDER}\",
    \"TENSORBOARD_DIR\": \"${TENSORBOARD_DIR}\",
    \"MLFLOW_TRACKING_URI\": \"${MLFLOW_TRACKING_URI:-}\"
  }
}"

# =============================================================================
# Start Training
# =============================================================================
LOG_DIR=${LOG_DIR:-"/mnt_out/myshang/logs/slime"}
LOG_FILE="${LOG_DIR}/train_kos_${NUM_NODES}nodes_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$(dirname "$LOG_FILE")"
echo "Training log: $LOG_FILE"

echo "=============================================="
echo "Starting Training with External SGLang + Remote KoS Rollout:"
echo "  Training Nodes: ${NUM_NODES}"
echo "  Training GPUs per Node: ${NUM_GPUS_PER_NODE}"
echo "  External SGLang: ${SGLANG_ROUTER_IP}:${SGLANG_ROUTER_PORT}"
echo "  SGLang Engines: ${SGLANG_ENGINE_ADDRS}"
echo "  KoS Servers: ${KOS_REMOTE_URLS}"
echo "=============================================="

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 ${SLIME_DIR}/train.py \
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
