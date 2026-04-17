#!/bin/bash

set -ex

# will prevent ray from buffering stdout/stderr
export PYTHONBUFFERED=16

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

# SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
SLIME_DIR=${SLIME_DIR:-"/mnt_out/myshang/codebase/slime"}
SCRIPT_DIR=${SLIME_DIR}/scripts
source "${SCRIPT_DIR}/models/qwen3-30B-A3B.sh"

export NUM_NODES=${NUM_NODES:-1}
export NUM_GPUS_PER_NODE=${NUM_GPUS_PER_NODE:-8}

if [ -n "$PET_NNODES" ]; then
    # Running in HyperPod/Kubernetes environment
    export NUM_NODES=${PET_NNODES}
    export MASTER_ADDR=${MASTER_ADDR:-$(hostname | sed -E 's/-worker-[0-9]+$/-worker-0/')}
    export MY_ADDR=$(hostname)
else
    # Running manually
    export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
    export MY_ADDR=${MY_ADDR:-"127.0.0.1"}
fi

echo "=============================================="
echo "Cluster Configuration:"
echo "  NUM_NODES: ${NUM_NODES}"
echo "  NUM_GPUS_PER_NODE: ${NUM_GPUS_PER_NODE}"
echo "  MASTER_ADDR: ${MASTER_ADDR}"
echo "  MY_ADDR: ${MY_ADDR}"
echo "=============================================="

export SWE_DOCKER_IMAGES_PATH=
export WORKSPACE_BASE_PATH=
export TRAJECTORY_FOLDER=


CKPT_ARGS=(
   --hf-checkpoint /mnt_out/myshang/models/qwen/Qwen3-Coder-30B-A3B-Instruct
   --ref-load /mnt_out/myshang/models/qwen/Qwen3-Coder-30B-A3B-Instruct-mcore
   --load /mnt_out/myshang/models/qwen/kiro_rl/Qwen3-30B_slime/
   --save /mnt_out/myshang/models/qwen/kiro_rl/Qwen3-30B_slime/
   --save-interval 20
)

ROLLOUT_ARGS=(
   --data-source-path examples.kiro_agent.custom_data_source.AgentDataSource
   --prompt-data /mnt_out/zhenghuj/data/preprocessed_sweap_582_v8.0_train.parquet
   --input-key problem_statement
   --label-key patch
   --apply-chat-template
   --rollout-shuffle
   --num-rollout 3000
   --rollout-batch-size 8
   --n-samples-per-prompt 2
   --rollout-max-response-len 16384
   --rollout-temperature 1

   --global-batch-size 4
   --balance-data
)

EVAL_ARGS=(
   --eval-interval 200
   --eval-prompt-data aime /mnt_out/zhenghuj/data/preprocessed_sweap_12_v8.0_val.parquet
   --n-samples-per-eval-prompt 16
   --eval-max-response-len 16384
   --eval-top-p 1
)

PERF_ARGS=(
   --tensor-model-parallel-size 4
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 8
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   # --micro-batch-size 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 20480
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

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

WANDB_ARGS=(
   #--use-wandb
   # --wandb-project slime-dev
   # --wandb-group qwen3-30B-A3B-test
   # --wandb-key ${WANDB_KEY}
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 8
   --sglang-mem-fraction-static 0.7
   --sglang-cuda-graph-bs 1 2 4 8 $(seq 16 8 256)
)

MISC_ARGS=(
   # default dropout in megatron is 0.1
   --attention-dropout 0.0
   --hidden-dropout 0.0
   # should be good for model performance
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   # need to comment this when using model with MLA
   --attention-backend flash
)

CUSTOM_ARGS=(
   --custom-generate-function-path examples.kiro_agent.kiro_generate.generate
   --custom-rm-path examples.kiro_agent.kiro_generate.generate.reward_func
)


# =============================================================================
# Ray Cluster Setup (skipped if Ray already running, e.g., from K8s YAML)
# =============================================================================
export no_proxy="127.0.0.1,${MASTER_ADDR}"
export RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE=1

# Track if we started Ray (for cleanup decision)
STARTED_RAY=false

if ray status &>/dev/null; then
    echo "Existing Ray cluster detected (likely started by K8s job)"
    ray status
else
    STARTED_RAY=true
    echo "Starting new Ray cluster..."
    
    # Determine if this is the head node
    if [[ "${MY_ADDR}" == "${MASTER_ADDR}" ]] || [[ "${MASTER_ADDR}" == "127.0.0.1" ]]; then
        echo "Starting Ray head node on ${MASTER_ADDR}..."
        ray start --head \
            --node-ip-address=${MASTER_ADDR} \
            --num-gpus=${NUM_GPUS_PER_NODE} \
            --object-store-memory=200000000000 \
            --dashboard-host=0.0.0.0 \
            --dashboard-port=8265 \
            --disable-usage-stats
        
        # Start worker nodes if WORKER_IPS is set (manual multi-node setup)
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
            
            # Wait for all nodes to join
            echo "Waiting for all nodes to join the cluster..."
            while true; do
                count=$(ray status 2>/dev/null | grep -Eo '^ *1 node_[0-9a-f]+' | wc -l)
                if [ "$count" -ge "$NUM_NODES" ]; then
                    echo "All $count / $NUM_NODES nodes have joined."
                    break
                fi
                echo "Waiting for $NUM_NODES nodes... (currently $count)"
                sleep 5
            done
        fi
    else
        echo "This is a worker node, joining cluster at ${MASTER_ADDR}:6379..."
        ray start --address=${MASTER_ADDR}:6379 \
            --num-gpus=${NUM_GPUS_PER_NODE} \
            --node-ip-address=${MY_ADDR} \
            --object-store-memory=200000000000 \
            --disable-usage-stats
        
        echo "Worker node joined. Exiting (head node will run training)."
        exit 0
    fi
fi

ray status
TOTAL_GPUS=$((NUM_NODES * NUM_GPUS_PER_NODE))
echo "Ray cluster ready with ${NUM_NODES} nodes, ${TOTAL_GPUS} total GPUs"

# =============================================================================
# Build Runtime Environment
# =============================================================================
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/:${SLIME_DIR}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"no_proxy\": \"127.0.0.1,${MASTER_ADDR}\",
    \"MASTER_ADDR\": \"${MASTER_ADDR}\",
    \"NCCL_SOCKET_IFNAME\": \"^lo,docker0,veth_def_agent\",
    \"NCCL_DEBUG\": \"INFO\",
    \"CUDA_LAUNCH_BLOCKING\": \"0\"
  }
}"

# =============================================================================
# Create Log Directory and Start Training
# =============================================================================
LOG_FILE="${LOG_DIR}/train_${NUM_NODES}nodes_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$(dirname "$LOG_FILE")"
echo "Training log: $LOG_FILE"

echo "=============================================="
echo "Starting Multi-Node Training:"
echo "  Nodes: ${NUM_NODES}"
echo "  Actor GPUs per node: ${ACTOR_GPUS_PER_NODE}"
echo "  Rollout GPUs per node: ${ROLLOUT_GPUS_PER_NODE}"
echo "  Total Actor GPUs: ${TOTAL_ACTOR_GPUS}"
echo "  Total Rollout GPUs: ${TOTAL_ROLLOUT_GPUS}"
echo "  Tensor Parallelism: ${TP_SIZE}"
echo "  Rollout Batch Size: ${ROLLOUT_BATCH_SIZE}"
echo "  Global Batch Size: ${GLOBAL_BATCH_SIZE}"
echo "=============================================="


ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 ${SLIME_DIR}/train.py \
   --actor-num-nodes 2 \
   --actor-num-gpus-per-node 8 \
   --colocate \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${CUSTOM_ARGS[@]} \
   2>&1 | tee "$LOG_FILE"


# =============================================================================
# Cleanup (only if we started Ray ourselves)
# =============================================================================
if [ "$STARTED_RAY" = true ]; then
    echo "Training completed. Cleaning up Ray cluster we started..."
    sleep 3

    # Cleanup worker nodes
    if [ -n "$WORKER_IPS" ]; then
        for WORKER_IP in $WORKER_IPS; do
            echo "Cleaning up worker node ${WORKER_IP}..."
            ssh root@"${WORKER_IP}" "ray stop --force; pkill -9 ray; pkill -9 python" 2>/dev/null &
        done
        wait
    fi

    # Cleanup head node
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