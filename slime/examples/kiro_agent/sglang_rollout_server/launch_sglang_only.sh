#!/bin/bash
# =============================================================================
# SGLang-Only Server Launcher (GPU Nodes)
# =============================================================================
#
# Separated from launch_sglang_and_kos.sh for disaggregated deployment:
#   - This script runs on GPU nodes: Ray cluster + SGLang engines + router
#   - launch_kos_only.sh runs on CPU nodes: KoS agent servers
#
# Architecture:
#   ┌──────────────────────────────────────────────────────────────────┐
#   │  GPU Head Node                                                   │
#   │  ┌─────────────┐  ┌─────────────┐                               │
#   │  │ SGLang      │  │ SGLang      │                               │
#   │  │ Router      │  │ Engine 0    │                               │
#   │  │ (port 30000)│  │ (Ray Actor) │                               │
#   │  └─────────────┘  └─────────────┘                               │
#   └──────────────────────────────────────────────────────────────────┘
#   ┌──────────────────────────────────────────────────────────────────┐
#   │  GPU Worker Node N                                               │
#   │  ┌─────────────┐                                                 │
#   │  │ SGLang      │                                                 │
#   │  │ Engine N    │                                                 │
#   │  │ (Ray Actor) │                                                 │
#   │  └─────────────┘                                                 │
#   └──────────────────────────────────────────────────────────────────┘
#
# Outputs:
#   - sglang_external_rollout.env  (router IP/port, engine addrs)
#   - kos_servers.env              (placeholder — written by launch_kos_only.sh)
#
# Usage:
#   # HyperPod (auto-detects from PET_NNODES):
#   bash launch_sglang_only.sh --model-path /path/to/model
#
#   # Manual multi-node:
#   MASTER_ADDR=<head_ip> WORKER_IPS="<w1> <w2>" \
#       bash launch_sglang_only.sh --model-path /path/to/model
#
# =============================================================================

set -e

# =============================================================================
# Default Configuration
# =============================================================================

MODEL_PATH=""
NUM_ENGINES=0  # 0 = auto from cluster GPUs
TP_SIZE=8
DP_SIZE=1
MEM_FRACTION=0.6
CHUNKED_PREFILL_SIZE=-1
ROUTER_PORT=30000
SERVER_BASE_PORT=13140

# --- General ---
RUN_NAME=${RUN_NAME:-"default"}
OUTPUT_DIR=${OUTPUT_DIR:-"/mnt_out/myshang/logs/slime"}
SLIME_DIR=${SLIME_DIR:-"/mnt_out/myshang/codebase/slime"}
RAY_ADDRESS=""

# --- Cluster ---
NUM_GPUS_PER_NODE=${NUM_GPUS_PER_NODE:-$(nvidia-smi -L 2>/dev/null | wc -l || echo 8)}
MASTER_ADDR=${MASTER_ADDR:-""}
WORKER_IPS=${WORKER_IPS:-""}

# =============================================================================
# Parse Arguments
# =============================================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --model-path)        MODEL_PATH="$2";        shift 2 ;;
        --num-engines)       NUM_ENGINES="$2";       shift 2 ;;
        --tp-size)           TP_SIZE="$2";           shift 2 ;;
        --dp-size)           DP_SIZE="$2";           shift 2 ;;
        --mem-fraction)      MEM_FRACTION="$2";      shift 2 ;;
        --chunked-prefill-size) CHUNKED_PREFILL_SIZE="$2"; shift 2 ;;
        --router-port)       ROUTER_PORT="$2";       shift 2 ;;
        --server-base-port)  SERVER_BASE_PORT="$2";  shift 2 ;;
        --num-gpus-per-node) NUM_GPUS_PER_NODE="$2"; shift 2 ;;
        --ray-address)       RAY_ADDRESS="$2";       shift 2 ;;
        --master-addr)       MASTER_ADDR="$2";       shift 2 ;;
        --worker-ips)        WORKER_IPS="$2";        shift 2 ;;
        --run-name)          RUN_NAME="$2";          shift 2 ;;
        --output-dir)        OUTPUT_DIR="$2";        shift 2 ;;
        --slime-dir)         SLIME_DIR="$2";         shift 2 ;;
        --help)
            echo "Usage: $0 --model-path <path> [options]"
            echo ""
            echo "SGLang Options:"
            echo "  --model-path PATH           HuggingFace model path (required)"
            echo "  --num-engines N             Number of engines (default: auto)"
            echo "  --tp-size N                 Tensor parallel size (default: 8)"
            echo "  --dp-size N                 Data parallel size (default: 1)"
            echo "  --mem-fraction F            GPU memory fraction (default: 0.6)"
            echo "  --chunked-prefill-size N    Chunked prefill size (default: -1, disabled)"
            echo "  --router-port PORT          Router port (default: 30000)"
            echo "  --server-base-port PORT     Engine base port (default: 13140)"
            echo "  --ray-address ADDR          Existing Ray cluster address"
            echo "  --master-addr ADDR          Head node IP"
            echo "  --worker-ips IPS            Space-separated worker IPs"
            echo ""
            echo "General Options:"
            echo "  --run-name NAME             Run name for isolation (default: default)"
            echo "  --output-dir DIR            Output directory"
            echo "  --slime-dir DIR             SLIME codebase path"
            exit 0
            ;;
        *)
            echo "Unknown option: $1 (use --help)"
            exit 1
            ;;
    esac
done

if [ -z "$MODEL_PATH" ]; then
    echo "ERROR: --model-path is required"; exit 1
fi

# =============================================================================
# Derived Paths
# =============================================================================
SCRIPT_DIR=${SLIME_DIR}/examples/kiro_agent/sglang_rollout_server
RUN_DIR="${OUTPUT_DIR}/kos/${RUN_NAME}"
ENV_DIR="${OUTPUT_DIR}"
if [ -n "$RUN_NAME" ] && [ "$RUN_NAME" != "default" ]; then
    ENV_DIR="${OUTPUT_DIR}/${RUN_NAME}"
fi
SGLANG_ENV_FILE="${ENV_DIR}/sglang_external_rollout.env"

# =============================================================================
# Detect Cluster Topology
# =============================================================================
if [ -n "${PET_NNODES:-}" ]; then
    echo "Detected HyperPod/PyTorchJob (PET_NNODES=${PET_NNODES})"
    WORLD_SIZE=${PET_NNODES}
    if [ -z "$MASTER_ADDR" ]; then
        MASTER_ADDR=$(hostname | sed -E 's/-worker-[0-9]+$/-worker-0/')
    fi
    MY_ADDR=$(hostname)
    if [[ "${MASTER_ADDR}" == "${MY_ADDR}" ]]; then
        IS_MASTER=true
    else
        IS_MASTER=false
    fi
else
    IS_MASTER=true
    if [ -z "$MASTER_ADDR" ]; then
        MASTER_ADDR=$(hostname -I | awk '{print $1}')
    fi
    WORKER_COUNT=0
    if [ -n "$WORKER_IPS" ]; then
        WORKER_COUNT=$(echo "$WORKER_IPS" | wc -w)
    fi
    WORLD_SIZE=$((1 + WORKER_COUNT))
fi

GPUS_PER_ENGINE=$((TP_SIZE * DP_SIZE))
EXPECTED_GPUS=$((WORLD_SIZE * NUM_GPUS_PER_NODE))
if [ "$NUM_ENGINES" -le 0 ]; then
    NUM_ENGINES=$((EXPECTED_GPUS / GPUS_PER_ENGINE))
fi

echo "=============================================="
echo "SGLang-Only Launcher (GPU Nodes)"
echo "=============================================="
echo "Cluster:           ${WORLD_SIZE} nodes × ${NUM_GPUS_PER_NODE} GPUs"
echo "SGLang Engines:    ${NUM_ENGINES} (TP=${TP_SIZE}, DP=${DP_SIZE})"
echo "Model:             ${MODEL_PATH}"
echo "Run Name:          ${RUN_NAME}"
echo "Env File:          ${SGLANG_ENV_FILE}"
echo "=============================================="

# =============================================================================
# Worker Node Path: Join Ray cluster and wait
# =============================================================================
if [ "$IS_MASTER" = false ]; then
    echo "[Worker] Waiting for head node at ${MASTER_ADDR}:6379..."
    MAX_WAIT=2400
    WAITED=0
    until ray health-check --address="${MASTER_ADDR}:6379" 2>/dev/null; do
        if [ "$WAITED" -ge "$MAX_WAIT" ]; then
            echo "ERROR: Timeout waiting for head node"; exit 1
        fi
        sleep 5
        WAITED=$((WAITED + 5))
    done

    ray start \
        --address="${MASTER_ADDR}:6379" \
        --node-ip-address="$(hostname -I | awk '{print $1}')" \
        --num-gpus="$NUM_GPUS_PER_NODE" \
        --object-store-memory=200000000000 \
        --disable-usage-stats

    echo "[Worker] Joined Ray cluster. Sleeping to keep node alive for SGLang engines..."

    # Worker nodes just need to stay alive — SGLang engines run as Ray actors.
    # Wait for the env file as a signal that everything is healthy, then sleep.
    rm -f "${SGLANG_ENV_FILE}"
    MAX_WAIT=900
    WAITED=0
    while [ ! -f "${SGLANG_ENV_FILE}" ]; do
        if [ "$WAITED" -ge "$MAX_WAIT" ]; then
            echo "ERROR: Timeout waiting for ${SGLANG_ENV_FILE}"; exit 1
        fi
        sleep 10
        WAITED=$((WAITED + 10))
        echo "  Waiting for SGLang... (${WAITED}s)"
    done
    source "${SGLANG_ENV_FILE}"
    echo "[Worker] SGLang ready: router=${SGLANG_ROUTER_IP}:${SGLANG_ROUTER_PORT}"
    echo "[Worker] Keeping alive for Ray actors..."

    # Sleep forever — Ray actors run in the background
    sleep infinity
    exit 0
fi

# =============================================================================
# Head Node Path: Start Ray cluster, launch SGLang
# =============================================================================

# --- Step 1: Start Ray Cluster ---
if [ -z "$RAY_ADDRESS" ]; then
    echo ""
    echo "[Step 1/3] Setting up Ray cluster..."

    ray stop --force 2>/dev/null || true
    sleep 2
    export RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE=1

    ray start --head \
        --node-ip-address="$MASTER_ADDR" \
        --num-gpus="$NUM_GPUS_PER_NODE" \
        --object-store-memory=200000000000 \
        --disable-usage-stats \
        --dashboard-host=0.0.0.0 \
        --dashboard-port=8265

    # Start workers via SSH (manual mode)
    if [ -n "$WORKER_IPS" ]; then
        echo "Starting Ray workers via SSH..."
        for WORKER_IP in $WORKER_IPS; do
            echo "  - Worker: $WORKER_IP"
            ssh -o StrictHostKeyChecking=no "$WORKER_IP" "
                ray stop --force 2>/dev/null || true
                sleep 2
                export RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE=1
                ray start --address=${MASTER_ADDR}:6379 \
                    --num-gpus=${NUM_GPUS_PER_NODE} \
                    --node-ip-address=\$(hostname -I | awk '{print \$1}') \
                    --object-store-memory=200000000000 \
                    --disable-usage-stats
            " &
        done
        wait
    fi

    # Wait for all nodes
    echo "Waiting for ${WORLD_SIZE} nodes..."
    MAX_WAIT=300
    WAITED=0
    while true; do
        NODE_COUNT=$(python3 -c "
import ray
try:
    ray.init(address='auto', ignore_reinit_error=True)
    nodes = [n for n in ray.nodes() if n.get('Alive', False)]
    print(len(nodes))
except:
    print(0)
" 2>/dev/null || echo "0")

        if [ "$NODE_COUNT" -ge "$WORLD_SIZE" ]; then
            echo "All ${NODE_COUNT}/${WORLD_SIZE} nodes joined."
            break
        fi
        if [ "$WAITED" -ge "$MAX_WAIT" ]; then
            echo "ERROR: Timeout. Only ${NODE_COUNT}/${WORLD_SIZE} nodes."
            ray status
            exit 1
        fi
        sleep 5
        WAITED=$((WAITED + 5))
    done
else
    echo "[Step 1/3] Using existing Ray cluster at $RAY_ADDRESS"
fi

echo ""
echo "[Step 2/3] Verifying Ray cluster..."
ray status

# --- Step 3: Launch SGLang Engines + Router ---
echo ""
echo "[Step 3/3] Launching SGLang engines + router..."

# Remove stale env file so KoS nodes don't start prematurely
rm -f "${SGLANG_ENV_FILE}"

SGLANG_CMD="python3 ${SCRIPT_DIR}/launch_external_sglang.py"
SGLANG_CMD="${SGLANG_CMD} --model-path ${MODEL_PATH}"
SGLANG_CMD="${SGLANG_CMD} --num-engines ${NUM_ENGINES}"
SGLANG_CMD="${SGLANG_CMD} --tp-size ${TP_SIZE}"
SGLANG_CMD="${SGLANG_CMD} --dp-size ${DP_SIZE}"
SGLANG_CMD="${SGLANG_CMD} --mem-fraction-static ${MEM_FRACTION}"
SGLANG_CMD="${SGLANG_CMD} --chunked-prefill-size ${CHUNKED_PREFILL_SIZE}"
SGLANG_CMD="${SGLANG_CMD} --router-port ${ROUTER_PORT}"
SGLANG_CMD="${SGLANG_CMD} --server-base-port ${SERVER_BASE_PORT}"
if [ -n "$RUN_NAME" ] && [ "$RUN_NAME" != "default" ]; then
    SGLANG_CMD="${SGLANG_CMD} --run-name ${RUN_NAME}"
fi
SGLANG_CMD="${SGLANG_CMD} --output-dir ${OUTPUT_DIR}"

echo "Executing: ${SGLANG_CMD}"

SGLANG_LOG="${RUN_DIR}/logs/sglang_launcher.log"
mkdir -p "$(dirname "$SGLANG_LOG")"
${SGLANG_CMD} > "${SGLANG_LOG}" 2>&1 &
SGLANG_PID=$!
echo "SGLang launcher PID: ${SGLANG_PID}"

# Wait for the env file (signals router + all engines are healthy)
echo "Waiting for SGLang to be ready..."
MAX_WAIT=1800
WAITED=0
while [ ! -f "${SGLANG_ENV_FILE}" ]; do
    if ! kill -0 ${SGLANG_PID} 2>/dev/null; then
        echo "ERROR: SGLang launcher exited unexpectedly. Log:"
        tail -50 "${SGLANG_LOG}"
        exit 1
    fi
    if [ "$WAITED" -ge "$MAX_WAIT" ]; then
        echo "ERROR: Timeout waiting for SGLang (${MAX_WAIT}s). Log:"
        tail -50 "${SGLANG_LOG}"
        exit 1
    fi
    sleep 10
    WAITED=$((WAITED + 10))
    echo "  Waiting for SGLang... (${WAITED}s)"
done

source "${SGLANG_ENV_FILE}"
echo ""
echo "SGLang ready!"
echo "  Router: ${SGLANG_ROUTER_IP}:${SGLANG_ROUTER_PORT}"
echo "  Engines: ${SGLANG_ENGINE_ADDRS}"

# Verify router is actually responding
ROUTER_URL="http://${SGLANG_ROUTER_IP}:${SGLANG_ROUTER_PORT}/health"
echo "Verifying router health at ${ROUTER_URL}..."
for i in $(seq 1 10); do
    if curl -sf "${ROUTER_URL}" > /dev/null 2>&1; then
        echo "Router health check passed."
        break
    fi
    if [ "$i" -eq 10 ]; then
        echo "ERROR: Router not responding at ${ROUTER_URL}"
        exit 1
    fi
    sleep 3
done

echo ""
echo "=============================================="
echo "SGLang servers running!"
echo "  SGLang PID:  ${SGLANG_PID}"
echo "  Router:      http://${SGLANG_ROUTER_IP}:${SGLANG_ROUTER_PORT}"
echo "  Engines:     ${SGLANG_ENGINE_ADDRS}"
echo "  Env File:    ${SGLANG_ENV_FILE}"
echo "=============================================="
echo ""
echo "KoS CPU nodes can now start using:"
echo "  bash launch_kos_only.sh --sglang-env-file ${SGLANG_ENV_FILE} ..."
echo ""
echo "Press Ctrl+C to shutdown."

# =============================================================================
# Cleanup Handler
# =============================================================================
cleanup() {
    echo ""
    echo "Shutting down SGLang services..."

    kill -9 ${SGLANG_PID} 2>/dev/null || true
    pkill -9 -f "sglang" 2>/dev/null || true

    if [ -n "$WORKER_IPS" ] && [ -z "$RAY_ADDRESS" ]; then
        for WORKER_IP in $WORKER_IPS; do
            ssh -o StrictHostKeyChecking=no "$WORKER_IP" "
                pkill -9 -f sglang 2>/dev/null || true
                ray stop --force 2>/dev/null || true
            " &
        done
        wait
    fi

    if [ -z "$RAY_ADDRESS" ]; then
        ray stop --force 2>/dev/null || true
    fi

    echo "Cleanup complete."
    exit 0
}

trap cleanup SIGINT SIGTERM

# Wait — keeps script alive so Ray actors persist
wait
