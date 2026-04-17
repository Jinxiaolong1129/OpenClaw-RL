#!/bin/bash
# =============================================================================
# Combined SGLang Server + KoS Remote Rollout Launcher
# =============================================================================
#
# This script:
#   1. Starts a Ray cluster across all nodes
#   2. Launches SGLang engines + router via launch_external_sglang.py
#   3. Waits for SGLang to be fully healthy (router + all engines)
#   4. Launches sglang_remote_rollout.py (KoS server) on each worker node
#
# Architecture:
#   ┌──────────────────────────────────────────────────────────────────┐
#   │  Head Node                                                       │
#   │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
#   │  │ SGLang      │  │ SGLang      │  │ KoS Remote Rollout      │  │
#   │  │ Router      │  │ Engine 0    │  │ Server (port 5000)      │  │
#   │  │ (port 30000)│  │ (Ray Actor) │  │                         │  │
#   │  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
#   └──────────────────────────────────────────────────────────────────┘
#   ┌──────────────────────────────────────────────────────────────────┐
#   │  Worker Node N                                                   │
#   │  ┌─────────────┐  ┌─────────────────────────┐                   │
#   │  │ SGLang      │  │ KoS Remote Rollout      │                   │
#   │  │ Engine N    │  │ Server (port 5000)      │                   │
#   │  │ (Ray Actor) │  │                         │                   │
#   │  └─────────────┘  └─────────────────────────┘                   │
#   └──────────────────────────────────────────────────────────────────┘
#
# Usage:
#   # HyperPod (auto-detects from PET_NNODES):
#   bash launch_sglang_and_kos.sh --model-path /path/to/model \
#       --kos-workspace-path /path/to/workspace
#
#   # Manual multi-node:
#   MASTER_ADDR=<head_ip> WORKER_IPS="<w1> <w2> <w3>" \
#       bash launch_sglang_and_kos.sh --model-path /path/to/model \
#           --kos-workspace-path /path/to/workspace
#
# =============================================================================

set -e

# =============================================================================
# Default Configuration
# =============================================================================

# --- SGLang ---
MODEL_PATH=""
NUM_ENGINES=0  # 0 = auto from cluster GPUs
TP_SIZE=8
DP_SIZE=1
MEM_FRACTION=0.6
ROUTER_PORT=30000
SERVER_BASE_PORT=13140

# --- KoS Remote Rollout ---
KOS_HOME=${KOS_HOME:-"/mnt_out/myshang/codebase/Kiro-on-Strands"}
KOS_PORT=${KOS_PORT:-5000}
KOS_NUM_WORKERS=${KOS_NUM_WORKERS:-4}
KOS_MAX_ITERATIONS=${KOS_MAX_ITERATIONS:-30}
KOS_TIMEOUT=${KOS_TIMEOUT:-3600}
KOS_MODEL_CONFIG=${KOS_MODEL_CONFIG:-"hostedsglang"}
KOS_WORKSPACE_PATH=${KOS_WORKSPACE_PATH:-""}
KOS_SWE_DOCKER_IMAGES=${KOS_SWE_DOCKER_IMAGES:-""}
KOS_IS_SWE_TASK=${KOS_IS_SWE_TASK:-true}
KOS_MAX_CONCURRENT_AGENTS=${KOS_MAX_CONCURRENT_AGENTS:-32}
KOS_AGENT_CONCURRENCY_MULTIPLIER=${KOS_AGENT_CONCURRENCY_MULTIPLIER:-2}
KOS_TEST_EXECUTOR_WORKERS=${KOS_TEST_EXECUTOR_WORKERS:-32}
KOS_TARGET_PATCH_COLUMN=${KOS_TARGET_PATCH_COLUMN:-"test_patch"}
KOS_METRICS_DISCOUNT_FACTOR=${KOS_METRICS_DISCOUNT_FACTOR:-0.9}
KOS_METRICS_MAX_TRAJECTORY_LENGTH=${KOS_METRICS_MAX_TRAJECTORY_LENGTH:-134272}
METRICS_MAX_RESPONSE_LENGTH=${METRICS_MAX_RESPONSE_LENGTH:-117888}

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
        # SGLang
        --model-path)        MODEL_PATH="$2";        shift 2 ;;
        --num-engines)       NUM_ENGINES="$2";       shift 2 ;;
        --tp-size)           TP_SIZE="$2";           shift 2 ;;
        --dp-size)           DP_SIZE="$2";           shift 2 ;;
        --mem-fraction)      MEM_FRACTION="$2";      shift 2 ;;
        --router-port)       ROUTER_PORT="$2";       shift 2 ;;
        --server-base-port)  SERVER_BASE_PORT="$2";  shift 2 ;;
        --num-gpus-per-node) NUM_GPUS_PER_NODE="$2"; shift 2 ;;
        --ray-address)       RAY_ADDRESS="$2";       shift 2 ;;
        --master-addr)       MASTER_ADDR="$2";       shift 2 ;;
        --worker-ips)        WORKER_IPS="$2";        shift 2 ;;
        # KoS
        --kos-home)          KOS_HOME="$2";          shift 2 ;;
        --kos-port)          KOS_PORT="$2";          shift 2 ;;
        --kos-num-workers)   KOS_NUM_WORKERS="$2";   shift 2 ;;
        --kos-max-iterations) KOS_MAX_ITERATIONS="$2"; shift 2 ;;
        --kos-timeout)       KOS_TIMEOUT="$2";       shift 2 ;;
        --kos-model-config)  KOS_MODEL_CONFIG="$2";  shift 2 ;;
        --kos-workspace-path) KOS_WORKSPACE_PATH="$2"; shift 2 ;;
        --kos-swe-docker-images) KOS_SWE_DOCKER_IMAGES="$2"; shift 2 ;;
        --kos-is-swe-task)   KOS_IS_SWE_TASK="$2";  shift 2 ;;
        --kos-max-concurrent-agents) KOS_MAX_CONCURRENT_AGENTS="$2"; shift 2 ;;
        --kos-agent-concurrency-multiplier) KOS_AGENT_CONCURRENCY_MULTIPLIER="$2"; shift 2 ;;
        --kos-test-executor-workers) KOS_TEST_EXECUTOR_WORKERS="$2"; shift 2 ;;
        # General
        --run-name)          RUN_NAME="$2";          shift 2 ;;
        --output-dir)        OUTPUT_DIR="$2";        shift 2 ;;
        --slime-dir)         SLIME_DIR="$2";         shift 2 ;;
        --help)
            echo "Usage: $0 --model-path <path> --kos-workspace-path <path> [options]"
            echo ""
            echo "SGLang Options:"
            echo "  --model-path PATH           HuggingFace model path (required)"
            echo "  --num-engines N             Number of engines (default: auto)"
            echo "  --tp-size N                 Tensor parallel size (default: 8)"
            echo "  --dp-size N                 Data parallel size (default: 1)"
            echo "  --mem-fraction F            GPU memory fraction (default: 0.6)"
            echo "  --router-port PORT          Router port (default: 30000)"
            echo "  --server-base-port PORT     Engine base port (default: 13140)"
            echo "  --ray-address ADDR          Existing Ray cluster address"
            echo "  --master-addr ADDR          Head node IP"
            echo "  --worker-ips IPS            Space-separated worker IPs"
            echo ""
            echo "KoS Options:"
            echo "  --kos-home PATH             Kiro-on-Strands repo (default: /root/Kiro-on-Strands)"
            echo "  --kos-port PORT             KoS server port (default: 5000)"
            echo "  --kos-num-workers N         Worker threads per node (default: 4)"
            echo "  --kos-max-iterations N      Max agent iterations (default: 30)"
            echo "  --kos-timeout SECS          Timeout per run (default: 3600)"
            echo "  --kos-model-config NAME     Model config name (default: hostedsglang)"
            echo "  --kos-workspace-path PATH   Workspace path (required)"
            echo "  --kos-swe-docker-images PATH  SWE docker images path"
            echo "  --kos-max-concurrent-agents N Max concurrent agent sessions per node (default: 4)"
            echo "  --kos-is-swe-task BOOL      SWE task mode (default: true)"
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
if [ -z "$KOS_WORKSPACE_PATH" ]; then
    echo "ERROR: --kos-workspace-path is required"; exit 1
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
echo "SGLang + KoS Combined Launcher"
echo "=============================================="
echo "Cluster:           ${WORLD_SIZE} nodes × ${NUM_GPUS_PER_NODE} GPUs"
echo "SGLang Engines:    ${NUM_ENGINES} (TP=${TP_SIZE}, DP=${DP_SIZE})"
echo "Model:             ${MODEL_PATH}"
echo "Run Name:          ${RUN_NAME}"
echo "KoS Workers/Node:  ${KOS_NUM_WORKERS}"
echo "KoS Concurrent:    ${KOS_MAX_CONCURRENT_AGENTS}"
echo "KoS Port:          ${KOS_PORT}"
echo "=============================================="

# =============================================================================
# Helper: Start Docker daemon (needed for SWE tasks on KoS nodes)
# =============================================================================
start_docker_daemon() {
    if docker info &>/dev/null; then
        echo "Docker daemon already running"
        return 0
    fi
    echo "Starting Docker daemon..."
    for attempt in 1 2 3; do
        dockerd --data-root=/dev/shm &>/tmp/dockerd.log &
        local pid=$!
        for i in $(seq 1 30); do
            sleep 1
            kill -0 $pid 2>/dev/null || break
            docker info &>/dev/null && { echo "Docker started"; return 0; }
        done
        echo "Docker attempt $attempt failed"
        kill -9 $pid 2>/dev/null || true
        wait $pid 2>/dev/null || true
        sleep 2
    done
    echo "ERROR: Docker failed to start"
    cat /tmp/dockerd.log 2>/dev/null
    return 1
}

# =============================================================================
# Helper: Update model_config.json with SGLang router URL
# =============================================================================
update_model_config() {
    local router_ip="$1"
    local router_port="$2"
    local src_config="${KOS_HOME}/model_config.json"

    if [ ! -f "$src_config" ]; then
        echo "WARNING: ${src_config} not found, skipping model config update"
        return
    fi

    # Copy to a per-node local path to avoid race conditions when multiple
    # nodes on a shared filesystem all try to update the same file.
    local local_config="/tmp/kos_model_config_$(hostname).json"
    cp "$src_config" "$local_config"

    # Always point to the SGLang router (reachable from all nodes)
    local llm_url="http://${router_ip}:${router_port}/v1"
    echo "Updating model_config (local copy): base_url -> ${llm_url}"
    jq ".${KOS_MODEL_CONFIG}.client_args.base_url = \"${llm_url}\"" \
        "$local_config" > "${local_config}.tmp" && mv "${local_config}.tmp" "$local_config"

    # Verify the write succeeded
    local written_url
    written_url=$(jq -r ".${KOS_MODEL_CONFIG}.client_args.base_url" "$local_config" 2>/dev/null)
    echo "Verified model_config base_url: ${written_url}"

    # Export so launch_kos_server can pass it to the Python process
    export KOS_LOCAL_MODEL_CONFIG="$local_config"
}

# =============================================================================
# Helper: Launch KoS remote rollout server
# =============================================================================
launch_kos_server() {
    local router_ip="$1"
    local router_port="$2"
    local engine_addrs="$3"

    # Create run directories
    mkdir -p "${RUN_DIR}/trajectories"
    mkdir -p "${RUN_DIR}/logs/rollout_logs"
    mkdir -p "${RUN_DIR}/logs/docker_logs"

    # Start Docker if SWE task
    if [ "$KOS_IS_SWE_TASK" = true ]; then
        start_docker_daemon || { echo "FATAL: Docker required for SWE tasks"; exit 1; }
    fi

    # Update model config with router URL
    update_model_config "$router_ip" "$router_port"

    # Build sticky routing args from engine addresses
    STICKY_ROUTING_ARGS=""
    if [ -n "${engine_addrs}" ]; then
        ENGINE_URLS=""
        for addr in ${engine_addrs}; do
            if [[ "${addr}" != http* ]]; then
                ENGINE_URLS="${ENGINE_URLS} http://${addr}"
            else
                ENGINE_URLS="${ENGINE_URLS} ${addr}"
            fi
        done
        STICKY_ROUTING_ARGS="--sglang_engine_urls ${ENGINE_URLS}"
        echo "Sticky routing enabled:${ENGINE_URLS}"
    fi

    # Build SWE task args
    SWE_ARGS=""
    if [ "$KOS_IS_SWE_TASK" = true ]; then
        SWE_ARGS="--is_swe_task"
        if [ -n "$KOS_SWE_DOCKER_IMAGES" ]; then
            SWE_ARGS="${SWE_ARGS} --swe_docker_images_path ${KOS_SWE_DOCKER_IMAGES}"
        fi
    fi

    local log_file="${RUN_DIR}/logs/rollout_logs/remote_rollout_$(hostname).log"
    echo "Starting KoS server (log: ${log_file})..."

    export PYTHONPATH="${KOS_HOME}:${PYTHONPATH:-}"
    export HOME="${KOS_HOME}"
    export DEFAULT_MODEL_PATH="${MODEL_PATH:-/mnt_out/songyanh/models/Qwen3-Coder-30B-A3B-Instruct}"
    export AGENT_CONCURRENCY_MULTIPLIER=${KOS_AGENT_CONCURRENCY_MULTIPLIER}
    export TEST_EXECUTOR_WORKERS=${KOS_TEST_EXECUTOR_WORKERS}
    export TARGET_PATCH_COLUMN=${KOS_TARGET_PATCH_COLUMN}
    export METRICS_DISCOUNT_FACTOR=${KOS_METRICS_DISCOUNT_FACTOR}
    export METRICS_MAX_TRAJECTORY_LENGTH=${KOS_METRICS_MAX_TRAJECTORY_LENGTH}
    export METRICS_MAX_RESPONSE_LENGTH=${METRICS_MAX_RESPONSE_LENGTH}
    export MAX_CACHED_IMAGES=10
    export SKIP_TEST_EXECUTION=${SKIP_TEST_EXECUTION:-0}

    # Point the KoS server at the per-node local model config (avoids shared-fs race)
    if [ -n "${KOS_LOCAL_MODEL_CONFIG:-}" ]; then
        export KOS_MODEL_CONFIG_PATH="${KOS_LOCAL_MODEL_CONFIG}"
        echo "Using local model config: ${KOS_MODEL_CONFIG_PATH}"
    fi

    cd "${KOS_HOME}"
    python3 "${KOS_HOME}/remote_rollout_server/sglang_remote_rollout.py" \
        --trajectory_folder "${RUN_DIR}/trajectories" \
        --sglang_log_folder "${RUN_DIR}/logs" \
        --max_iterations ${KOS_MAX_ITERATIONS} \
        --model_config_name ${KOS_MODEL_CONFIG} \
        --workspace_path "${KOS_WORKSPACE_PATH}" \
        --timeout_seconds ${KOS_TIMEOUT} \
        --target_patch_column ${KOS_TARGET_PATCH_COLUMN} \
        --metrics_discount_factor ${KOS_METRICS_DISCOUNT_FACTOR} \
        --metrics_max_trajectory_length ${KOS_METRICS_MAX_TRAJECTORY_LENGTH} \
        --metrics_max_response_length ${METRICS_MAX_RESPONSE_LENGTH} \
        --num_processes ${KOS_NUM_WORKERS} \
        --max_concurrent_agents ${KOS_MAX_CONCURRENT_AGENTS} \
        --agent_concurrency_multiplier ${KOS_AGENT_CONCURRENCY_MULTIPLIER} \
        --test_executor_workers ${KOS_TEST_EXECUTOR_WORKERS} \
        ${SWE_ARGS} \
        ${STICKY_ROUTING_ARGS} \
        2>&1 | tee "${log_file}"
}

# =============================================================================
# Worker Node Path: Join Ray, wait for SGLang, launch KoS
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

    echo "[Worker] Joined Ray. Waiting for SGLang env file..."

    # Remove stale env file from previous runs so we don't read old data.
    # The head node also does rm -f before launching SGLang, but the worker
    # may start waiting before the head gets there.
    rm -f "${SGLANG_ENV_FILE}"

    # Wait for the env file (written by head after SGLang is ready)
    MAX_WAIT=600
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

    # Launch KoS server (blocks forever)
    # NOTE: We intentionally do NOT pass SGLANG_ENGINE_ADDRS for sticky routing.
    # All KoS servers use the SGLang router (SGLANG_ROUTER_IP:SGLANG_ROUTER_PORT)
    # as the LLM backend. The router handles KV-cache-aware request distribution
    # to engines automatically. Direct engine URLs may not be reachable across
    # nodes due to firewall/security group restrictions on non-standard ports.
    launch_kos_server "${SGLANG_ROUTER_IP}" "${SGLANG_ROUTER_PORT}" ""
    exit 0
fi

# =============================================================================
# Head Node Path: Start Ray cluster, launch SGLang, then KoS
# =============================================================================

# --- Step 1: Start Ray Cluster ---
if [ -z "$RAY_ADDRESS" ]; then
    echo ""
    echo "[Step 1/4] Setting up Ray cluster..."

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
    echo "[Step 1/4] Using existing Ray cluster at $RAY_ADDRESS"
fi

echo ""
echo "[Step 2/4] Verifying Ray cluster..."
ray status

# --- Step 3: Launch SGLang Engines + Router ---
echo ""
echo "[Step 3/4] Launching SGLang engines + router..."

# Remove stale env file so workers don't start prematurely
rm -f "${SGLANG_ENV_FILE}"

SGLANG_CMD="python3 ${SCRIPT_DIR}/launch_external_sglang.py"
SGLANG_CMD="${SGLANG_CMD} --model-path ${MODEL_PATH}"
SGLANG_CMD="${SGLANG_CMD} --num-engines ${NUM_ENGINES}"
SGLANG_CMD="${SGLANG_CMD} --tp-size ${TP_SIZE}"
SGLANG_CMD="${SGLANG_CMD} --dp-size ${DP_SIZE}"
SGLANG_CMD="${SGLANG_CMD} --mem-fraction-static ${MEM_FRACTION}"
SGLANG_CMD="${SGLANG_CMD} --router-port ${ROUTER_PORT}"
SGLANG_CMD="${SGLANG_CMD} --server-base-port ${SERVER_BASE_PORT}"
if [ -n "$RUN_NAME" ] && [ "$RUN_NAME" != "default" ]; then
    SGLANG_CMD="${SGLANG_CMD} --run-name ${RUN_NAME}"
fi
SGLANG_CMD="${SGLANG_CMD} --output-dir ${OUTPUT_DIR}"

echo "Executing: ${SGLANG_CMD}"

# Launch SGLang in background — it writes the env file when ready, then
# stays alive to keep Ray actors running.
SGLANG_LOG="${RUN_DIR}/logs/sglang_launcher.log"
mkdir -p "$(dirname "$SGLANG_LOG")"
${SGLANG_CMD} > "${SGLANG_LOG}" 2>&1 &
SGLANG_PID=$!
echo "SGLang launcher PID: ${SGLANG_PID}"

# Wait for the env file (signals router + all engines are healthy)
echo "Waiting for SGLang to be ready..."
MAX_WAIT=900
WAITED=0
while [ ! -f "${SGLANG_ENV_FILE}" ]; do
    # Check if the launcher process died
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

# --- Step 4: Launch KoS Remote Rollout Servers ---
echo ""
echo "[Step 4/4] Launching KoS remote rollout servers on all nodes..."
echo "  All KoS servers will use the SGLang router at ${SGLANG_ROUTER_IP}:${SGLANG_ROUTER_PORT}"
echo "  (sticky routing disabled — router handles KV-cache-aware distribution)"

# Build SWE task flag
SWE_FLAG=""
if [ "$KOS_IS_SWE_TASK" = true ]; then
    SWE_FLAG="--is_swe_task"
    if [ -n "$KOS_SWE_DOCKER_IMAGES" ]; then
        SWE_FLAG="${SWE_FLAG} --swe_docker_images_path ${KOS_SWE_DOCKER_IMAGES}"
    fi
fi

# Launch KoS on worker nodes via SSH (background)
KOS_PIDS=()
if [ -n "$WORKER_IPS" ]; then
    for WORKER_IP in $WORKER_IPS; do
        echo "  Starting KoS on worker: ${WORKER_IP}"
        KOS_WORKER_LOG="${RUN_DIR}/logs/rollout_logs/remote_rollout_${WORKER_IP}.log"

        # Write a launcher script to the shared filesystem to avoid SSH quoting issues
        WORKER_LAUNCH_SCRIPT="${RUN_DIR}/logs/_launch_kos_${WORKER_IP}.sh"
        cat > "${WORKER_LAUNCH_SCRIPT}" <<WORKER_EOF
#!/bin/bash
set -e
export PYTHONPATH="${KOS_HOME}:\${PYTHONPATH:-}"
export HOME="${KOS_HOME}"
export DEFAULT_MODEL_PATH="${MODEL_PATH:-/mnt_out/songyanh/models/Qwen3-Coder-30B-A3B-Instruct}"
export AGENT_CONCURRENCY_MULTIPLIER=${KOS_AGENT_CONCURRENCY_MULTIPLIER}
export TEST_EXECUTOR_WORKERS=${KOS_TEST_EXECUTOR_WORKERS}
export TARGET_PATCH_COLUMN="${KOS_TARGET_PATCH_COLUMN}"
export METRICS_DISCOUNT_FACTOR=${KOS_METRICS_DISCOUNT_FACTOR}
export METRICS_MAX_TRAJECTORY_LENGTH=${KOS_METRICS_MAX_TRAJECTORY_LENGTH}
export MAX_CACHED_IMAGES=10
export SKIP_TEST_EXECUTION=${SKIP_TEST_EXECUTION:-0}

# Start Docker if needed
if [ "${KOS_IS_SWE_TASK}" = true ]; then
    if ! docker info &>/dev/null; then
        dockerd --data-root=/dev/shm &>/tmp/dockerd.log &
        for i in \$(seq 1 30); do
            sleep 1
            docker info &>/dev/null && break
        done
    fi
fi

# Copy model config to a local path and set router URL
LOCAL_CONFIG="/tmp/kos_model_config_\$(hostname).json"
CONFIG_FILE="${KOS_HOME}/model_config.json"
if [ -f "\${CONFIG_FILE}" ]; then
    cp "\${CONFIG_FILE}" "\${LOCAL_CONFIG}"
    LLM_URL="http://${SGLANG_ROUTER_IP}:${SGLANG_ROUTER_PORT}/v1"
    echo "Setting base_url to \${LLM_URL}"
    jq '.${KOS_MODEL_CONFIG}.client_args.base_url = "'\${LLM_URL}'"' \\
        "\${LOCAL_CONFIG}" > "\${LOCAL_CONFIG}.tmp" && mv "\${LOCAL_CONFIG}.tmp" "\${LOCAL_CONFIG}"
    WRITTEN_URL=\$(jq -r '.${KOS_MODEL_CONFIG}.client_args.base_url' "\${LOCAL_CONFIG}" 2>/dev/null)
    echo "Verified model_config base_url: \${WRITTEN_URL}"
    export KOS_MODEL_CONFIG_PATH="\${LOCAL_CONFIG}"
    echo "Using local model config: \${LOCAL_CONFIG}"
fi

mkdir -p "${RUN_DIR}/trajectories" "${RUN_DIR}/logs/rollout_logs" "${RUN_DIR}/logs/docker_logs"

cd "${KOS_HOME}"
exec python3 "${KOS_HOME}/remote_rollout_server/sglang_remote_rollout.py" \\
    --trajectory_folder "${RUN_DIR}/trajectories" \\
    --sglang_log_folder "${RUN_DIR}/logs" \\
    --max_iterations ${KOS_MAX_ITERATIONS} \\
    --model_config_name ${KOS_MODEL_CONFIG} \\
    --workspace_path "${KOS_WORKSPACE_PATH}" \\
    --timeout_seconds ${KOS_TIMEOUT} \\
    --target_patch_column "${KOS_TARGET_PATCH_COLUMN}" \\
    --metrics_discount_factor ${KOS_METRICS_DISCOUNT_FACTOR} \\
    --metrics_max_trajectory_length ${KOS_METRICS_MAX_TRAJECTORY_LENGTH} \\
    --metrics_max_response_length ${METRICS_MAX_RESPONSE_LENGTH} \\
    --num_processes ${KOS_NUM_WORKERS} \\
    --max_concurrent_agents ${KOS_MAX_CONCURRENT_AGENTS} \\
    --agent_concurrency_multiplier ${KOS_AGENT_CONCURRENCY_MULTIPLIER} \\
    --test_executor_workers ${KOS_TEST_EXECUTOR_WORKERS} \\
    ${SWE_FLAG}
WORKER_EOF
        chmod +x "${WORKER_LAUNCH_SCRIPT}"

        ssh -o StrictHostKeyChecking=no "$WORKER_IP" \
            "bash ${WORKER_LAUNCH_SCRIPT}" > "${KOS_WORKER_LOG}" 2>&1 &
        KOS_PIDS+=($!)
    done
fi

# Launch KoS on head node too
echo "  Starting KoS on head node..."
launch_kos_server "${SGLANG_ROUTER_IP}" "${SGLANG_ROUTER_PORT}" "" &
KOS_HEAD_PID=$!
KOS_PIDS+=($KOS_HEAD_PID)

echo ""
echo "=============================================="
echo "All services running!"
echo "  SGLang PID:  ${SGLANG_PID}"
echo "  KoS PIDs:    ${KOS_PIDS[*]}"
echo "  Router:      http://${SGLANG_ROUTER_IP}:${SGLANG_ROUTER_PORT}"
echo "  KoS servers: port ${KOS_PORT} on all ${WORLD_SIZE} nodes"
echo "  Logs:        ${RUN_DIR}/logs/"
echo "=============================================="
echo ""
echo "Press Ctrl+C to shutdown all services."

# =============================================================================
# Cleanup Handler
# =============================================================================
cleanup() {
    echo ""
    echo "Shutting down all services..."

    # Kill KoS servers
    for pid in "${KOS_PIDS[@]}"; do
        kill -9 $pid 2>/dev/null || true
    done

    # Kill SGLang launcher
    kill -9 ${SGLANG_PID} 2>/dev/null || true

    # Kill SGLang processes
    pkill -9 -f "sglang" 2>/dev/null || true
    pkill -9 -f "sglang_remote_rollout" 2>/dev/null || true

    # Cleanup workers
    if [ -n "$WORKER_IPS" ] && [ -z "$RAY_ADDRESS" ]; then
        for WORKER_IP in $WORKER_IPS; do
            ssh -o StrictHostKeyChecking=no "$WORKER_IP" "
                pkill -9 -f sglang_remote_rollout 2>/dev/null || true
                pkill -9 -f sglang 2>/dev/null || true
                ray stop --force 2>/dev/null || true
            " &
        done
        wait
    fi

    # Stop local Ray
    if [ -z "$RAY_ADDRESS" ]; then
        ray stop --force 2>/dev/null || true
    fi

    echo "Cleanup complete."
    exit 0
}

trap cleanup SIGINT SIGTERM

# Wait for any child to exit (keeps script alive)
wait
