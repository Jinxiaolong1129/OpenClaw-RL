#!/usr/bin/env bash
# Run remote rollout server for Kiro async training


set -o pipefail
set -ex

export HOSTNAME="${HOSTNAME}"
export HOME=/app/Kiro-on-Strands
export PYTHONPATH=/app/Kiro-on-Strands:$PYTHONPATH

# === Start Docker daemon first (required for SWE tasks) ===
# If Docker is already running (shared node), use it.
# If not, try to start it. If that fails, let k8s reschedule.
start_docker_with_retry() {
    local max_attempts=3
    
    for attempt in $(seq 1 $max_attempts); do
        echo "Starting Docker daemon (attempt $attempt/$max_attempts)..."
        
        if docker info &>/dev/null; then
            echo "Docker daemon is already running"
            return 0
        fi
        
        dockerd --data-root=/dev/shm &>/tmp/dockerd.log &
        local pid=$!
        
        for i in $(seq 1 30); do
            sleep 1
            kill -0 $pid 2>/dev/null || break
            docker info &>/dev/null && { echo "Docker started"; return 0; }
        done
        
        echo "Docker failed to start on attempt $attempt"
        kill -9 $pid 2>/dev/null || true
        wait $pid 2>/dev/null || true
        sleep 2
    done
    
    echo "ERROR: Docker failed to start"; cat /tmp/dockerd.log
    return 1
}

# Start Docker and fail if it doesn't work
if ! start_docker_with_retry; then
    echo "FATAL: Cannot start Docker daemon. Exiting."
    exit 1
fi

# === Update model_config.json with router URL ===
if [ -n "${VLLM_BASE_URL}" ]; then
    echo "Updating model_config.json with VLLM_BASE_URL: ${VLLM_BASE_URL}"
    cd ${HOME}
    # Use jq to update the base_url in hostedvllm config
    jq ".hostedvllm.client_args.base_url = \"${VLLM_BASE_URL}\"" model_config.json > model_config.json.tmp && \
        mv model_config.json.tmp model_config.json
    echo "Updated model_config.json:"
    jq '.hostedvllm' model_config.json
else
    echo "WARNING: VLLM_BASE_URL not set, using default model_config.json"
fi

# === Use RUN_ID from environment (set by deploy.py via YAML) ===
if [ -z "$RUN_ID" ]; then
    echo "ERROR: RUN_ID environment variable not set!"
    echo "Use deploy.py to deploy pods with a consistent RUN_ID."
    exit 1
fi

export RUN_DIR="/mnt_out/async-kiro/experiments/${RUN_ID}"
echo "Using run directory from RUN_ID: ${RUN_DIR}"

mkdir -p "${RUN_DIR}/trajectories"
mkdir -p "${RUN_DIR}/logs/rollout_logs"
mkdir -p "${RUN_DIR}/logs/docker_logs"

# === Paths ===
export DEFAULT_MODEL_PATH="/mnt_out/songyanh/models/Qwen3-Coder-30B-A3B-Instruct"

# === Metrics Saving Configuration ===
export TARGET_PATCH_COLUMN="test_patch"
export METRICS_DISCOUNT_FACTOR=0.9
export METRICS_MAX_TRAJECTORY_LENGTH=64000

# === Async Rollout Server Configuration ===
export NUM_WORKERS=8
export MAX_CACHED_IMAGES=10
# NUM_HELD_OUT_WORKERS: 0 for sync mode (validation uses all workers), 1+ for async mode
export NUM_HELD_OUT_WORKERS=${NUM_HELD_OUT_WORKERS:-1}

# === Extract router address from VLLM_BASE_URL ===
if [ -z "${VLLM_BASE_URL}" ]; then
    echo "ERROR: VLLM_BASE_URL environment variable not set!"
    exit 1
fi
ROUTER_ADDRESS=$(echo "${VLLM_BASE_URL}" | sed 's|http://||' | sed 's|/v1||')
echo "Using router address: ${ROUTER_ADDRESS}"

# === Start Async Remote Rollout Server ===
cd ${HOME}

python3 ${HOME}/remote_rollout_server/remote_rollout_async.py \
  --vllm_server_address ${ROUTER_ADDRESS} \
  --trajectory_folder ${RUN_DIR}/trajectories \
  --docker_log_folder ${RUN_DIR}/logs/docker_logs \
  --max_iterations 30 \
  --model_config_name hostedvllm \
  --workspace_path /mnt_private/zhenghuj/crux_0314_java_srctest_300_lines/crux_0318_java_9k_workspace/ \
  --timeout_seconds 3600 \
  --target_patch_column $TARGET_PATCH_COLUMN \
  --metrics_discount_factor $METRICS_DISCOUNT_FACTOR \
  --metrics_max_trajectory_length $METRICS_MAX_TRAJECTORY_LENGTH \
  --qwen_model \
  --swe_docker_images_path /mnt_out/zhenghuj/docker_images/sweap_images_800/ \
  --is_swe_task \
  --num_workers $NUM_WORKERS \
  --max_cached_images $MAX_CACHED_IMAGES \
  --num_held_out_workers $NUM_HELD_OUT_WORKERS \
  2>&1 | tee ${RUN_DIR}/logs/rollout_logs/remote_rollout_${HOSTNAME}.log
