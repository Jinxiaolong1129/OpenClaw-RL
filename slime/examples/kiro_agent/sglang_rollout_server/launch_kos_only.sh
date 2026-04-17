#!/bin/bash
# =============================================================================
# KoS-Only Server Launcher (CPU Nodes)
# =============================================================================
#
# Separated from launch_sglang_and_kos.sh for disaggregated deployment:
#   - launch_sglang_only.sh runs on GPU nodes: Ray cluster + SGLang engines
#   - This script runs on CPU nodes: KoS agent servers (tool execution + Docker)
#
# Architecture:
#   ┌──────────────────────────────────────────────────────────────────┐
#   │  CPU Node 0                                                      │
#   │  ┌─────────────────────────┐                                     │
#   │  │ KoS Remote Rollout      │──── HTTP ────▶ SGLang Router        │
#   │  │ Server (port 5000)      │               (GPU head node)       │
#   │  │ + Docker containers     │                                     │
#   │  └─────────────────────────┘                                     │
#   └──────────────────────────────────────────────────────────────────┘
#   ┌──────────────────────────────────────────────────────────────────┐
#   │  CPU Node N                                                      │
#   │  ┌─────────────────────────┐                                     │
#   │  │ KoS Remote Rollout      │──── HTTP ────▶ SGLang Router        │
#   │  │ Server (port 5000)      │               (GPU head node)       │
#   │  │ + Docker containers     │                                     │
#   │  └─────────────────────────┘                                     │
#   └──────────────────────────────────────────────────────────────────┘
#
# This script:
#   1. Waits for the SGLang env file (written by launch_sglang_only.sh on GPU nodes)
#   2. Reads the SGLang router IP/port from the env file
#   3. Writes a kos_servers.env file listing all KoS server URLs
#   4. Launches sglang_remote_rollout.py on this node
#
# Usage:
#   # HyperPod (auto-detects from PET_NNODES):
#   bash launch_kos_only.sh \
#       --sglang-env-file /mnt_out/.../sglang_external_rollout.env \
#       --kos-workspace-path /path/to/workspace
#
#   # Manual:
#   bash launch_kos_only.sh \
#       --sglang-router-ip 10.0.1.5 --sglang-router-port 30000 \
#       --kos-workspace-path /path/to/workspace
#
# =============================================================================

set -e

# =============================================================================
# Default Configuration
# =============================================================================

# --- SGLang connection (read from env file or passed directly) ---
SGLANG_ENV_FILE=${SGLANG_ENV_FILE:-""}
SGLANG_ROUTER_IP=${SGLANG_ROUTER_IP:-""}
SGLANG_ROUTER_PORT=${SGLANG_ROUTER_PORT:-"30000"}

# --- KoS Remote Rollout ---
KOS_HOME=${KOS_HOME:-"/mnt_out/yawenwuu/packages/rl_test_repo/Kiro-on-Strands/src/Kiro-on-Strands"}
KOS_PORT=${KOS_PORT:-5000}
KOS_NUM_WORKERS=${KOS_NUM_WORKERS:-4}
KOS_MAX_ITERATIONS=${KOS_MAX_ITERATIONS:-30}
KOS_TIMEOUT=${KOS_TIMEOUT:-3600}
KOS_MODEL_CONFIG=${KOS_MODEL_CONFIG:-"hostedsglang"}
KOS_WORKSPACE_PATH=${KOS_WORKSPACE_PATH:-""}
KOS_SWE_DOCKER_IMAGES=${KOS_SWE_DOCKER_IMAGES:-""}
KOS_IS_SWE_TASK=${KOS_IS_SWE_TASK:-true}
KOS_IS_CGS_TASK=${KOS_IS_CGS_TASK:-false}
KOS_CGS_DOCKER_IMAGE=${KOS_CGS_DOCKER_IMAGE:-""}
KOS_CGS_DOCKER_TAR=${KOS_CGS_DOCKER_TAR:-""}
KOS_CGS_TARBALL_BASE=${KOS_CGS_TARBALL_BASE:-""}
KOS_MAX_CONCURRENT_AGENTS=${KOS_MAX_CONCURRENT_AGENTS:-32}
KOS_AGENT_CONCURRENCY_MULTIPLIER=${KOS_AGENT_CONCURRENCY_MULTIPLIER:-1}
KOS_TEST_EXECUTOR_WORKERS=${KOS_TEST_EXECUTOR_WORKERS:-8}
KOS_TARGET_PATCH_COLUMN=${KOS_TARGET_PATCH_COLUMN:-"test_patch"}
KOS_METRICS_DISCOUNT_FACTOR=${KOS_METRICS_DISCOUNT_FACTOR:-0.9}
KOS_METRICS_MAX_TRAJECTORY_LENGTH=${KOS_METRICS_MAX_TRAJECTORY_LENGTH:-81920}
METRICS_MAX_RESPONSE_LENGTH=${METRICS_MAX_RESPONSE_LENGTH:-73728}

# --- CGS Agent-as-Judge ---
KOS_USE_AGENT_JUDGE=${KOS_USE_AGENT_JUDGE:-false}
KOS_USE_JUDGE_AS_REWARD=${KOS_USE_JUDGE_AS_REWARD:-false}
KOS_JUDGE_TRAJECTORY_MODE=${KOS_JUDGE_TRAJECTORY_MODE:-false}
KOS_JUDGE_MODEL_CONFIG=${KOS_JUDGE_MODEL_CONFIG:-"opus_v4.6"}
KOS_MAX_TOOL_CALLS_PER_TURN=${KOS_MAX_TOOL_CALLS_PER_TURN:-0}

# --- General ---
RUN_NAME=${RUN_NAME:-"default"}
OUTPUT_DIR=${OUTPUT_DIR:-"/mnt_out/yawenwuu/logs/slime"}
MODEL_PATH=${MODEL_PATH:-""}

# =============================================================================
# Parse Arguments
# =============================================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        # SGLang connection
        --sglang-env-file)   SGLANG_ENV_FILE="$2";   shift 2 ;;
        --sglang-router-ip)  SGLANG_ROUTER_IP="$2";  shift 2 ;;
        --sglang-router-port) SGLANG_ROUTER_PORT="$2"; shift 2 ;;
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
        --kos-is-cgs-task)   KOS_IS_CGS_TASK="$2";  shift 2 ;;
        --kos-cgs-docker-image) KOS_CGS_DOCKER_IMAGE="$2"; shift 2 ;;
        --kos-cgs-docker-tar) KOS_CGS_DOCKER_TAR="$2"; shift 2 ;;
        --kos-cgs-tarball-base) KOS_CGS_TARBALL_BASE="$2"; shift 2 ;;
        --kos-use-agent-judge) KOS_USE_AGENT_JUDGE="$2"; shift 2 ;;
        --kos-use-judge-as-reward) KOS_USE_JUDGE_AS_REWARD="$2"; shift 2 ;;
        --kos-judge-trajectory-mode) KOS_JUDGE_TRAJECTORY_MODE="$2"; shift 2 ;;
        --kos-judge-model-config) KOS_JUDGE_MODEL_CONFIG="$2"; shift 2 ;;
        --kos-max-tool-calls-per-turn) KOS_MAX_TOOL_CALLS_PER_TURN="$2"; shift 2 ;;
        --kos-max-concurrent-agents) KOS_MAX_CONCURRENT_AGENTS="$2"; shift 2 ;;
        --kos-agent-concurrency-multiplier) KOS_AGENT_CONCURRENCY_MULTIPLIER="$2"; shift 2 ;;
        --kos-test-executor-workers) KOS_TEST_EXECUTOR_WORKERS="$2"; shift 2 ;;
        # General
        --model-path)        MODEL_PATH="$2";        shift 2 ;;
        --run-name)          RUN_NAME="$2";          shift 2 ;;
        --output-dir)        OUTPUT_DIR="$2";        shift 2 ;;
        --help)
            echo "Usage: $0 --kos-workspace-path <path> [--sglang-env-file <path> | --sglang-router-ip <ip>] [options]"
            echo ""
            echo "SGLang Connection (one of):"
            echo "  --sglang-env-file PATH      Path to sglang_external_rollout.env (waits if not ready)"
            echo "  --sglang-router-ip IP       SGLang router IP (direct, no waiting)"
            echo "  --sglang-router-port PORT   SGLang router port (default: 30000)"
            echo ""
            echo "KoS Options:"
            echo "  --kos-home PATH             Kiro-on-Strands repo path"
            echo "  --kos-port PORT             KoS server port (default: 5000)"
            echo "  --kos-num-workers N         Worker threads per node (default: 4)"
            echo "  --kos-max-iterations N      Max agent iterations (default: 30)"
            echo "  --kos-timeout SECS          Timeout per run (default: 3600)"
            echo "  --kos-model-config NAME     Model config name (default: hostedsglang)"
            echo "  --kos-workspace-path PATH   Workspace path (required)"
            echo "  --kos-swe-docker-images PATH  SWE docker images path"
            echo "  --kos-max-concurrent-agents N Max concurrent agents per node (default: 32)"
            echo "  --kos-is-swe-task BOOL      SWE task mode (default: true)"
            echo ""
            echo "General Options:"
            echo "  --model-path PATH           Model path (for DEFAULT_MODEL_PATH env)"
            echo "  --run-name NAME             Run name for isolation (default: default)"
            echo "  --output-dir DIR            Output directory"
            exit 0
            ;;
        *)
            echo "Unknown option: $1 (use --help)"
            exit 1
            ;;
    esac
done

if [ -z "$KOS_WORKSPACE_PATH" ]; then
    echo "ERROR: --kos-workspace-path is required"; exit 1
fi

# =============================================================================
# Derived Paths
# =============================================================================
RUN_DIR="${OUTPUT_DIR}/kos/${RUN_NAME}"
ENV_DIR="${OUTPUT_DIR}"
if [ -n "$RUN_NAME" ] && [ "$RUN_NAME" != "default" ]; then
    ENV_DIR="${OUTPUT_DIR}/${RUN_NAME}"
fi

# If no env file specified, construct default path
if [ -z "$SGLANG_ENV_FILE" ] && [ -z "$SGLANG_ROUTER_IP" ]; then
    SGLANG_ENV_FILE="${ENV_DIR}/sglang_external_rollout.env"
    echo "No --sglang-env-file or --sglang-router-ip specified."
    echo "Defaulting to: ${SGLANG_ENV_FILE}"
fi

# KoS env file: lists all KoS server URLs for the trainer to consume
KOS_ENV_FILE="${ENV_DIR}/kos_servers.env"

# =============================================================================
# Step 1: Get SGLang Router Address
# =============================================================================
if [ -n "$SGLANG_ENV_FILE" ] && [ -z "$SGLANG_ROUTER_IP" ]; then
    echo "[Step 1/3] Waiting for SGLang env file: ${SGLANG_ENV_FILE}"
    MAX_WAIT=1800  # 30 min — GPU nodes may take a while to start
    WAITED=0
    while [ ! -f "${SGLANG_ENV_FILE}" ]; do
        if [ "$WAITED" -ge "$MAX_WAIT" ]; then
            echo "ERROR: Timeout waiting for ${SGLANG_ENV_FILE} (${MAX_WAIT}s)"
            echo "Make sure launch_sglang_only.sh is running on GPU nodes."
            exit 1
        fi
        sleep 10
        WAITED=$((WAITED + 10))
        if [ $((WAITED % 60)) -eq 0 ]; then
            echo "  Waiting for SGLang GPU nodes... (${WAITED}s)"
        fi
    done
    source "${SGLANG_ENV_FILE}"
    echo "SGLang ready: router=${SGLANG_ROUTER_IP}:${SGLANG_ROUTER_PORT}"
else
    echo "[Step 1/3] Using provided SGLang router: ${SGLANG_ROUTER_IP}:${SGLANG_ROUTER_PORT}"
fi

# Verify router is reachable
ROUTER_URL="http://${SGLANG_ROUTER_IP}:${SGLANG_ROUTER_PORT}/health"
echo "Verifying SGLang router at ${ROUTER_URL}..."
MAX_WAIT=120
WAITED=0
while true; do
    if curl -sf "${ROUTER_URL}" > /dev/null 2>&1; then
        echo "SGLang router health check passed."
        break
    fi
    if [ "$WAITED" -ge "$MAX_WAIT" ]; then
        echo "ERROR: SGLang router not responding at ${ROUTER_URL}"
        echo "Check that GPU nodes are running and router port ${SGLANG_ROUTER_PORT} is accessible."
        exit 1
    fi
    sleep 5
    WAITED=$((WAITED + 5))
done

# =============================================================================
# Step 2: Register this KoS node
# =============================================================================
echo ""
echo "[Step 2/3] Registering KoS server..."

MY_IP=$(hostname -I | awk '{print $1}')
MY_KOS_URL="http://${MY_IP}:${KOS_PORT}"

mkdir -p "$(dirname "${KOS_ENV_FILE}")"

# Append this node's URL to the shared KoS env file (atomic via temp + mv).
# Multiple KoS nodes write to this file; the trainer reads it.
# Use a lock file to avoid concurrent write corruption on shared filesystems.
LOCK_FILE="${KOS_ENV_FILE}.lock"
(
    flock -w 30 200 || { echo "WARNING: Could not acquire lock, writing anyway"; }

    # Read existing URLs (if any)
    EXISTING_URLS=""
    if [ -f "${KOS_ENV_FILE}" ]; then
        EXISTING_URLS=$(grep '^KOS_REMOTE_URLS=' "${KOS_ENV_FILE}" 2>/dev/null | cut -d= -f2- | tr -d '"' || true)
    fi

    # Append our URL if not already present
    if echo "${EXISTING_URLS}" | grep -q "${MY_KOS_URL}"; then
        echo "  Already registered: ${MY_KOS_URL}"
    else
        if [ -n "${EXISTING_URLS}" ]; then
            NEW_URLS="${EXISTING_URLS},${MY_KOS_URL}"
        else
            NEW_URLS="${MY_KOS_URL}"
        fi
        # Write atomically
        TMP_FILE="${KOS_ENV_FILE}.tmp.$$"
        echo "KOS_REMOTE_URLS=\"${NEW_URLS}\"" > "${TMP_FILE}"
        echo "KOS_PORT=${KOS_PORT}" >> "${TMP_FILE}"
        mv "${TMP_FILE}" "${KOS_ENV_FILE}"
        echo "  Registered: ${MY_KOS_URL} (total: $(echo "${NEW_URLS}" | tr ',' '\n' | wc -l) servers)"
    fi
) 200>"${LOCK_FILE}"

echo "  KoS env file: ${KOS_ENV_FILE}"

# =============================================================================
# Step 3: Launch KoS Server
# =============================================================================
echo ""
echo "[Step 3/3] Launching KoS remote rollout server..."

# Create run directories
mkdir -p "${RUN_DIR}/trajectories"
mkdir -p "${RUN_DIR}/logs/rollout_logs"
mkdir -p "${RUN_DIR}/logs/docker_logs"

# --- Start Docker daemon (needed for SWE tasks) ---
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

if [ "$KOS_IS_SWE_TASK" = true ] || [ "$KOS_IS_CGS_TASK" = true ]; then
    start_docker_daemon || { echo "FATAL: Docker required for SWE/CGS tasks"; exit 1; }
fi

# Load CGS Docker image from tar if specified
if [ -n "$KOS_CGS_DOCKER_TAR" ] && [ -f "$KOS_CGS_DOCKER_TAR" ]; then
    echo "Loading CGS Docker image from ${KOS_CGS_DOCKER_TAR}..."
    docker load -i "${KOS_CGS_DOCKER_TAR}"
    echo "CGS Docker image loaded."
fi

# --- Update model_config.json with SGLang router URL ---
SRC_CONFIG="${KOS_HOME}/model_config.json"
LOCAL_CONFIG="/tmp/kos_model_config_$(hostname).json"

if [ -f "$SRC_CONFIG" ]; then
    cp "$SRC_CONFIG" "$LOCAL_CONFIG"
    LLM_URL="http://${SGLANG_ROUTER_IP}:${SGLANG_ROUTER_PORT}/v1"
    echo "Updating model_config: base_url -> ${LLM_URL}"
    jq ".${KOS_MODEL_CONFIG}.client_args.base_url = \"${LLM_URL}\"" \
        "$LOCAL_CONFIG" > "${LOCAL_CONFIG}.tmp" && mv "${LOCAL_CONFIG}.tmp" "$LOCAL_CONFIG"
    WRITTEN_URL=$(jq -r ".${KOS_MODEL_CONFIG}.client_args.base_url" "$LOCAL_CONFIG" 2>/dev/null)
    echo "Verified model_config base_url: ${WRITTEN_URL}"
    export KOS_MODEL_CONFIG_PATH="$LOCAL_CONFIG"
else
    echo "WARNING: ${SRC_CONFIG} not found, skipping model config update"
fi

# --- Build args ---
SWE_ARGS=""
if [ "$KOS_IS_SWE_TASK" = true ]; then
    SWE_ARGS="--is_swe_task"
    if [ -n "$KOS_SWE_DOCKER_IMAGES" ]; then
        SWE_ARGS="${SWE_ARGS} --swe_docker_images_path ${KOS_SWE_DOCKER_IMAGES}"
    fi
fi

CGS_ARGS=""
if [ "$KOS_IS_CGS_TASK" = true ]; then
    CGS_ARGS="--is_cgs_task"
    if [ -n "$KOS_CGS_DOCKER_IMAGE" ]; then
        CGS_ARGS="${CGS_ARGS} --cgs_docker_image ${KOS_CGS_DOCKER_IMAGE}"
    fi
    if [ -n "$KOS_CGS_TARBALL_BASE" ]; then
        CGS_ARGS="${CGS_ARGS} --cgs_tarball_base ${KOS_CGS_TARBALL_BASE}"
    fi
    CGS_ARGS="${CGS_ARGS} --qwen_model"
    CGS_ARGS="${CGS_ARGS} --max_tool_calls_per_turn ${KOS_MAX_TOOL_CALLS_PER_TURN}"
    if [ "$KOS_USE_AGENT_JUDGE" = true ]; then
        CGS_ARGS="${CGS_ARGS} --use_agent_judge"
        CGS_ARGS="${CGS_ARGS} --judge_model_config ${KOS_JUDGE_MODEL_CONFIG}"
        if [ "$KOS_USE_JUDGE_AS_REWARD" = true ]; then
            CGS_ARGS="${CGS_ARGS} --use_judge_as_reward"
        fi
        if [ "$KOS_JUDGE_TRAJECTORY_MODE" = true ]; then
            CGS_ARGS="${CGS_ARGS} --judge_trajectory_mode"
        fi
    fi
fi

LOG_FILE="${RUN_DIR}/logs/rollout_logs/remote_rollout_$(hostname).log"

export PYTHONPATH="${KOS_HOME}:${PYTHONPATH:-}"
export HOME="${KOS_HOME}"
export DEFAULT_MODEL_PATH="${MODEL_PATH:-/mnt_out/songyanh/models/Qwen3-Coder-30B-A3B-Instruct}"
export AGENT_CONCURRENCY_MULTIPLIER=${KOS_AGENT_CONCURRENCY_MULTIPLIER}
export TEST_EXECUTOR_WORKERS=${KOS_TEST_EXECUTOR_WORKERS}
export TARGET_PATCH_COLUMN=${KOS_TARGET_PATCH_COLUMN}
export METRICS_DISCOUNT_FACTOR=${KOS_METRICS_DISCOUNT_FACTOR}
export METRICS_MAX_TRAJECTORY_LENGTH=${KOS_METRICS_MAX_TRAJECTORY_LENGTH}
export METRICS_MAX_RESPONSE_LENGTH=${METRICS_MAX_RESPONSE_LENGTH}
export MAX_CACHED_IMAGES=${MAX_CACHED_IMAGES:-10}
export SKIP_TEST_EXECUTION=${SKIP_TEST_EXECUTION:-0}

echo "=============================================="
echo "KoS Server (CPU Node)"
echo "=============================================="
echo "  This node:       ${MY_IP}:${KOS_PORT}"
echo "  SGLang Router:   ${SGLANG_ROUTER_IP}:${SGLANG_ROUTER_PORT}"
echo "  Workers/Node:    ${KOS_NUM_WORKERS}"
echo "  Max Concurrent:  ${KOS_MAX_CONCURRENT_AGENTS}"
echo "  Workspace:       ${KOS_WORKSPACE_PATH}"
echo "  Log:             ${LOG_FILE}"
echo "=============================================="

cd "${KOS_HOME}"
exec python3 "${KOS_HOME}/remote_rollout_server/sglang_remote_rollout.py" \
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
    ${CGS_ARGS} \
    2>&1 | tee "${LOG_FILE}"
