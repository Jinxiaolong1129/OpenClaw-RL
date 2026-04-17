#!/bin/bash
# =============================================================================
# Auto-Restart Wrapper for SLIME Async Training
# =============================================================================
#
# Wraps run-qwen3-30b-kiro-kos-async-debug-run.sh with automatic restart on
# training failures (e.g., master node crash, NCCL timeout, OOM, Ray errors).
#
# On each failure the script:
#   1. Tears down the Ray cluster completely (head + workers)
#   2. Cleans up stale state (/dev/shm, Ray temp files)
#   3. Waits for a cooldown period before retrying
#   4. Re-launches the full training script (which rebuilds Ray + resumes
#      from the latest checkpoint via --load)
#
# The training script already handles checkpoint resume through --load, so
# restarting the whole script is sufficient to pick up from the last save.
#
# Usage:
#   bash run-qwen3-30b-kiro-kos-async-debug-run-autorestart.sh
#
# Environment variables (all optional):
#   MAX_RESTARTS        Max restart attempts (default: 50)
#   RESTART_COOLDOWN    Seconds to wait between restarts (default: 60)
#   RESTART_LOG_DIR     Directory for restart logs (default: $LOG_DIR/restarts)
# =============================================================================

set -euo pipefail

# =============================================================================
# Configuration
# =============================================================================
MAX_RESTARTS=${MAX_RESTARTS:-50}
RESTART_COOLDOWN=${RESTART_COOLDOWN:-60}

SLIME_DIR=${SLIME_DIR:-"/mnt_out/myshang/codebase/slime"}
RUN_NAME=${RUN_NAME:-"smoke_test_async"}
LOG_DIR=${LOG_DIR:-"/mnt_out/myshang/logs/slime"}
RESTART_LOG_DIR=${RESTART_LOG_DIR:-"${LOG_DIR}/restarts/${RUN_NAME}"}
KOS_BINARY_REWARD=${KOS_BINARY_REWARD:-"true"}
export KOS_BINARY_REWARD

TRAIN_SCRIPT="${SLIME_DIR}/examples/kiro_agent/run-qwen3-30b-kiro-kos-async-debug-run.sh"

mkdir -p "${RESTART_LOG_DIR}"

RESTART_COUNT=0
RESTART_STATE_FILE="${RESTART_LOG_DIR}/restart_state.txt"

# Resume restart count if we were previously tracking
if [ -f "${RESTART_STATE_FILE}" ]; then
    RESTART_COUNT=$(cat "${RESTART_STATE_FILE}" 2>/dev/null || echo 0)
    echo "[autorestart] Resuming from previous restart count: ${RESTART_COUNT}"
fi

# =============================================================================
# Cleanup function — kill Ray and stale processes on this node
# =============================================================================
cleanup_node() {
    echo "[autorestart] Cleaning up node..."

    # Stop Ray gracefully first, then force-kill
    ray stop --force 2>/dev/null || true
    sleep 2
    pkill -9 -f "ray" 2>/dev/null || true
    pkill -9 -f "sglang" 2>/dev/null || true
    # Don't kill ourselves
    pkill -9 -f "train_async" 2>/dev/null || true

    # Clean up /dev/shm (same as the YAML entrypoint does)
    rm -rf /dev/shm/containerd /dev/shm/containers /dev/shm/docker_data
    rm -rf /dev/shm/buildkit /dev/shm/rootfs /dev/shm/volumes
    rm -rf /dev/shm/plugins /dev/shm/network /dev/shm/swarm
    rm -rf /dev/shm/tmp /dev/shm/runtimes
    rm -f /dev/shm/engine-id /dev/shm/nccl-*

    # Clean up Ray temp files
    rm -rf /tmp/ray/* 2>/dev/null || true

    echo "[autorestart] Cleanup done."
}

# =============================================================================
# Also clean up workers if WORKER_IPS is set (manual multi-node mode)
# =============================================================================
cleanup_workers() {
    local NUM_GPUS_PER_NODE=${NUM_GPUS_PER_NODE:-8}
    if [ -n "${WORKER_IPS:-}" ]; then
        echo "[autorestart] Cleaning up worker nodes..."
        for WORKER_IP in ${WORKER_IPS}; do
            ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 root@"${WORKER_IP}" \
                "ray stop --force 2>/dev/null || true; \
                 pkill -9 ray 2>/dev/null || true; \
                 pkill -9 python 2>/dev/null || true; \
                 rm -rf /tmp/ray/* 2>/dev/null || true" 2>/dev/null &
        done
        wait
        echo "[autorestart] Worker cleanup done."
    fi
}

# =============================================================================
# Main restart loop
# =============================================================================
echo "=============================================="
echo "[autorestart] Auto-Restart Training Wrapper"
echo "  Train Script:    ${TRAIN_SCRIPT}"
echo "  Max Restarts:    ${MAX_RESTARTS}"
echo "  Cooldown:        ${RESTART_COOLDOWN}s"
echo "  Restart Log Dir: ${RESTART_LOG_DIR}"
echo "=============================================="

while true; do
    ATTEMPT=$((RESTART_COUNT + 1))
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    echo ""
    echo "[autorestart] ====== Attempt ${ATTEMPT} / $((MAX_RESTARTS + 1)) — ${TIMESTAMP} ======"

    # Run the training script, capture exit code
    EXIT_CODE=0
    bash "${TRAIN_SCRIPT}" || EXIT_CODE=$?

    if [ ${EXIT_CODE} -eq 0 ]; then
        echo "[autorestart] Training completed successfully."
        echo "${TIMESTAMP} attempt=${ATTEMPT} status=SUCCESS exit_code=0" >> "${RESTART_LOG_DIR}/restart_history.log"
        break
    fi

    # Training failed
    RESTART_COUNT=$((RESTART_COUNT + 1))
    echo "${RESTART_COUNT}" > "${RESTART_STATE_FILE}"
    echo "${TIMESTAMP} attempt=${ATTEMPT} status=FAILED exit_code=${EXIT_CODE}" >> "${RESTART_LOG_DIR}/restart_history.log"

    echo "[autorestart] Training failed with exit code ${EXIT_CODE} (failure #${RESTART_COUNT})"

    if [ ${RESTART_COUNT} -ge ${MAX_RESTARTS} ]; then
        echo "[autorestart] Reached max restarts (${MAX_RESTARTS}). Giving up."
        exit 1
    fi

    # Full cleanup before retry
    cleanup_node
    cleanup_workers

    echo "[autorestart] Waiting ${RESTART_COOLDOWN}s before restart..."
    sleep "${RESTART_COOLDOWN}"
done

echo "[autorestart] Done."
