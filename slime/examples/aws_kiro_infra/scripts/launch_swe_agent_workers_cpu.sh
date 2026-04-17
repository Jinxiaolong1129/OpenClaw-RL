#!/bin/bash
# =============================================================================
# CPU agent server launcher (mini-swe-agent + docker + FastAPI)
# =============================================================================
#
# Runs on every pod of the CPU worker PyTorchJob
# (k8s/launch_swe_agent_workers.yaml). Mirrors Kiro's
# sglang_rollout_server/launch_kos_only.sh in structure: each pod starts
# dockerd + a FastAPI server, then appends its URL to ``swe_agents.env``
# via flock. Trainer reads that file to discover all workers.
#
# Differences from Kiro's launch_kos_only.sh:
#   * KoS FastAPI (Strands agent) → swe_agent_server (mini-swe-agent scaffold)
#   * `sglang_remote_rollout.py` → `server.swe_agent_server:app` (uvicorn)
#   * KoS env name ``kos_servers.env`` → ``swe_agents.env``
#
# Usage:
#   bash launch_swe_agent_workers_cpu.sh \
#       --sglang-env-file <path>/sglang_external_rollout.env \
#       --model-path /mnt_out/.../Qwen3-4B-Instruct-2507 \
#       --swe-rl-dir  /mnt_out/.../OpenClaw-RL/swe-rl \
#       --aws-kiro-dir /mnt_out/.../aws_kiro_infra
# =============================================================================

set -e

# ---- Defaults ----
SGLANG_ENV_FILE=${SGLANG_ENV_FILE:-""}
SGLANG_ROUTER_IP=${SGLANG_ROUTER_IP:-""}
SGLANG_ROUTER_PORT=${SGLANG_ROUTER_PORT:-"30000"}

MODEL_PATH=${MODEL_PATH:-""}
AGENT_PORT=${AGENT_PORT:-5000}
AGENT_MAX_CONCURRENT=${AGENT_MAX_CONCURRENT:-16}

RUN_NAME=${RUN_NAME:-"default"}
OUTPUT_DIR=${OUTPUT_DIR:-"/mnt_out/${USER:-default}/logs/swerl-aws-kiro"}
SWE_RL_DIR=${SWE_RL_DIR:-""}
AWS_KIRO_DIR=${AWS_KIRO_DIR:-""}
SLIME_DIR=${SLIME_DIR:-""}

SWE_DOCKER_IMAGES_PATH=${SWE_DOCKER_IMAGES_PATH:-""}
SWE_CONFIG_PATH=${SWE_CONFIG_PATH:-""}
SWE_CONTAINER_MEMORY=${SWE_CONTAINER_MEMORY:-"8g"}
SWE_CONTAINER_PIDS_LIMIT=${SWE_CONTAINER_PIDS_LIMIT:-"1024"}

# ---- Parse args ----
while [[ $# -gt 0 ]]; do
  case $1 in
    --sglang-env-file)    SGLANG_ENV_FILE="$2";     shift 2 ;;
    --sglang-router-ip)   SGLANG_ROUTER_IP="$2";    shift 2 ;;
    --sglang-router-port) SGLANG_ROUTER_PORT="$2";  shift 2 ;;
    --model-path)         MODEL_PATH="$2";          shift 2 ;;
    --agent-port)         AGENT_PORT="$2";          shift 2 ;;
    --agent-max-concurrent) AGENT_MAX_CONCURRENT="$2"; shift 2 ;;
    --run-name)           RUN_NAME="$2";            shift 2 ;;
    --output-dir)         OUTPUT_DIR="$2";          shift 2 ;;
    --swe-rl-dir)         SWE_RL_DIR="$2";          shift 2 ;;
    --aws-kiro-dir)       AWS_KIRO_DIR="$2";        shift 2 ;;
    --slime-dir)          SLIME_DIR="$2";           shift 2 ;;
    --swe-docker-images)  SWE_DOCKER_IMAGES_PATH="$2"; shift 2 ;;
    --swe-config-path)    SWE_CONFIG_PATH="$2";     shift 2 ;;
    --help|-h)
      echo "Usage: $0 --sglang-env-file <file> --model-path <path> --aws-kiro-dir <dir> [options]"
      exit 0
      ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

[[ -z "${MODEL_PATH}"    ]] && { echo "ERROR: --model-path required";    exit 1; }
[[ -z "${AWS_KIRO_DIR}"  ]] && { echo "ERROR: --aws-kiro-dir required";  exit 1; }
# SWE_RL_DIR dependency removed — self-contained under slime/examples/

# ---- Derived paths ----
RUN_DIR="${OUTPUT_DIR}/swe_agents/${RUN_NAME}"
ENV_DIR="${OUTPUT_DIR}/${RUN_NAME}"
AGENT_ENV_FILE="${ENV_DIR}/swe_agents.env"

mkdir -p "${RUN_DIR}/logs" "${ENV_DIR}"

if [ -z "${SGLANG_ENV_FILE}" ] && [ -z "${SGLANG_ROUTER_IP}" ]; then
  SGLANG_ENV_FILE="${ENV_DIR}/sglang_external_rollout.env"
  echo "(defaulting --sglang-env-file to ${SGLANG_ENV_FILE})"
fi

# ---- Step 1: resolve SGLang router ----
if [ -n "${SGLANG_ENV_FILE}" ] && [ -z "${SGLANG_ROUTER_IP}" ]; then
  echo "[Step 1/4] Waiting for ${SGLANG_ENV_FILE}..."
  MAX_WAIT=1800; WAITED=0
  while [ ! -f "${SGLANG_ENV_FILE}" ]; do
    [ "${WAITED}" -ge "${MAX_WAIT}" ] && { echo "ERROR: SGLang env file timeout"; exit 1; }
    sleep 10; WAITED=$((WAITED + 10))
    [ $((WAITED % 60)) -eq 0 ] && echo "  waiting for GPU pool... (${WAITED}s)"
  done
  source "${SGLANG_ENV_FILE}"
  echo "  SGLang ready: router=${SGLANG_ROUTER_IP}:${SGLANG_ROUTER_PORT}"
fi

# Verify router reachability
ROUTER_URL="http://${SGLANG_ROUTER_IP}:${SGLANG_ROUTER_PORT}/health"
echo "  Verifying router at ${ROUTER_URL}..."
MAX_WAIT=120; WAITED=0
until curl -fsS --max-time 5 "${ROUTER_URL}" >/dev/null 2>&1; do
  [ "${WAITED}" -ge "${MAX_WAIT}" ] && { echo "ERROR: router unreachable"; exit 1; }
  sleep 5; WAITED=$((WAITED + 5))
done
echo "  Router health OK"

# ---- Step 2: start dockerd (privileged pod required) ----
echo ""
echo "[Step 2/4] Starting dockerd..."
if ! docker info &>/dev/null; then
  dockerd --data-root=/dev/shm/docker_data &>/tmp/dockerd.log &
  DOCKERD_PID=$!
  for i in $(seq 1 30); do
    sleep 1
    docker info &>/dev/null && { echo "  dockerd up"; break; }
    [ "${i}" -eq 30 ] && { echo "ERROR: dockerd failed"; cat /tmp/dockerd.log; exit 1; }
  done
else
  echo "  dockerd already running"
fi

# ---- Step 3: register this worker's URL via flock ----
echo ""
echo "[Step 3/4] Registering worker URL..."
MY_IP=$(hostname -I | awk '{print $1}')
MY_URL="http://${MY_IP}:${AGENT_PORT}"
LOCK_FILE="${AGENT_ENV_FILE}.lock"
(
  flock -w 30 200 || { echo "  WARNING: could not acquire lock, writing anyway"; }
  EXISTING=""
  if [ -f "${AGENT_ENV_FILE}" ]; then
    EXISTING=$(grep '^SWE_AGENT_URLS=' "${AGENT_ENV_FILE}" 2>/dev/null | cut -d= -f2- | tr -d '"' || true)
  fi
  if echo "${EXISTING}" | grep -q "${MY_URL}"; then
    echo "  already registered: ${MY_URL}"
  else
    if [ -n "${EXISTING}" ]; then
      NEW="${EXISTING},${MY_URL}"
    else
      NEW="${MY_URL}"
    fi
    TMP="${AGENT_ENV_FILE}.tmp.$$"
    {
      echo "SWE_AGENT_URLS=\"${NEW}\""
      echo "SWE_AGENT_PORT=${AGENT_PORT}"
    } > "${TMP}"
    mv "${TMP}" "${AGENT_ENV_FILE}"
    echo "  registered: ${MY_URL} (total=$(echo "${NEW}" | tr ',' '\n' | wc -l))"
  fi
) 200>"${LOCK_FILE}"
echo "  swe_agents env: ${AGENT_ENV_FILE}"

# ---- Step 4: start the FastAPI agent server ----
echo ""
echo "[Step 4/4] Launching swe_agent_server on :${AGENT_PORT}..."

export MODEL_PATH
export SWE_MAX_CONCURRENT="${AGENT_MAX_CONCURRENT}"
[ -n "${SWE_DOCKER_IMAGES_PATH}" ] && export SWE_DOCKER_IMAGES_PATH
[ -n "${SWE_CONFIG_PATH}"        ] && export SWE_CONFIG_PATH
export SWE_CONTAINER_MEMORY SWE_CONTAINER_PIDS_LIMIT

# PYTHONPATH: swe-rl (for swe_utils) + aws_kiro_infra parent (for server package)
export PYTHONPATH="${AWS_KIRO_DIR}/..:${PYTHONPATH:-}"

LOG_FILE="${RUN_DIR}/logs/agent_$(hostname)_$(date +%Y%m%d_%H%M%S).log"
echo "  Log: ${LOG_FILE}"

# Use the fully qualified module path so FastAPI imports cleanly.
cd "${AWS_KIRO_DIR}/.."
exec python3 -m uvicorn \
  aws_kiro_infra.server.swe_agent_server:app \
  --host 0.0.0.0 --port "${AGENT_PORT}" \
  --log-level info \
  2>&1 | tee "${LOG_FILE}"
