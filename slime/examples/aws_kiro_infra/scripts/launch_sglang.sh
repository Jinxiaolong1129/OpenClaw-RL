#!/bin/bash
# =============================================================================
# SGLang-only launcher (GPU pool)
# =============================================================================
#
# Runs on each pod of the SGLang PyTorchJob (k8s/launch_sglang_*.yaml).
# Mirrors Kiro's sglang_rollout_server/launch_sglang_only.sh. Head pod starts
# Ray + launches SGLang engines + router (via launch_external_sglang.py);
# worker pods join Ray and wait.
#
# Produces: <OUTPUT_DIR>/<RUN_NAME>/sglang_external_rollout.env
# =============================================================================

set -e

# ---- Defaults ----
MODEL_PATH=""
NUM_ENGINES=0       # 0 = auto from cluster GPU count
TP_SIZE=8
DP_SIZE=1
MEM_FRACTION=0.80
CHUNKED_PREFILL_SIZE=-1
ROUTER_PORT=30000
SERVER_BASE_PORT=13140
CONTEXT_LENGTH=""

RUN_NAME=${RUN_NAME:-"default"}
OUTPUT_DIR=${OUTPUT_DIR:-"/mnt_out/${USER:-default}/logs/swerl-aws-kiro"}
SLIME_DIR=${SLIME_DIR:-""}

NUM_GPUS_PER_NODE=${NUM_GPUS_PER_NODE:-$(nvidia-smi -L 2>/dev/null | wc -l || echo 8)}
MASTER_ADDR=${MASTER_ADDR:-""}

# ---- Parse args ----
while [[ $# -gt 0 ]]; do
  case $1 in
    --model-path)         MODEL_PATH="$2";         shift 2 ;;
    --num-engines)        NUM_ENGINES="$2";        shift 2 ;;
    --tp-size)            TP_SIZE="$2";            shift 2 ;;
    --dp-size)            DP_SIZE="$2";            shift 2 ;;
    --mem-fraction)       MEM_FRACTION="$2";       shift 2 ;;
    --chunked-prefill-size) CHUNKED_PREFILL_SIZE="$2"; shift 2 ;;
    --router-port)        ROUTER_PORT="$2";        shift 2 ;;
    --server-base-port)   SERVER_BASE_PORT="$2";   shift 2 ;;
    --context-length)     CONTEXT_LENGTH="$2";     shift 2 ;;
    --num-gpus-per-node)  NUM_GPUS_PER_NODE="$2";  shift 2 ;;
    --master-addr)        MASTER_ADDR="$2";        shift 2 ;;
    --run-name)           RUN_NAME="$2";           shift 2 ;;
    --output-dir)         OUTPUT_DIR="$2";         shift 2 ;;
    --slime-dir)          SLIME_DIR="$2";          shift 2 ;;
    --help|-h)
      echo "Usage: $0 --model-path <path> --run-name <name> --slime-dir <path> [options]"
      exit 0
      ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

[[ -z "${MODEL_PATH}" ]] && { echo "ERROR: --model-path required"; exit 1; }
[[ -z "${SLIME_DIR}"  ]] && { echo "ERROR: --slime-dir required";  exit 1; }

# ---- Derived paths ----
SCRIPT_DIR="${SLIME_DIR}/examples/kiro_agent/sglang_rollout_server"
RUN_DIR="${OUTPUT_DIR}/sglang/${RUN_NAME}"
ENV_DIR="${OUTPUT_DIR}/${RUN_NAME}"
SGLANG_ENV_FILE="${ENV_DIR}/sglang_external_rollout.env"

mkdir -p "${RUN_DIR}/logs" "${ENV_DIR}"

# ---- Cluster topology ----
if [ -n "${PET_NNODES:-}" ]; then
  echo "PyTorchJob detected (PET_NNODES=${PET_NNODES}, RANK=${PET_NODE_RANK})"
  WORLD_SIZE=${PET_NNODES}
  [[ -z "${MASTER_ADDR}" ]] && MASTER_ADDR=$(hostname | sed -E 's/-worker-[0-9]+$/-worker-0/')
  MY_HOST=$(hostname)
  [[ "${MASTER_ADDR}" == "${MY_HOST}" ]] && IS_MASTER=true || IS_MASTER=false
else
  IS_MASTER=true
  [[ -z "${MASTER_ADDR}" ]] && MASTER_ADDR=$(hostname -I | awk '{print $1}')
  WORLD_SIZE=1
fi
MY_IP=$(hostname -i | awk '{print $1}')

GPUS_PER_ENGINE=$((TP_SIZE * DP_SIZE))
EXPECTED_GPUS=$((WORLD_SIZE * NUM_GPUS_PER_NODE))
if [ "${NUM_ENGINES}" -le 0 ]; then
  NUM_ENGINES=$((EXPECTED_GPUS / GPUS_PER_ENGINE))
fi

echo "=============================================="
echo "SGLang-only launcher"
echo "  RUN_NAME     : ${RUN_NAME}"
echo "  World size   : ${WORLD_SIZE} pods × ${NUM_GPUS_PER_NODE} GPU"
echo "  Engines      : ${NUM_ENGINES} (TP=${TP_SIZE}, DP=${DP_SIZE})"
echo "  Model        : ${MODEL_PATH}"
echo "  Master       : ${MASTER_ADDR}"
echo "  Is master    : ${IS_MASTER}"
echo "  Env file     : ${SGLANG_ENV_FILE}"
echo "=============================================="

# ---- Worker path ----
if [[ "${IS_MASTER}" == "false" ]]; then
  echo "[Worker] Waiting for Ray head at ${MASTER_ADDR}:6379..."
  MAX_WAIT=2400; WAITED=0
  until ray health-check --address="${MASTER_ADDR}:6379" 2>/dev/null; do
    [ "${WAITED}" -ge "${MAX_WAIT}" ] && { echo "ERROR: Ray head timeout"; exit 1; }
    sleep 5; WAITED=$((WAITED + 5))
  done

  ray start --address="${MASTER_ADDR}:6379" \
    --node-ip-address="${MY_IP}" \
    --num-gpus="${NUM_GPUS_PER_NODE}" \
    --object-store-memory=200000000000 \
    --disable-usage-stats

  # Wait for env file so worker can log "ready" before going idle
  rm -f "${SGLANG_ENV_FILE}"
  MAX_WAIT=1800; WAITED=0
  while [ ! -f "${SGLANG_ENV_FILE}" ]; do
    [ "${WAITED}" -ge "${MAX_WAIT}" ] && { echo "ERROR: env file timeout"; exit 1; }
    sleep 10; WAITED=$((WAITED + 10))
    [ $((WAITED % 60)) -eq 0 ] && echo "  waiting... (${WAITED}s)"
  done
  source "${SGLANG_ENV_FILE}"
  echo "[Worker] SGLang ready. Router=${SGLANG_ROUTER_IP}:${SGLANG_ROUTER_PORT}"
  sleep infinity
  exit 0
fi

# ---- Head path ----
echo ""
echo "[Step 1/3] Start Ray head..."
ray stop --force 2>/dev/null || true
sleep 2
export RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE=1
ray start --head \
  --node-ip-address="${MY_IP}" \
  --num-gpus="${NUM_GPUS_PER_NODE}" \
  --object-store-memory=200000000000 \
  --disable-usage-stats \
  --dashboard-host=0.0.0.0 --dashboard-port=8265

echo ""
echo "[Step 2/3] Wait for ${WORLD_SIZE} Ray nodes..."
MAX_WAIT=600; WAITED=0
while true; do
  COUNT=$(python3 -c "
import ray
try:
    ray.init(address='auto', ignore_reinit_error=True)
    print(sum(1 for n in ray.nodes() if n.get('Alive', False)))
except Exception:
    print(0)
" 2>/dev/null || echo 0)
  [ "${COUNT}" -ge "${WORLD_SIZE}" ] && { echo "  All ${COUNT}/${WORLD_SIZE} joined"; break; }
  [ "${WAITED}" -ge "${MAX_WAIT}" ] && { echo "ERROR: Ray timeout"; ray status; exit 1; }
  sleep 5; WAITED=$((WAITED + 5))
done
ray status

echo ""
echo "[Step 3/3] Launch SGLang engines + router (via Kiro's launch_external_sglang.py)..."
rm -f "${SGLANG_ENV_FILE}"

CMD="python3 ${SCRIPT_DIR}/launch_external_sglang.py"
CMD="${CMD} --model-path ${MODEL_PATH}"
CMD="${CMD} --num-engines ${NUM_ENGINES}"
CMD="${CMD} --tp-size ${TP_SIZE}"
CMD="${CMD} --dp-size ${DP_SIZE}"
CMD="${CMD} --mem-fraction-static ${MEM_FRACTION}"
CMD="${CMD} --chunked-prefill-size ${CHUNKED_PREFILL_SIZE}"
CMD="${CMD} --router-port ${ROUTER_PORT}"
CMD="${CMD} --server-base-port ${SERVER_BASE_PORT}"
CMD="${CMD} --output-dir ${OUTPUT_DIR}"
if [ -n "${RUN_NAME}" ] && [ "${RUN_NAME}" != "default" ]; then
  CMD="${CMD} --run-name ${RUN_NAME}"
fi
if [ -n "${CONTEXT_LENGTH}" ]; then
  CMD="${CMD} --context-length ${CONTEXT_LENGTH}"
fi

LOG="${RUN_DIR}/logs/sglang_launcher.log"
echo "  Cmd: ${CMD}"
echo "  Log: ${LOG}"
${CMD} > "${LOG}" 2>&1 &
SGLANG_PID=$!
echo "  PID: ${SGLANG_PID}"

echo "  Waiting for ${SGLANG_ENV_FILE}..."
MAX_WAIT=1800; WAITED=0
while [ ! -f "${SGLANG_ENV_FILE}" ]; do
  kill -0 "${SGLANG_PID}" 2>/dev/null || { echo "ERROR: launcher died"; tail -100 "${LOG}"; exit 1; }
  [ "${WAITED}" -ge "${MAX_WAIT}" ] && { echo "ERROR: timeout"; tail -100 "${LOG}"; exit 1; }
  sleep 10; WAITED=$((WAITED + 10))
  echo "    (${WAITED}s)"
done
source "${SGLANG_ENV_FILE}"
echo ""
echo "SGLang ready!"
echo "  Router : ${SGLANG_ROUTER_IP}:${SGLANG_ROUTER_PORT}"
echo "  Engines: ${SGLANG_ENGINE_ADDRS}"

# Router health check
for i in $(seq 1 10); do
  if curl -fsS --max-time 5 "http://${SGLANG_ROUTER_IP}:${SGLANG_ROUTER_PORT}/health" >/dev/null 2>&1; then
    echo "Router /health OK"; break
  fi
  [ "${i}" -eq 10 ] && { echo "ERROR: router not responding"; exit 1; }
  sleep 3
done

echo ""
echo "=== SGLang services running ==="
echo "  Env file     : ${SGLANG_ENV_FILE}"
echo "  Router       : http://${SGLANG_ROUTER_IP}:${SGLANG_ROUTER_PORT}"
echo "==============================="

# Trap cleanup
cleanup() {
  kill -9 "${SGLANG_PID}" 2>/dev/null || true
  pkill -9 -f sglang 2>/dev/null || true
  ray stop --force 2>/dev/null || true
  exit 0
}
trap cleanup SIGINT SIGTERM

wait "${SGLANG_PID}"
