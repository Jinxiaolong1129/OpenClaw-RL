#!/bin/bash
# =============================================================================
# 4B trainer — Kiro 3-pool topology, mini-swe-agent scaffold
# =============================================================================
# Hyperparameters: CURRENT swe-rl 4B recipe (DAPO asymmetric clip, sync
# train.py, trajectory-level aux LLM judge). Deliberately NOT aligned with
# Kiro 30B recipe — user opted to keep 4B as-is.
#
# Runs inside every pod of k8s/launch_trainer_4b.yaml. Reads:
#   - <SGLANG_ENV_BASE>/<SGLANG_RUN_NAME>/sglang_external_rollout.env   (from SGLang pool)
#   - <SGLANG_ENV_BASE>/<SWE_AGENT_RUN_NAME>/swe_agents.env             (from CPU agent pool)
# =============================================================================

set -ex
set -o pipefail

# ---- Required paths ----
SLIME_DIR=${SLIME_DIR:?"ERROR: SLIME_DIR"}
# SWE_RL_DIR removed — self-contained under slime/examples/
AWS_KIRO_DIR=${AWS_KIRO_DIR:-"${SLIME_DIR}/examples/aws_kiro_infra"}
MEGATRON_LM_PATH=${MEGATRON_LM_PATH:?"ERROR: MEGATRON_LM_PATH"}

# ---- Two coordination env files (from 2 separate rollout jobs) ----
SGLANG_ENV_BASE=${SGLANG_ENV_BASE:?"ERROR: SGLANG_ENV_BASE"}
SGLANG_RUN_NAME=${SGLANG_RUN_NAME:?"ERROR: SGLANG_RUN_NAME (rollout SGLang job)"}
SWE_AGENT_RUN_NAME=${SWE_AGENT_RUN_NAME:?"ERROR: SWE_AGENT_RUN_NAME (CPU agent job)"}
RUN_NAME=${RUN_NAME:-"swerl_4b_train"}

SGLANG_ENV_FILE="${SGLANG_ENV_BASE}/${SGLANG_RUN_NAME}/sglang_external_rollout.env"
SWE_AGENT_ENV_FILE="${SGLANG_ENV_BASE}/${SWE_AGENT_RUN_NAME}/swe_agents.env"

[[ ! -f "${SGLANG_ENV_FILE}"    ]] && { echo "ERROR: missing ${SGLANG_ENV_FILE}";    exit 1; }
[[ ! -f "${SWE_AGENT_ENV_FILE}" ]] && { echo "ERROR: missing ${SWE_AGENT_ENV_FILE}"; exit 1; }

source "${SGLANG_ENV_FILE}"
source "${SWE_AGENT_ENV_FILE}"

SGLANG_ROUTER_IP=${SGLANG_ROUTER_IP:?"ERROR: SGLANG_ROUTER_IP unset"}
SGLANG_ROUTER_PORT=${SGLANG_ROUTER_PORT:?"ERROR: SGLANG_ROUTER_PORT unset"}
SGLANG_ENGINE_ADDRS=${SGLANG_ENGINE_ADDRS:?"ERROR: SGLANG_ENGINE_ADDRS unset"}
SGLANG_GPUS_PER_ENGINE=${SGLANG_GPUS_PER_ENGINE:?"ERROR: SGLANG_GPUS_PER_ENGINE unset"}
SWE_AGENT_URLS=${SWE_AGENT_URLS:?"ERROR: SWE_AGENT_URLS unset"}
export SWE_AGENT_URLS

# ---- Log + model args ----
RUN_TIMESTAMP=${RUN_TIMESTAMP:-$(date +%F_%H%M%S)}
source "${SLIME_DIR}/scripts/models/qwen3-4B-Instruct-2507.sh"

export NUM_NODES=${NUM_NODES:-${PET_NNODES:-2}}
export NUM_GPUS_PER_NODE=${NUM_GPUS_PER_NODE:-8}
export ACTOR_GPUS_PER_NODE=${ACTOR_GPUS_PER_NODE:-${NUM_GPUS_PER_NODE}}

if [ -n "${PET_NNODES:-}" ]; then
  export MASTER_ADDR=${MASTER_ADDR:-$(hostname | sed -E 's/-worker-[0-9]+$/-worker-0/')}
  export MY_ADDR=$(hostname)
  export NODE_RANK=${PET_NODE_RANK:-0}
else
  export MASTER_ADDR=${MASTER_ADDR:-$(hostname -I | awk '{print $1}')}
  export MY_ADDR=${MY_ADDR:-$(hostname -I | awk '{print $1}')}
  export NODE_RANK=${NODE_RANK:-0}
fi
MY_IP=$(hostname -i | awk '{print $1}')

echo "=============================================="
echo "Trainer 4B"
echo "  NUM_NODES           : ${NUM_NODES}"
echo "  MASTER_ADDR         : ${MASTER_ADDR}"
echo "  SGLang router       : ${SGLANG_ROUTER_IP}:${SGLANG_ROUTER_PORT}"
echo "  SGLang engines      : ${SGLANG_ENGINE_ADDRS}"
echo "  swe_agent servers   : ${SWE_AGENT_URLS}"
echo "=============================================="

# ---- SWE-RL env ----
export SWE_REMOTE_MAX_RETRIES=${SWE_REMOTE_MAX_RETRIES:-3}
export SWE_REMOTE_HTTP_TIMEOUT=${SWE_REMOTE_HTTP_TIMEOUT:-3600}
export SWE_MAX_CONCURRENT=${SWE_MAX_CONCURRENT:-128}

export AUX_JUDGE_MODEL=${AUX_JUDGE_MODEL:-"openai/gpt-4o-mini"}
export AUX_JUDGE_NUM_VOTES=${AUX_JUDGE_NUM_VOTES:-1}
export AUX_JUDGE_TEMPERATURE=${AUX_JUDGE_TEMPERATURE:-0.0}
export AUX_JUDGE_MAX_TOKENS=${AUX_JUDGE_MAX_TOKENS:-512}
export AUX_JUDGE_MAX_CONCURRENCY=${AUX_JUDGE_MAX_CONCURRENCY:-16}
export AUX_JUDGE_TIMEOUT=${AUX_JUDGE_TIMEOUT:-120}

export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE=1

# ---- Checkpoints ----
EXPORT_ROOT=${EXPORT_ROOT:-"/mnt_out/${USER:-default}/ckpt/swerl-aws-kiro"}
mkdir -p "${EXPORT_ROOT}"
HF_CKPT=${HF_CKPT:?"ERROR: HF_CKPT"}
REF_LOAD=${REF_LOAD:-${HF_CKPT}}
SAVE_CKPT=${SAVE_CKPT:-"${EXPORT_ROOT}/${RUN_NAME}"}
CKPT_ARGS=(
  --hf-checkpoint "${HF_CKPT}"
  --ref-load "${REF_LOAD}"
  --save "${SAVE_CKPT}"
  --save-interval ${SAVE_INTERVAL:-10}
  --megatron-to-hf-mode bridge
)

PROMPT_DATA=${PROMPT_DATA:?"ERROR: PROMPT_DATA"}
[[ ! -f "${PROMPT_DATA}" ]] && { echo "ERROR: missing ${PROMPT_DATA}"; exit 1; }
NUM_ROLLOUT=${NUM_ROLLOUT:-500}

ROLLOUT_ARGS=(
  --prompt-data "${PROMPT_DATA}"
  --input-key prompt
  --metadata-key instance
  --rollout-shuffle
  --reward-key score
  --num-rollout "${NUM_ROLLOUT}"
  --rollout-batch-size ${ROLLOUT_BATCH_SIZE:-4}
  --n-samples-per-prompt ${N_SAMPLES_PER_PROMPT:-4}
  --rollout-max-response-len ${ROLLOUT_MAX_RESPONSE_LEN:-4096}
  --rollout-max-context-len ${ROLLOUT_MAX_CONTEXT_LEN:-32768}
  --rollout-temperature ${ROLLOUT_TEMPERATURE:-1.0}
  --rollout-top-p ${ROLLOUT_TOP_P:-0.95}
  --num-steps-per-rollout 1
  --dynamic-sampling-filter-path "slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std"
)

PERF_ARGS=(
  --tensor-model-parallel-size ${TENSOR_MODEL_PARALLEL_SIZE:-4}
  --sequence-parallel
  --pipeline-model-parallel-size ${PIPELINE_MODEL_PARALLEL_SIZE:-1}
  --context-parallel-size ${CONTEXT_PARALLEL_SIZE:-1}
  --expert-model-parallel-size ${EXPERT_MODEL_PARALLEL_SIZE:-1}
  --expert-tensor-parallel-size ${EXPERT_TENSOR_PARALLEL_SIZE:-1}
  --recompute-granularity ${RECOMPUTE_GRANULARITY:-full}
  --recompute-method ${RECOMPUTE_METHOD:-uniform}
  --recompute-num-layers ${RECOMPUTE_NUM_LAYERS:-1}
  --use-dynamic-batch-size
  --max-tokens-per-gpu ${MAX_TOKENS_PER_GPU:-9216}
  --log-probs-chunk-size ${LOG_PROBS_CHUNK_SIZE:-1024}
  --balance-data
)

# ---- GRPO (4B recipe: DAPO asymmetric) ----
GRPO_ARGS=(
  --advantage-estimator grpo
  --use-kl-loss
  --kl-loss-coef ${KL_LOSS_COEF:-0.0}
  --kl-loss-type low_var_kl
  --entropy-coef 0.00
  --eps-clip 0.2
  --eps-clip-high 0.28
)

AUX_REWARD_ARGS=(
  --aux-reward-enable
  --aux-reward-coef ${AUX_REWARD_COEF:-0.5}
  --aux-judge-max-problem-len 8000
  --aux-judge-max-steps-shown 30
  --aux-judge-max-output-per-step 1200
  --aux-judge-max-patch-chars 4000
)

OPTIMIZER_ARGS=(
  --optimizer adam
  --lr ${LR:-1e-6}
  --lr-decay-style constant
  --weight-decay 0.01
  --adam-beta1 0.9
  --adam-beta2 0.999
  --optimizer-cpu-offload
  --overlap-cpu-optimizer-d2h-h2d
  --use-precision-aware-optimizer
)

SGLANG_ARGS=(
  --rollout-external
  --rollout-external-engine-addrs ${SGLANG_ENGINE_ADDRS}
  --sglang-router-ip ${SGLANG_ROUTER_IP}
  --sglang-router-port ${SGLANG_ROUTER_PORT}
  --rollout-num-gpus-per-engine ${SGLANG_GPUS_PER_ENGINE}
)

CUSTOM_ARGS=(
  --custom-generate-function-path generate_kiro.generate
  --custom-rm-path generate_kiro.reward_func
)

WANDB_KEY_VALUE=${WANDB_KEY:-${WANDB_API_KEY:-}}
if [ -n "${WANDB_KEY_VALUE}" ]; then
  WANDB_ARGS=(
    --use-wandb
    --wandb-project "${WANDB_PROJECT:-slime_swe}"
    --wandb-group qwen3-4B-instruct-swe-rl-kiro-aws
    --wandb-key "${WANDB_KEY_VALUE}"
  )
else
  WANDB_ARGS=()
fi

MISC_ARGS=(
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32
  --attention-backend flash
)

SWE_SAVE_TRAJ_DIR=${SWE_SAVE_TRAJ_DIR:-"${EXPORT_ROOT}/trajectories/${RUN_NAME}_${RUN_TIMESTAMP}"}
mkdir -p "${SWE_SAVE_TRAJ_DIR}"
export SWE_SAVE_TRAJ_DIR
export SWE_ROLLOUT_TIMEOUT=${SWE_ROLLOUT_TIMEOUT:-1800}
export SWE_EVAL_TIMEOUT=${SWE_EVAL_TIMEOUT:-300}

# ---- Ray (trainer's own cluster) ----
export no_proxy="127.0.0.1,${MASTER_ADDR}${no_proxy:+,${no_proxy}}"

if [[ "${NODE_RANK}" == "0" ]]; then
  ray stop --force 2>/dev/null || true
  sleep 2
  ray start --head --node-ip-address="${MY_IP}" --num-gpus="${NUM_GPUS_PER_NODE}" \
    --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265 \
    --object-store-memory=200000000000

  if [ "${NUM_NODES}" -gt 1 ]; then
    echo "Waiting for ${NUM_NODES} trainer pods..."
    MAX_WAIT=2400; WAITED=0
    while true; do
      count=$(ray status 2>/dev/null | grep -Eo '^ *1 node_[0-9a-f]+' | wc -l)
      [ "${count}" -ge "${NUM_NODES}" ] && { echo "All ${count}/${NUM_NODES} joined"; break; }
      [ "${WAITED}" -ge "${MAX_WAIT}" ] && { echo "ERROR: ray timeout"; ray status; exit 1; }
      sleep 5; WAITED=$((WAITED + 5))
    done
  fi
else
  echo "Trainer worker — waiting for head at ${MASTER_ADDR}:6379..."
  until ray health-check --address="${MASTER_ADDR}:6379" 2>/dev/null; do sleep 5; done
  ray start --address="${MASTER_ADDR}:6379" --num-gpus="${NUM_GPUS_PER_NODE}" \
    --node-ip-address="${MY_IP}" --object-store-memory=200000000000 --disable-usage-stats
  echo "Worker joined. Sleeping to keep Ray alive."
  while ray status >/dev/null 2>&1; do sleep 60; done
  exit 0
fi

ray status

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
HAS_NVLINK=$([ "${NVLINK_COUNT}" -gt 0 ] && echo 1 || echo 0)
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-"max_split_size_mb:2048"}

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${MEGATRON_LM_PATH}:${AWS_KIRO_DIR}:${SLIME_DIR}\",
    \"PYTHONUNBUFFERED\": \"1\",
    \"PYTHONFAULTHANDLER\": \"1\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"PYTORCH_CUDA_ALLOC_CONF\": \"${PYTORCH_CUDA_ALLOC_CONF}\",
    \"MASTER_ADDR\": \"${MASTER_ADDR}\",
    \"SWE_AGENT_URLS\": \"${SWE_AGENT_URLS}\",
    \"SWE_REMOTE_MAX_RETRIES\": \"${SWE_REMOTE_MAX_RETRIES}\",
    \"SWE_REMOTE_HTTP_TIMEOUT\": \"${SWE_REMOTE_HTTP_TIMEOUT}\",
    \"SWE_MAX_CONCURRENT\": \"${SWE_MAX_CONCURRENT}\",
    \"SWE_SAVE_TRAJ_DIR\": \"${SWE_SAVE_TRAJ_DIR}\",
    \"SWE_ROLLOUT_TIMEOUT\": \"${SWE_ROLLOUT_TIMEOUT}\",
    \"SWE_EVAL_TIMEOUT\": \"${SWE_EVAL_TIMEOUT}\",
    \"AUX_JUDGE_MODEL\": \"${AUX_JUDGE_MODEL}\",
    \"AUX_JUDGE_NUM_VOTES\": \"${AUX_JUDGE_NUM_VOTES}\",
    \"AUX_JUDGE_TEMPERATURE\": \"${AUX_JUDGE_TEMPERATURE}\",
    \"AUX_JUDGE_MAX_TOKENS\": \"${AUX_JUDGE_MAX_TOKENS}\",
    \"AUX_JUDGE_MAX_CONCURRENCY\": \"${AUX_JUDGE_MAX_CONCURRENCY}\",
    \"AUX_JUDGE_TIMEOUT\": \"${AUX_JUDGE_TIMEOUT}\",
    \"OPENAI_API_KEY\": \"${OPENAI_API_KEY:-}\",
    \"ANTHROPIC_API_KEY\": \"${ANTHROPIC_API_KEY:-}\",
    \"HF_HOME\": \"${HF_HOME:-/mnt_out/${USER:-default}/hf_cache}\",
    \"WANDB_API_KEY\": \"${WANDB_API_KEY:-}\"
  }
}"

RAY_JOB_SUBMISSION_ID=${RAY_JOB_SUBMISSION_ID:-"swerl_4b_trainer_${RUN_TIMESTAMP}"}

ray job submit --address="http://127.0.0.1:8265" \
  --submission-id "${RAY_JOB_SUBMISSION_ID}" \
  --no-wait \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- python3 -u "${SLIME_DIR}/train.py" \
  --actor-num-nodes "${NUM_NODES}" \
  --actor-num-gpus-per-node "${ACTOR_GPUS_PER_NODE}" \
  ${MODEL_ARGS[@]} \
  ${CKPT_ARGS[@]} \
  ${ROLLOUT_ARGS[@]} \
  ${OPTIMIZER_ARGS[@]} \
  ${GRPO_ARGS[@]} \
  ${AUX_REWARD_ARGS[@]} \
  ${WANDB_ARGS[@]} \
  ${PERF_ARGS[@]} \
  ${SGLANG_ARGS[@]} \
  ${MISC_ARGS[@]} \
  ${CUSTOM_ARGS[@]}

set +e
ray job logs --address="http://127.0.0.1:8265" "${RAY_JOB_SUBMISSION_ID}" -f --log-style=record
RAY_STATUS_OUTPUT=$(ray job status --address="http://127.0.0.1:8265" "${RAY_JOB_SUBMISSION_ID}" --log-style=record 2>&1)
echo "${RAY_STATUS_OUTPUT}"
set -e
if [[ "${RAY_STATUS_OUTPUT}" == *"SUCCEEDED"* ]]; then
  exit 0
fi
exit 1
