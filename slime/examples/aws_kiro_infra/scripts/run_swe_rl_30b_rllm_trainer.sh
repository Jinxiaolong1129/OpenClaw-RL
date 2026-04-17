#!/bin/bash
# =============================================================================
# 30B trainer — rllm / DeepSWE GRPO++ recipe, SYNC mode
# =============================================================================
# 对齐 rllm/DeepSWE 论文配方（`rllm/examples/swe/train_deepswe_32b.sh`）:
#   * Scaffold:      mini-swe-agent (bash backtick) — 我们自己的,不动
#   * Dataset:       R2E-Gym (默认 R2E-Gym-V1 4578 条, data_source=r2e-gym)
#   * Sync (train.py),无 update_weights_interval
#   * advantage_estimator: grpo without std normalization (≈ LOOP / Dr.GRPO)
#   * Clip: asymmetric 0.2 / 0.28 (DAPO high clip)
#   * No KL loss (strict DeepSWE)
#   * No entropy loss
#   * Dr.GRPO length norm (custom pg-loss reducer)
#   * Compact filter: 只有 voluntary submit 的 traj 进梯度
#                     (实现在 generate_kiro_rllm.generate)
#   * DAPO oversample: over-sampling-batch-size + check_reward_nonzero_std filter
#
# 关键差异 vs Kiro 30B 脚本 (run_swe_rl_30b_kiro_trainer.sh):
#   - train.py (sync) vs train_async.py
#   - 无 --use-tis
#   - 无 --use-kl-loss, kl_loss_coef=0
#   - 对称 0.2/0.2 → 非对称 0.2/0.28
#   - 无 --calculate-per-token-loss (被 dr_grpo_length_norm_reducer 替代)
#   - 加 --grpo-std-normalization False
#   - 加 --custom-pg-loss-reducer-function-path
#   - 加 --over-sampling-batch-size
#   - 加 SWE_RL_MAX_RESPONSE_LENGTH env
#   - custom-generate → generate_kiro_rllm (含 compact filter)
# =============================================================================

set -ex
set -o pipefail

export PYTHONBUFFERED=16

SLIME_DIR=${SLIME_DIR:?"ERROR: SLIME_DIR"}
# SWE_RL_DIR removed — self-contained under slime/examples/
AWS_KIRO_DIR=${AWS_KIRO_DIR:-"${SLIME_DIR}/examples/aws_kiro_infra"}
MEGATRON_LM_PATH=${MEGATRON_LM_PATH:?"ERROR: MEGATRON_LM_PATH"}

# ---- 两个协调 env 文件 ----
SGLANG_ENV_BASE=${SGLANG_ENV_BASE:?"ERROR: SGLANG_ENV_BASE"}
SGLANG_RUN_NAME=${SGLANG_RUN_NAME:?"ERROR: SGLANG_RUN_NAME"}
SWE_AGENT_RUN_NAME=${SWE_AGENT_RUN_NAME:?"ERROR: SWE_AGENT_RUN_NAME"}
RUN_NAME=${RUN_NAME:-"swerl_30b_rllm"}

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

RUN_TIMESTAMP=${RUN_TIMESTAMP:-$(date +%F_%H%M%S)}
export MODEL_ARGS_ROTARY_BASE=${MODEL_ARGS_ROTARY_BASE:-10000000}
source "${SLIME_DIR}/scripts/models/qwen3-30B-A3B.sh"

export NUM_NODES=${NUM_NODES:-${PET_NNODES:-16}}
export NUM_GPUS_PER_NODE=${NUM_GPUS_PER_NODE:-8}

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
echo "Trainer 30B (rllm / DeepSWE recipe, SYNC)"
echo "  NUM_NODES       : ${NUM_NODES}"
echo "  MASTER_ADDR     : ${MASTER_ADDR}"
echo "  SGLang router   : ${SGLANG_ROUTER_IP}:${SGLANG_ROUTER_PORT}"
echo "  swe_agent URLs  : ${SWE_AGENT_URLS}"
echo "=============================================="

export SWE_REMOTE_MAX_RETRIES=${SWE_REMOTE_MAX_RETRIES:-3}
export SWE_REMOTE_HTTP_TIMEOUT=${SWE_REMOTE_HTTP_TIMEOUT:-3600}
export SWE_MAX_CONCURRENT=${SWE_MAX_CONCURRENT:-1024}
export PYTHONUNBUFFERED=1
export RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE=1

# ★ Dr.GRPO length norm reducer 读这个
export SWE_RL_MAX_RESPONSE_LENGTH=${SWE_RL_MAX_RESPONSE_LENGTH:-32768}

# ---- Checkpoints ----
EXPORT_ROOT=${EXPORT_ROOT:-"/mnt_out/${USER:-default}/ckpt/swerl-aws-kiro"}
mkdir -p "${EXPORT_ROOT}"
HF_CKPT=${HF_CKPT:?"ERROR: HF_CKPT"}
REF_LOAD=${REF_LOAD:?"ERROR: REF_LOAD (mcore-converted 30B)"}
SAVE_CKPT=${SAVE_CKPT:-"${EXPORT_ROOT}/${RUN_NAME}"}
CKPT_ARGS=(
  --hf-checkpoint "${HF_CKPT}"
  --ref-load "${REF_LOAD}"
  --load "${SAVE_CKPT}"
  --save "${SAVE_CKPT}"
  --save-interval ${SAVE_INTERVAL:-10}
)

# ---- Dataset (default R2E-Gym V1) ----
PROMPT_DATA=${PROMPT_DATA:?"ERROR: PROMPT_DATA"}
[[ ! -f "${PROMPT_DATA}" ]] && { echo "ERROR: missing ${PROMPT_DATA}"; exit 1; }
NUM_ROLLOUT=${NUM_ROLLOUT:-1000}

ROLLOUT_ARGS=(
  --prompt-data "${PROMPT_DATA}"
  --input-key prompt
  --metadata-key instance
  --apply-chat-template
  --rollout-shuffle
  --num-rollout "${NUM_ROLLOUT}"
  --rollout-batch-size ${ROLLOUT_BATCH_SIZE:-8}       # DeepSWE 官方脚本值
  --n-samples-per-prompt ${N_SAMPLES_PER_PROMPT:-8}   # DeepSWE 官方脚本值
  --rollout-max-response-len ${ROLLOUT_MAX_RESPONSE_LEN:-32768}
  --rollout-temperature ${ROLLOUT_TEMPERATURE:-1.0}
  --num-steps-per-rollout 1
  --balance-data
  # ★ DAPO oversample + filter (rejection sampling)
  --over-sampling-batch-size ${OVER_SAMPLING_BATCH_SIZE:-16}
  --dynamic-sampling-filter-path slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std
)

TENSOR_MODEL_PARALLEL_SIZE=${TENSOR_MODEL_PARALLEL_SIZE:-4}
SEQUENCE_PARALLEL=${SEQUENCE_PARALLEL:-true}
PIPELINE_MODEL_PARALLEL_SIZE=${PIPELINE_MODEL_PARALLEL_SIZE:-2}
CONTEXT_PARALLEL_SIZE=${CONTEXT_PARALLEL_SIZE:-8}
EXPERT_MODEL_PARALLEL_SIZE=${EXPERT_MODEL_PARALLEL_SIZE:-8}
EXPERT_TENSOR_PARALLEL_SIZE=${EXPERT_TENSOR_PARALLEL_SIZE:-1}
RECOMPUTE_GRANULARITY=${RECOMPUTE_GRANULARITY:-full}
RECOMPUTE_METHOD=${RECOMPUTE_METHOD:-uniform}
RECOMPUTE_NUM_LAYERS=${RECOMPUTE_NUM_LAYERS:-1}
USE_DYNAMIC_BATCH_SIZE=${USE_DYNAMIC_BATCH_SIZE:-true}
MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU:-32768}
LOG_PROBS_MAX_TOKENS_PER_GPU=${LOG_PROBS_MAX_TOKENS_PER_GPU:-65536}
LOG_PROBS_CHUNK_SIZE=${LOG_PROBS_CHUNK_SIZE:-512}

PERF_ARGS=(
  --tensor-model-parallel-size ${TENSOR_MODEL_PARALLEL_SIZE}
  --pipeline-model-parallel-size ${PIPELINE_MODEL_PARALLEL_SIZE}
  --context-parallel-size ${CONTEXT_PARALLEL_SIZE}
  --expert-model-parallel-size ${EXPERT_MODEL_PARALLEL_SIZE}
  --expert-tensor-parallel-size ${EXPERT_TENSOR_PARALLEL_SIZE}
  --recompute-granularity ${RECOMPUTE_GRANULARITY}
  --recompute-method ${RECOMPUTE_METHOD}
  --recompute-num-layers ${RECOMPUTE_NUM_LAYERS}
  --max-tokens-per-gpu ${MAX_TOKENS_PER_GPU}
  --log-probs-max-tokens-per-gpu ${LOG_PROBS_MAX_TOKENS_PER_GPU}
  --log-probs-chunk-size ${LOG_PROBS_CHUNK_SIZE}
)
[ "${SEQUENCE_PARALLEL}" = "true" ] && PERF_ARGS+=(--sequence-parallel)
[ "${USE_DYNAMIC_BATCH_SIZE}" = "true" ] && PERF_ARGS+=(--use-dynamic-batch-size)

# ---- GRPO (rllm/DeepSWE GRPO++ recipe) ----
GRPO_ARGS=(
  --advantage-estimator grpo
  --grpo-std-normalization False              # ★ Dr.GRPO: 去掉 std 除法 (≈ LOOP)
  # 无 --use-kl-loss (strict DeepSWE)
  --entropy-coef 0.0                          # ★ 无 entropy loss
  --eps-clip 0.2
  --eps-clip-high 0.28                        # ★ DAPO 非对称 clip
  # ★ Dr.GRPO length norm (seq-mean-token-sum / MAX_RESP 常量)
  --custom-pg-loss-reducer-function-path swe_rl_loss_reducers.dr_grpo_length_norm_reducer
)

OPTIMIZER_ARGS=(
  --optimizer adam
  --lr ${LR:-1e-6}
  --lr-decay-style constant
  --weight-decay 0.01
  --adam-beta1 0.9
  --adam-beta2 0.98
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

# ★ rllm 变体:compact filter 在 generate 里做,不是 reward
CUSTOM_ARGS=(
  --custom-generate-function-path generate_kiro_rllm.generate
  --custom-rm-path generate_kiro_rllm.reward_func
)

EVAL_ARGS=()
if [ -n "${EVAL_PROMPT_DATA:-}" ]; then
  EVAL_ARGS=(
    --eval-interval ${EVAL_INTERVAL:-400}
    --eval-prompt-data r2e_val "${EVAL_PROMPT_DATA}"
    --n-samples-per-eval-prompt ${N_SAMPLES_PER_EVAL_PROMPT:-8}
    --eval-max-response-len ${EVAL_MAX_RESPONSE_LEN:-32768}
    --eval-top-p 1
  )
fi

WANDB_KEY_VALUE=${WANDB_KEY:-${WANDB_API_KEY:-}}
if [ -n "${WANDB_KEY_VALUE}" ]; then
  WANDB_ARGS=(
    --use-wandb
    --wandb-project "${WANDB_PROJECT:-slime_swe_30b_rllm}"
    --wandb-group qwen3-30B-A3B-rllm-deepswe
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
  --no-check-for-nan-in-loss-and-grad
)

SWE_SAVE_TRAJ_DIR=${SWE_SAVE_TRAJ_DIR:-"${EXPORT_ROOT}/trajectories/${RUN_NAME}_${RUN_TIMESTAMP}"}
mkdir -p "${SWE_SAVE_TRAJ_DIR}"
export SWE_SAVE_TRAJ_DIR
export SWE_ROLLOUT_TIMEOUT=${SWE_ROLLOUT_TIMEOUT:-3600}
export SWE_EVAL_TIMEOUT=${SWE_EVAL_TIMEOUT:-600}

# ---- Ray cluster (trainer 自己的,和 SGLang 独立) ----
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

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${MEGATRON_LM_PATH}:${AWS_KIRO_DIR}:${SLIME_DIR}\",
    \"PYTHONUNBUFFERED\": \"1\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"PYTORCH_ALLOC_CONF\": \"expandable_segments:True\",
    \"PYTORCH_CUDA_ALLOC_CONF\": \"expandable_segments:True\",
    \"MASTER_ADDR\": \"${MASTER_ADDR}\",
    \"NCCL_SOCKET_IFNAME\": \"^lo,docker0,veth_def_agent\",
    \"SWE_AGENT_URLS\": \"${SWE_AGENT_URLS}\",
    \"SWE_REMOTE_MAX_RETRIES\": \"${SWE_REMOTE_MAX_RETRIES}\",
    \"SWE_REMOTE_HTTP_TIMEOUT\": \"${SWE_REMOTE_HTTP_TIMEOUT}\",
    \"SWE_MAX_CONCURRENT\": \"${SWE_MAX_CONCURRENT}\",
    \"SWE_SAVE_TRAJ_DIR\": \"${SWE_SAVE_TRAJ_DIR}\",
    \"SWE_ROLLOUT_TIMEOUT\": \"${SWE_ROLLOUT_TIMEOUT}\",
    \"SWE_EVAL_TIMEOUT\": \"${SWE_EVAL_TIMEOUT}\",
    \"SWE_RL_MAX_RESPONSE_LENGTH\": \"${SWE_RL_MAX_RESPONSE_LENGTH}\",
    \"HF_HOME\": \"${HF_HOME:-/mnt_out/${USER:-default}/hf_cache}\",
    \"WANDB_API_KEY\": \"${WANDB_API_KEY:-}\"
  }
}"

RAY_JOB_SUBMISSION_ID=${RAY_JOB_SUBMISSION_ID:-"swerl_30b_rllm_${RUN_TIMESTAMP}"}

# ★ SYNC: train.py (不是 train_async.py)
ray job submit --address="http://127.0.0.1:8265" \
  --submission-id "${RAY_JOB_SUBMISSION_ID}" \
  --no-wait \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- python3 -u "${SLIME_DIR}/train.py" \
  --actor-num-nodes "${NUM_NODES}" \
  --actor-num-gpus-per-node "${NUM_GPUS_PER_NODE}" \
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
