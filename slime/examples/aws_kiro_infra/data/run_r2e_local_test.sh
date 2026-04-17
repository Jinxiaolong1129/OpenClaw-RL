#!/usr/bin/env bash
# =============================================================================
# R2E-Gym 本地端到端冒烟测试(一键版)
# =============================================================================
#
# 3 步合一:
#   Step 1. 跑 preprocess_r2egym.py 产出小样本 parquet(默认 10 条)
#   Step 2. 下载第一条 instance 的 docker image tarball (支持代理 rewrite)
#   Step 3. 跑 test_r2e_eval_local.py 做完整 eval 验证
#
# 代理支持
# --------
# R2E-Gym 的 image 在 HF dataset 的 `docker_image` 列里显式给出,典型形式:
#   docker.io/namanjain12/<repo>_final:<tag>
#
# 如果你机器直连 docker.io 慢/被限流,可以设代理重写:
#   - PROXY_DOCKER_IO=<your-proxy-registry>   # 把 "docker.io" 前缀换成它
#     例如 PROXY_DOCKER_IO=slime-agent-cn-beijing.cr.volces.com
#
# 代理只影响 `docker pull` 的拉取源。`docker save` 时会 re-tag 回 canonical
# (docker.io/...) 名字, tarball 落盘的 image 名字和 aws_kiro_infra 运行时一致。
#
# Usage
# -----
# 默认(不用代理):
#     bash data/run_r2e_local_test.sh
#
# 使用代理:
#     PROXY_DOCKER_IO=slime-agent-cn-beijing.cr.volces.com \
#         bash data/run_r2e_local_test.sh
#
# 测试完清理 tarball + parquet(默认保留以便二次验证):
#     KEEP_ARTIFACTS=0 bash data/run_r2e_local_test.sh
#
# 环境变量
# --------
#   PROXY_DOCKER_IO      docker.io 前缀替换目标 (default: "")
#   R2E_DATA_SOURCE      HF dataset ID (default: R2E-Gym/R2E-Gym-Subset)
#   R2E_SPLIT            HF split (default: train)
#   MAX_SAMPLES          Parquet 行数 (default: 10)
#   OUT_ROOT             输出根目录 (default: /data_storage/wyj/jxl/OpenClaw-RL/data/r2egym_smoke)
#   INSTANCE_ID          指定测哪一条 (default: 第一行)
#   SKIP_PREPROCESS      =1 跳过 Step 1 (用已有 parquet)
#   SKIP_DOWNLOAD        =1 跳过 Step 2 (用已有 tarball)
#   KEEP_ARTIFACTS       =0 测完删 parquet + tarball + image (default: 1)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AWS_KIRO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ---- Config ----
PROXY_DOCKER_IO=${PROXY_DOCKER_IO:-""}
R2E_DATA_SOURCE=${R2E_DATA_SOURCE:-"R2E-Gym/R2E-Gym-Subset"}
R2E_SPLIT=${R2E_SPLIT:-"train"}
MAX_SAMPLES=${MAX_SAMPLES:-10}
OUT_ROOT=${OUT_ROOT:-"/data_storage/wyj/jxl/OpenClaw-RL/data/r2egym_smoke"}
INSTANCE_ID=${INSTANCE_ID:-""}
SKIP_PREPROCESS=${SKIP_PREPROCESS:-0}
SKIP_DOWNLOAD=${SKIP_DOWNLOAD:-0}
KEEP_ARTIFACTS=${KEEP_ARTIFACTS:-1}
STREAMING_PREPROCESS=${STREAMING_PREPROCESS:-1}

PARQUET_PATH="${OUT_ROOT}/train.parquet"
TAR_DIR="${OUT_ROOT}/images"
mkdir -p "${OUT_ROOT}" "${TAR_DIR}"

echo "=============================================="
echo "R2E-Gym 本地端到端测试"
echo "  HF dataset   : ${R2E_DATA_SOURCE} [${R2E_SPLIT}]"
echo "  max samples  : ${MAX_SAMPLES}"
echo "  output root  : ${OUT_ROOT}"
echo "  parquet      : ${PARQUET_PATH}"
echo "  tarball dir  : ${TAR_DIR}"
echo "  proxy rewrite: ${PROXY_DOCKER_IO:-<none, direct docker.io>}"
[[ -n "${INSTANCE_ID}" ]] && echo "  instance_id  : ${INSTANCE_ID}"
echo "=============================================="

# =============================================================================
# Step 1: preprocess → parquet
# =============================================================================
if [[ "${SKIP_PREPROCESS}" != "1" ]]; then
  echo ""
  echo "=== Step 1/3: preprocess_r2egym.py → parquet ==="
  PREPROCESS_ARGS=(
    --data-source "${R2E_DATA_SOURCE}"
    --split "${R2E_SPLIT}"
    --max-samples "${MAX_SAMPLES}"
    --output-path "${PARQUET_PATH}"
  )
  if [[ "${STREAMING_PREPROCESS}" == "1" ]]; then
    PREPROCESS_ARGS+=(--streaming)
    echo "  mode        : streaming"
  else
    echo "  mode        : materialized split"
  fi
  python3 "${AWS_KIRO_DIR}/data/preprocess_r2egym.py" \
    "${PREPROCESS_ARGS[@]}"
else
  echo ""
  echo "=== Step 1/3: SKIPPED (using existing ${PARQUET_PATH}) ==="
  [[ ! -f "${PARQUET_PATH}" ]] && { echo "ERROR: ${PARQUET_PATH} not found"; exit 1; }
fi

# =============================================================================
# Step 2: 挑一条 instance, docker pull → tag canonical → save → gzip
# =============================================================================
if [[ "${SKIP_DOWNLOAD}" != "1" ]]; then
  echo ""
  echo "=== Step 2/3: download image tarball ==="

  # 读第一条(或指定 instance)的 docker_image + instance_id
  read IID DOCKER_IMAGE < <(python3 <<PY
import pandas as pd, sys
df = pd.read_parquet("${PARQUET_PATH}")
target_iid = "${INSTANCE_ID}"
# 过滤 r2e 行
mask = df["data_source"].astype(str).str.contains("r2e", case=False, na=False)
r2e = df[mask]
if len(r2e) == 0:
    print("ERROR: no r2e rows in parquet", file=sys.stderr); sys.exit(1)
if target_iid:
    match = r2e[r2e["instance"].apply(lambda x: x["instance_id"]) == target_iid]
    if len(match) == 0:
        print(f"ERROR: instance_id={target_iid!r} not found in r2e rows", file=sys.stderr); sys.exit(1)
    row = match.iloc[0]
else:
    row = r2e.iloc[0]
inst = row["instance"]
print(inst["instance_id"], inst["docker_image"])
PY
)
  echo "  instance_id : ${IID}"
  echo "  docker_image: ${DOCKER_IMAGE}"

  TAR_PATH="${TAR_DIR}/${IID}.tar.gz"

  if [[ -s "${TAR_PATH}" ]]; then
    echo "  ★ tarball already exists: ${TAR_PATH} — skip pull/save"
  else
    # 代理 rewrite
    if [[ -n "${PROXY_DOCKER_IO}" ]]; then
      # 把 "docker.io/" 换成 "<PROXY_DOCKER_IO>/"
      PROXY_IMAGE=$(echo "${DOCKER_IMAGE}" | sed "s|^docker\.io/|${PROXY_DOCKER_IO}/|")
      if [[ "${PROXY_IMAGE}" == "${DOCKER_IMAGE}" ]]; then
        # docker.io 前缀可能被省略了(docker.io/foo ≡ foo); 手动加
        PROXY_IMAGE="${PROXY_DOCKER_IO}/${DOCKER_IMAGE#docker.io/}"
      fi
      echo "  pull (via proxy): ${PROXY_IMAGE}"
    else
      PROXY_IMAGE="${DOCKER_IMAGE}"
      echo "  pull: ${PROXY_IMAGE}"
    fi

    # 重试 3 次
    OK=0
    for attempt in 1 2 3; do
      if docker pull "${PROXY_IMAGE}"; then
        OK=1; break
      fi
      echo "  attempt ${attempt}/3 failed"
      [[ ${attempt} -lt 3 ]] && sleep 10
    done
    if [[ ${OK} -ne 1 ]]; then
      echo "ERROR: docker pull failed for ${PROXY_IMAGE}"
      echo "  提示: R2E-Gym image 可能不在你的代理 registry 里"
      echo "  检查 parquet 里 docker_image 字段,然后看代理 (${PROXY_DOCKER_IO}) 是否 mirror 了它"
      exit 1
    fi

    # 重 tag 到 canonical (docker.io/...) ,这样 tarball 里保存的是 canonical 名字
    if [[ "${PROXY_IMAGE}" != "${DOCKER_IMAGE}" ]]; then
      echo "  tag: ${PROXY_IMAGE} → ${DOCKER_IMAGE}"
      docker tag "${PROXY_IMAGE}" "${DOCKER_IMAGE}"
    fi

    # save + gzip
    echo "  save: ${TAR_PATH}"
    docker save "${DOCKER_IMAGE}" | gzip -1 > "${TAR_PATH}.tmp"
    mv "${TAR_PATH}.tmp" "${TAR_PATH}"
    SIZE=$(du -h "${TAR_PATH}" | cut -f1)
    echo "  OK (${SIZE})"

    # 清理本地 image (只保留 tarball,节省磁盘)
    docker rmi "${PROXY_IMAGE}" >/dev/null 2>&1 || true
    if [[ "${PROXY_IMAGE}" != "${DOCKER_IMAGE}" ]]; then
      docker rmi "${DOCKER_IMAGE}" >/dev/null 2>&1 || true
    fi
  fi
else
  echo ""
  echo "=== Step 2/3: SKIPPED (using existing tarballs in ${TAR_DIR}) ==="
fi

# =============================================================================
# Step 3: E2E test (scenario A/B/C)
# =============================================================================
echo ""
echo "=== Step 3/3: run test_r2e_eval_local.py ==="

TEST_ARGS=(
  --parquet "${PARQUET_PATH}"
  --tar-dir "${TAR_DIR}"
)
[[ -n "${INSTANCE_ID}" ]] && TEST_ARGS+=(--instance-id "${INSTANCE_ID}")

set +e
python3 "${AWS_KIRO_DIR}/data/test_r2e_eval_local.py" "${TEST_ARGS[@]}"
TEST_EXIT=$?
set -e

# =============================================================================
# Cleanup (可选)
# =============================================================================
if [[ "${KEEP_ARTIFACTS}" == "0" ]]; then
  echo ""
  echo "=== Cleanup (KEEP_ARTIFACTS=0) ==="
  rm -rf "${OUT_ROOT}"
  echo "  removed ${OUT_ROOT}"
fi

echo ""
echo "=============================================="
if [[ ${TEST_EXIT} -eq 0 ]]; then
  echo "★ ALL SCENARIOS PASSED ✓"
elif [[ ${TEST_EXIT} -eq 2 ]]; then
  echo "⚠️ eval scenarios finished but unexpected outcomes — see test output above"
else
  echo "✗ test script failed (exit=${TEST_EXIT}) — see output above"
fi
echo "=============================================="

exit ${TEST_EXIT}
