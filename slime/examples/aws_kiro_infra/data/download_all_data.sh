#!/usr/bin/env bash
# =============================================================================
# 一键下载 30B 训练所需的全部数据（for teammate, no proxy needed）
# =============================================================================
#
# 做两件事:
#   Step 1. 从 HuggingFace 下 R2E-Gym-Subset → parquet (~200 MB, 几分钟)
#   Step 2. 从 Docker Hub 拉 4578 个 SWE-Bench image → tarball (~10-14 TB,
#           挂 nohup 后台,50-80 小时)
#
# 用法
# ----
#   bash data/download_all_data.sh --output-dir /data/r2egym_subset
#
# 选项
# ----
#   --output-dir <dir>        数据输出根目录(必填)
#   --parallel <N>            Docker pull 并发数(默认 20)
#   --max-samples <N>         只取前 N 条(smoke 用,默认 0=全量 4578)
#   --skip-parquet            跳过 Step 1(parquet 已存在)
#   --skip-images             跳过 Step 2(tarball 已存在)
#   --sync                    Step 2 前台跑(默认后台 nohup)
#
# 国内代理(可选,外网直连不用)
# ----------------------------
#   HTTP_PROXY=...    HTTPS_PROXY=...   ← HuggingFace 下 parquet 要走这个
#   PROXY_DOCKER_IO=...                 ← Docker pull 走这个(docker.io/x → <proxy>/x)
#
# 产出
# ----
#   <output-dir>/train.parquet           # Trainer 的 PROMPT_DATA
#   <output-dir>/images/<iid>.tar.gz     # CPU agent pod 的 SWE_DOCKER_IMAGES_PATH
#   <output-dir>/download.log            # tarball 下载日志
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Defaults
OUTPUT_DIR=""
PARALLEL=20
MAX_SAMPLES=0
SKIP_PARQUET=0
SKIP_IMAGES=0
RUN_SYNC=0

while [[ $# -gt 0 ]]; do
  case $1 in
    --output-dir)   OUTPUT_DIR="$2";   shift 2 ;;
    --parallel)     PARALLEL="$2";     shift 2 ;;
    --max-samples)  MAX_SAMPLES="$2";  shift 2 ;;
    --skip-parquet) SKIP_PARQUET=1;    shift ;;
    --skip-images)  SKIP_IMAGES=1;     shift ;;
    --sync)         RUN_SYNC=1;        shift ;;
    --help|-h)      sed -n '3,40p' "$0"; exit 0 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

[[ -z "${OUTPUT_DIR}" ]] && { echo "ERROR: --output-dir required"; exit 1; }

mkdir -p "${OUTPUT_DIR}" "${OUTPUT_DIR}/images"
PARQUET="${OUTPUT_DIR}/train.parquet"
TAR_DIR="${OUTPUT_DIR}/images"
LOG="${OUTPUT_DIR}/download.log"

echo "=============================================="
echo "R2E-Gym 全量数据一键下载"
echo "  output-dir : ${OUTPUT_DIR}"
echo "  parallel   : ${PARALLEL}"
echo "  max-samples: ${MAX_SAMPLES:-0} (0 = 全量 4578)"
echo "  proxy (HF) : ${HTTP_PROXY:-<none>}"
echo "  proxy (Docker): ${PROXY_DOCKER_IO:-<none, direct docker.io>}"
echo "=============================================="

# =============================================================================
# Step 1: HF → parquet
# =============================================================================
if [[ "${SKIP_PARQUET}" == "1" ]]; then
  echo ""
  echo "=== Step 1/2: SKIPPED (using existing ${PARQUET}) ==="
  [[ ! -f "${PARQUET}" ]] && { echo "ERROR: ${PARQUET} not found"; exit 1; }
else
  echo ""
  echo "=== Step 1/2: HuggingFace → parquet ==="
  PREPROCESS_ARGS=(--output-path "${PARQUET}")
  [[ ${MAX_SAMPLES} -gt 0 ]] && PREPROCESS_ARGS+=(--max-samples "${MAX_SAMPLES}" --streaming)

  python3 "${SCRIPT_DIR}/preprocess_r2egym.py" "${PREPROCESS_ARGS[@]}"
fi

# 校验
N_ROWS=$(python3 -c "import pandas as pd; print(len(pd.read_parquet('${PARQUET}')))")
echo "  parquet rows : ${N_ROWS}"

# =============================================================================
# Step 2: 拉 Docker images → tarballs
# =============================================================================
if [[ "${SKIP_IMAGES}" == "1" ]]; then
  echo ""
  echo "=== Step 2/2: SKIPPED ==="
  echo "数据准备完成"
  exit 0
fi

echo ""
echo "=== Step 2/2: Docker pull → tarballs ==="
echo "  parallel   : ${PARALLEL}"
echo "  target dir : ${TAR_DIR}"
echo "  log file   : ${LOG}"
echo ""

DOWNLOAD_ARGS=(
  --prompt-data "${PARQUET}"
  --output-dir  "${TAR_DIR}"
  --parallel    "${PARALLEL}"
)
[[ ${MAX_SAMPLES} -gt 0 ]] && DOWNLOAD_ARGS+=(--max "${MAX_SAMPLES}")

if [[ "${RUN_SYNC}" == "1" ]]; then
  echo "前台跑,Ctrl-C 可中断(但已下的 tarball 保留,重跑会跳过)"
  bash "${SCRIPT_DIR}/download_swe_images.sh" "${DOWNLOAD_ARGS[@]}" 2>&1 | tee "${LOG}"
else
  nohup bash "${SCRIPT_DIR}/download_swe_images.sh" "${DOWNLOAD_ARGS[@]}" \
    > "${LOG}" 2>&1 &
  PID=$!
  echo "后台 PID: ${PID}"
  echo ""
  echo "监控进度:"
  echo "  tail -f ${LOG}"
  echo "  watch -n 30 'ls ${TAR_DIR}/*.tar.gz | wc -l; du -sh ${TAR_DIR}'"
  echo ""
  echo "杀掉重来(已下的保留):"
  echo "  kill ${PID}"
fi

echo ""
echo "=============================================="
echo "完成后 parquet + tarballs 都在 ${OUTPUT_DIR}/"
echo "rsync 到 PVC 后,改 K8s YAML:"
echo "  PROMPT_DATA            = <PVC>/r2egym_subset_train.parquet"
echo "  SWE_DOCKER_IMAGES_PATH = <PVC>/r2egym_images/"
echo "=============================================="
