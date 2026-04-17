#!/usr/bin/env bash
# =============================================================================
# Download SWE-Bench / SWE-Gym docker images AS TARBALLS for AWS PVC
# =============================================================================
#
# Produces:
#   <OUT_DIR>/<instance_id>.tar.gz    (one per instance)
#
# The aws_kiro_infra CPU agent pod expects this layout:
#   - ${SWE_DOCKER_IMAGES_PATH} env var points to the directory
#   - server/docker_ops.py::load_image_from_tar calls `docker load -i <tar>`
#     on demand when a trajectory needs that instance
#
# Why tarballs (not just `docker pull`)?
#   - Each K8s CPU pod has its own dockerd (state doesn't persist)
#   - PVC is the only shared medium across pods
#   - Pre-downloaded tarballs avoid registry pull during training (saves
#     1-3 min per cold instance + immune to registry rate limits)
#
# Usage:
#   # Reads instance IDs from a parquet/JSONL. Auto-detects data_source.
#   bash download_swe_images.sh --prompt-data /path/to/train.parquet \
#                               --output-dir /mnt_out/.../sweap_images_800
#
#   # Smoke test: first 10 only
#   bash download_swe_images.sh --prompt-data train.parquet --max 10 \
#                               --output-dir /tmp/swe_images_10
#
# Env overrides:
#   PROXY_PREFIX_SWE_GYM, PROXY_PREFIX_SWE_BENCH   — registry mirror hostnames
#   MAX_RETRIES, RETRY_SLEEP                       — pull retry policy
#   PARALLEL                                        — parallel downloads (default 4)
#   KEEP_LOCAL_IMAGE                                — 1 = keep in dockerd after save
# =============================================================================

set -euo pipefail

PROMPT_DATA=""
OUTPUT_DIR=""
MAX=0
PARALLEL=${PARALLEL:-4}

# Canonical image names (what CPU agent pod's `docker_ops.py` expects with --pull never).
# These are the tags that end up inside the .tar.gz via `docker save`, so agent pods
# can `docker load` → `docker run <canonical>` without any re-tagging.
CANONICAL_PREFIX_SWE_GYM="docker.io/xingyaoww"
CANONICAL_PREFIX_SWE_BENCH="docker.io/swebench"

# Proxy / mirror to pull FROM (defaults to canonical, override for China / rate limit).
PROXY_PREFIX_SWE_GYM=${PROXY_PREFIX_SWE_GYM:-${CANONICAL_PREFIX_SWE_GYM}}
PROXY_PREFIX_SWE_BENCH=${PROXY_PREFIX_SWE_BENCH:-${CANONICAL_PREFIX_SWE_BENCH}}
MAX_RETRIES=${MAX_RETRIES:-5}
RETRY_SLEEP=${RETRY_SLEEP:-15}
KEEP_LOCAL_IMAGE=${KEEP_LOCAL_IMAGE:-0}

while [[ $# -gt 0 ]]; do
  case $1 in
    --prompt-data) PROMPT_DATA="$2"; shift 2 ;;
    --output-dir)  OUTPUT_DIR="$2";  shift 2 ;;
    --max)         MAX="$2";         shift 2 ;;
    --parallel)    PARALLEL="$2";    shift 2 ;;
    --help|-h)
      echo "Usage: $0 --prompt-data <parquet|jsonl> --output-dir <dir> [--max N] [--parallel N]"
      exit 0
      ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

[[ -z "${PROMPT_DATA}" ]] && { echo "ERROR: --prompt-data required"; exit 1; }
[[ -z "${OUTPUT_DIR}"  ]] && { echo "ERROR: --output-dir required";  exit 1; }
[[ ! -f "${PROMPT_DATA}" ]] && { echo "ERROR: ${PROMPT_DATA} not found"; exit 1; }

mkdir -p "${OUTPUT_DIR}"
LOG_FILE="${OUTPUT_DIR}/_download_$(date +%F_%H%M%S).log"
echo "Log: ${LOG_FILE}"

# ---- Extract (instance_id, data_source) pairs via Python ----
MANIFEST="${OUTPUT_DIR}/_manifest.tsv"
python3 - <<PYEOF > "${MANIFEST}"
import sys, json
from pathlib import Path

p = Path("${PROMPT_DATA}")
records = []
if p.suffix == ".parquet":
    import pandas as pd
    df = pd.read_parquet(p)
    for _, row in df.iterrows():
        inst = row["instance"]
        iid = inst["instance_id"] if hasattr(inst, "__getitem__") else inst.get("instance_id")
        ds = row.get("data_source", "swe-gym")
        records.append((iid, ds))
elif p.suffix in (".jsonl", ".json"):
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            r = json.loads(line)
            meta = r.get("metadata", {})
            inst = meta.get("instance", {})
            iid = inst.get("instance_id", "")
            ds = meta.get("data_source", "swe-gym")
            if iid:
                records.append((iid, ds))
else:
    print(f"ERROR: unsupported format {p.suffix}", file=sys.stderr)
    sys.exit(1)

n = int("${MAX}")
if n > 0:
    records = records[:n]

for iid, ds in records:
    print(f"{iid}\t{ds}")
PYEOF

TOTAL=$(wc -l < "${MANIFEST}")
echo "Manifest: ${TOTAL} instances → ${MANIFEST}"

# ---- Helper: resolve (proxy_image, canonical_image) from (iid, data_source) ----
# Prints "<pull_from>\t<save_as>" — two tab-separated URLs.
resolve_image() {
  local iid="$1"
  local ds="$2"
  local compat proxy canonical
  if [[ "$(echo "$ds" | tr '[:upper:]' '[:lower:]')" == *"swe-bench"* ]]; then
    compat=$(echo "${iid}" | tr '[:upper:]' '[:lower:]' | sed 's/__/_1776_/g')
    proxy="${PROXY_PREFIX_SWE_BENCH}/sweb.eval.x86_64.${compat}:latest"
    canonical="${CANONICAL_PREFIX_SWE_BENCH}/sweb.eval.x86_64.${compat}:latest"
  else
    compat=$(echo "${iid}" | tr '[:upper:]' '[:lower:]' | sed 's/__/_s_/g')
    proxy="${PROXY_PREFIX_SWE_GYM}/sweb.eval.x86_64.${compat}:latest"
    canonical="${CANONICAL_PREFIX_SWE_GYM}/sweb.eval.x86_64.${compat}:latest"
  fi
  printf '%s\t%s\n' "${proxy}" "${canonical}"
}

# ---- Process one instance: pull → save → gzip → cleanup ----
process_one() {
  local iid="$1"
  local ds="$2"
  local tar_path="${OUTPUT_DIR}/${iid}.tar.gz"

  # Skip if already saved
  if [[ -s "${tar_path}" ]]; then
    echo "  skip (exists): ${iid}"
    return 0
  fi

  local line proxy_image canonical_image
  line=$(resolve_image "${iid}" "${ds}")
  proxy_image=$(echo "${line}" | cut -f1)
  canonical_image=$(echo "${line}" | cut -f2)

  # Pull with retry (from proxy/mirror)
  local ok=false
  for attempt in $(seq 1 "${MAX_RETRIES}"); do
    if docker pull "${proxy_image}" >/dev/null 2>&1; then
      ok=true
      break
    fi
    echo "  pull attempt ${attempt}/${MAX_RETRIES} failed for ${iid}  ← ${proxy_image}"
    [[ ${attempt} -lt ${MAX_RETRIES} ]] && sleep "${RETRY_SLEEP}"
  done
  if ! ${ok}; then
    echo "  FAILED (pull): ${iid}  ← ${proxy_image}"
    return 1
  fi

  # Re-tag proxy → canonical so the saved tarball carries the canonical name
  # (agent pod's docker_ops.py expects `docker.io/xingyaoww/...` via --pull never)
  if [[ "${proxy_image}" != "${canonical_image}" ]]; then
    if ! docker tag "${proxy_image}" "${canonical_image}"; then
      echo "  FAILED (tag): ${iid}  ← ${proxy_image} → ${canonical_image}"
      return 1
    fi
  fi

  # Save to gzip tarball (under canonical name)
  if docker save "${canonical_image}" | gzip -1 > "${tar_path}.tmp"; then
    mv "${tar_path}.tmp" "${tar_path}"
    local size=$(du -h "${tar_path}" | cut -f1)
    echo "  OK: ${iid} (${size})  [canonical=${canonical_image}]"
  else
    echo "  FAILED (save): ${iid}"
    rm -f "${tar_path}.tmp"
    return 1
  fi

  # Cleanup both tags to save dockerd disk
  if [[ "${KEEP_LOCAL_IMAGE}" != "1" ]]; then
    docker rmi "${proxy_image}"     >/dev/null 2>&1 || true
    if [[ "${proxy_image}" != "${canonical_image}" ]]; then
      docker rmi "${canonical_image}" >/dev/null 2>&1 || true
    fi
  fi
}

export -f process_one resolve_image
export PROXY_PREFIX_SWE_GYM PROXY_PREFIX_SWE_BENCH OUTPUT_DIR MAX_RETRIES RETRY_SLEEP KEEP_LOCAL_IMAGE
export CANONICAL_PREFIX_SWE_GYM CANONICAL_PREFIX_SWE_BENCH

echo "Starting ${TOTAL} downloads (parallel=${PARALLEL})..."
echo "========================================"

# Parallel download via xargs
cat "${MANIFEST}" | awk -F'\t' '{print $1" "$2}' | \
  xargs -P "${PARALLEL}" -I {} bash -c 'process_one $1 $2' _ {} \
  2>&1 | tee -a "${LOG_FILE}"

echo "========================================"
N_DONE=$(ls "${OUTPUT_DIR}"/*.tar.gz 2>/dev/null | wc -l)
echo "Completed: ${N_DONE}/${TOTAL} tarballs in ${OUTPUT_DIR}"
TOTAL_SIZE=$(du -sh "${OUTPUT_DIR}" | cut -f1)
echo "Total size: ${TOTAL_SIZE}"
