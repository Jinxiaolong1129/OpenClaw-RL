#!/bin/bash
# =============================================================================
# R2E-Gym 本机 reward 验证 v3
# =============================================================================
# 修复:
#   1. 创建 /testbed/r2e_tests symlink (R2E runtime setup_env 做的)
#   2. git clean 前备份 run_tests.sh, 之后恢复
#   3. gold patch 用 new_file_content 直接写文件(parsed_commit_content 是 JSON)
# =============================================================================

set -euo pipefail

SMOKE_DIR="/data_storage/wyj/jxl/OpenClaw-RL/data/r2egym_smoke"
TAR_DIR="${SMOKE_DIR}/images_r2e_5_unique"
PARQUET="${SMOKE_DIR}/train.parquet"
CONTAINER="r2e_reward_test"
AWS_KIRO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

IID="orange3_final__2d9617bd0cb1f0ba61771258410ab8fae8e7e24d"
TAR="${TAR_DIR}/${IID}.tar.gz"
IMAGE="namanjain12/orange3_final:2d9617bd0cb1f0ba61771258410ab8fae8e7e24d"

echo "=============================================="
echo "R2E-Gym reward 本机验证 v3"
echo "  instance : ${IID}"
echo "=============================================="

[[ ! -f "${TAR}" ]] && { echo "ERROR: tarball not found: ${TAR}"; exit 1; }

docker rm -f "${CONTAINER}" 2>/dev/null || true

# ---- 1. Load + Run + Setup ----
echo ""
echo "=== 1. docker load + run + R2E setup ==="
docker load -i "${TAR}" 2>&1 | tail -2
docker run -d --name "${CONTAINER}" --pull never "${IMAGE}" sleep 3600 | head -1
echo "container up"

# R2E runtime setup_env 做的关键步骤(我们手动补上):
# - 创建 /testbed/r2e_tests symlink(测试文件在 /r2e_tests, pytest 从 /testbed 跑)
# - 备份 run_tests.sh 到 /root/(防止 git clean 删除)
cat > /tmp/r2e_setup.sh <<'SETUP'
#!/bin/bash
# 模拟 R2E-Gym runtime setup_env()
cd /testbed

# 1. r2e_tests symlink: /r2e_tests → /testbed/r2e_tests
if [ -d /r2e_tests ] && [ ! -e /testbed/r2e_tests ]; then
  ln -s /r2e_tests /testbed/r2e_tests
  echo "SETUP: created /testbed/r2e_tests → /r2e_tests"
fi

# 2. 备份 run_tests.sh 到 /root(防止 git clean 删除)
if [ -f /testbed/run_tests.sh ]; then
  cp /testbed/run_tests.sh /root/run_tests.sh.bak
  echo "SETUP: backed up run_tests.sh to /root/"
fi

# 3. 确认 .venv 存在
ls -la /testbed/.venv/bin/python 2>/dev/null && echo "SETUP: .venv OK" || echo "SETUP: .venv not found (may be elsewhere)"

echo "SETUP: done"
SETUP
docker cp /tmp/r2e_setup.sh "${CONTAINER}:/tmp/r2e_setup.sh"
docker exec "${CONTAINER}" bash /tmp/r2e_setup.sh

# 确认 run_tests.sh
rm -f /tmp/r2e_check
docker cp "${CONTAINER}:/testbed/run_tests.sh" /tmp/r2e_check 2>/dev/null
echo "run_tests.sh: $(wc -c < /tmp/r2e_check 2>/dev/null || echo 'NOT FOUND') bytes"
cat /tmp/r2e_check 2>/dev/null || true
echo ""

# 确认 r2e_tests
docker exec "${CONTAINER}" bash -c "ls /testbed/r2e_tests/ 2>/dev/null | head -5 > /tmp/r2e_tests_ls.txt"
docker cp "${CONTAINER}:/tmp/r2e_tests_ls.txt" /tmp/r2e_tests_ls.txt 2>/dev/null
echo "r2e_tests contents:"
cat /tmp/r2e_tests_ls.txt 2>/dev/null || echo "(empty or not found)"

# ---- 2. Scenario A: 不打 patch ----
echo ""
echo "=== 2. Scenario A: 不打 patch ==="
cat > /tmp/r2e_run_A.sh <<'RUN'
#!/bin/bash
cd /testbed
bash /testbed/run_tests.sh > /tmp/eval_A.log 2>&1
echo "EXIT=$?" >> /tmp/eval_A.log
RUN
docker cp /tmp/r2e_run_A.sh "${CONTAINER}:/tmp/r2e_run_A.sh"
docker exec "${CONTAINER}" chmod +x /tmp/r2e_run_A.sh

echo "running tests (1-5 min)..."
docker exec "${CONTAINER}" bash /tmp/r2e_run_A.sh || true
sleep 2
docker cp "${CONTAINER}:/tmp/eval_A.log" /tmp/r2e_eval_A.log 2>/dev/null

if [[ -f /tmp/r2e_eval_A.log ]] && [[ -s /tmp/r2e_eval_A.log ]]; then
  echo "captured $(wc -l < /tmp/r2e_eval_A.log) lines"
  echo "--- last 30 lines ---"
  tail -30 /tmp/r2e_eval_A.log
  echo "--- end ---"
else
  echo "ERROR: no output captured"
  docker rm -f "${CONTAINER}" 2>/dev/null; exit 1
fi

echo ""
echo "=== 3. Grade Scenario A ==="
python3 <<PY
import sys; sys.path.insert(0, "${AWS_KIRO_DIR}")
from server.patch_utils import parse_r2e_pytest_log, grade_eval_output_r2e
import pandas as pd

with open("/tmp/r2e_eval_A.log") as f:
    output = f.read()
df = pd.read_parquet("${PARQUET}")
inst = df[df["instance"].apply(lambda x: x["instance_id"]) == "${IID}"].iloc[0]["instance"]

parsed = parse_r2e_pytest_log(output)
result = grade_eval_output_r2e(inst, output)
print(f"  parsed tests : {len(parsed)}")
print(f"  resolved     : {result['resolved']}")
print(f"  reason       : {result['report'].get('reason')}")
if parsed:
    print(f"  first 3 tests: {dict(list(parsed.items())[:3])}")
if not result['resolved'] and result['report'].get('mismatches'):
    print(f"  mismatches   : {len(result['report']['mismatches'])}")
print(f"  SCENARIO A   : {'PASS (reward=0)' if not result['resolved'] else '⚠ UNEXPECTED reward=1'}")
PY

# ---- 4. Scenario B: 写 gold 文件内容 ----
echo ""
echo "=== 4. Scenario B: apply gold patch ==="

# 从 parsed_commit_content 提取 new_file_content 并直接写入容器
# (不用 git apply — R2E 的 parsed_commit_content 是 JSON 不是 git diff)
python3 <<PY
import json, os, pandas as pd

df = pd.read_parquet("${PARQUET}")
inst = df[df["instance"].apply(lambda x: x["instance_id"]) == "${IID}"].iloc[0]["instance"]
pcc = json.loads(inst["parsed_commit_content"])

print(f"  files in gold commit: {len(pcc['file_diffs'])}")
os.makedirs("/tmp/r2e_gold_files", exist_ok=True)

# 写一个 apply 脚本:
# 1) git reset --hard (不用 git clean! 保留 run_tests.sh + r2e_tests)
# 2) 对每个 file_diff, 写 new_file_content 到对应路径
lines = ["#!/bin/bash", "cd /testbed", "git reset --hard HEAD 2>&1"]

for i, fd in enumerate(pcc["file_diffs"]):
    header = fd.get("header", {})
    # 文件路径从 header 或 minus_file/plus_file 拿
    fpath = ""
    if isinstance(header, dict):
        fpath = header.get("path", "") or header.get("file", {}).get("path", "")
    if not fpath and fd.get("plus_file"):
        pf = fd["plus_file"]
        fpath = pf.get("path", "").replace("b/", "", 1) if isinstance(pf, dict) else ""
    if not fpath:
        print(f"  [{i}] SKIP (no path)")
        continue

    new_content = fd.get("new_file_content", "")
    if not new_content:
        print(f"  [{i}] {fpath}: SKIP (no new_file_content)")
        continue

    # 写到本地临时文件
    local_file = f"/tmp/r2e_gold_files/file_{i}.py"
    with open(local_file, "w") as f:
        f.write(new_content)
    # 脚本里加一行: docker cp 进去 → mv 到正确路径
    print(f"  [{i}] {fpath}: {len(new_content)} chars")

    # 记录文件映射
    with open("/tmp/r2e_gold_files/manifest.txt", "a") as f:
        f.write(f"{local_file}\t{fpath}\n")

print("  gold files extracted")
PY

# 把 gold 文件 cp 进容器
echo "copying gold files into container..."
while IFS=$'\t' read -r LOCAL REMOTE; do
  docker cp "${LOCAL}" "${CONTAINER}:/tmp/gold_file_tmp"
  # 用 exec 把临时文件移到目标路径(创建必要目录)
  docker exec "${CONTAINER}" bash -c "mkdir -p /testbed/\$(dirname '${REMOTE}') && mv /tmp/gold_file_tmp /testbed/'${REMOTE}'"
done < /tmp/r2e_gold_files/manifest.txt
echo "gold files applied"

# 确保 run_tests.sh 还在(如果被 git reset 搞掉)
docker exec "${CONTAINER}" bash -c "
  if [ ! -f /testbed/run_tests.sh ] && [ -f /root/run_tests.sh.bak ]; then
    cp /root/run_tests.sh.bak /testbed/run_tests.sh
    echo 'restored run_tests.sh from backup'
  fi
  # 确保 r2e_tests symlink 还在
  if [ ! -e /testbed/r2e_tests ] && [ -d /r2e_tests ]; then
    ln -s /r2e_tests /testbed/r2e_tests
    echo 'restored r2e_tests symlink'
  fi
"

# 跑测试
echo "running tests with gold patch..."
cat > /tmp/r2e_run_B.sh <<'RUN'
#!/bin/bash
cd /testbed
bash /testbed/run_tests.sh > /tmp/eval_B.log 2>&1
echo "EXIT=$?" >> /tmp/eval_B.log
RUN
docker cp /tmp/r2e_run_B.sh "${CONTAINER}:/tmp/r2e_run_B.sh"
docker exec "${CONTAINER}" chmod +x /tmp/r2e_run_B.sh
docker exec "${CONTAINER}" bash /tmp/r2e_run_B.sh || true
sleep 2
docker cp "${CONTAINER}:/tmp/eval_B.log" /tmp/r2e_eval_B.log 2>/dev/null

if [[ -f /tmp/r2e_eval_B.log ]] && [[ -s /tmp/r2e_eval_B.log ]]; then
  echo "captured $(wc -l < /tmp/r2e_eval_B.log) lines"
  echo "--- last 30 lines ---"
  tail -30 /tmp/r2e_eval_B.log
  echo "--- end ---"
else
  echo "ERROR: no output captured"
fi

echo ""
echo "=== 5. Grade Scenario B ==="
python3 <<PY
import sys; sys.path.insert(0, "${AWS_KIRO_DIR}")
from server.patch_utils import parse_r2e_pytest_log, grade_eval_output_r2e
import pandas as pd

with open("/tmp/r2e_eval_B.log") as f:
    output = f.read()
df = pd.read_parquet("${PARQUET}")
inst = df[df["instance"].apply(lambda x: x["instance_id"]) == "${IID}"].iloc[0]["instance"]

parsed = parse_r2e_pytest_log(output)
result = grade_eval_output_r2e(inst, output)
print(f"  parsed tests : {len(parsed)}")
print(f"  resolved     : {result['resolved']}")
print(f"  reason       : {result['report'].get('reason')}")
if parsed:
    print(f"  first 3 tests: {dict(list(parsed.items())[:3])}")
if not result['resolved'] and result['report'].get('mismatches'):
    n = len(result['report']['mismatches'])
    print(f"  mismatches   : {n}")
    for m in result['report']['mismatches'][:5]:
        print(f"    {m}")
print(f"  SCENARIO B   : {'PASS (reward=1 ✓)' if result['resolved'] else 'FAIL (reward=0)'}")
PY

# ---- Summary ----
echo ""
echo "=============================================="
echo "  A=PASS + B=PASS → reward 链路完全正确 ✓"
echo "  A=PASS + B=FAIL → 可能是 gold patch 解析/apply 问题"
echo "=============================================="

echo ""
docker rm -f "${CONTAINER}" 2>/dev/null || true
docker rmi "${IMAGE}" 2>/dev/null | tail -1 || true
rm -rf /tmp/r2e_gold_files /tmp/r2e_*.sh /tmp/r2e_*.log /tmp/r2e_*.patch /tmp/r2e_check
echo "done"
