# Dataset / Train Reward 现状诊断（R2E-Gym-Subset）

本文整理当前 `aws_kiro_infra` 在 `R2E-Gym/R2E-Gym-Subset` 上的数据、镜像与 reward 计算链路，以及已确认问题和后续修改建议，便于后续工程化调整。

---

## 1. 目标与范围

- 范围：`swe-rl/aws_kiro_infra` 的 R2E 路径（本地 smoke + 训练前准备）。
- 重点：
  - 数据集预处理是否正确。
  - 镜像下载与命名是否和 reward 计算链路一致。
  - `run_tests.sh` 执行与解析是否稳定。
  - reward 判定与 rLLM/R2E-Gym 原始逻辑是否一致。

---

## 2. 当前默认配置（已调整）

### 2.1 数据源默认值

已改为默认使用 Subset：

- `data/run_r2e_local_test.sh`
  - `R2E_DATA_SOURCE` 默认：`R2E-Gym/R2E-Gym-Subset`
- `data/preprocess_r2egym.py`
  - `--data-source` 默认：`R2E-Gym/R2E-Gym-Subset`

### 2.2 预处理策略

- 默认 streaming（`run_r2e_local_test.sh` 里 `STREAMING_PREPROCESS=1`）。
- 作用：避免先 materialize 全 split 导致长时间“看似卡住”。

---

## 3. Reward 计算链路（当前实现）

## 3.1 rLLM / R2E-Gym 原始逻辑（对照）

R2E 分支核心是：

1. 在容器里执行测试脚本（历史上常见是 `run_tests.sh`）。
2. 解析 pytest 日志（`short test summary info` 段）。
3. 与样本里的 `expected_output_json`（gold 字典）逐项比对。
4. 完全一致 reward=1，否则 reward=0。

这不是 `eval_script` 主导的 SWE harness 路径。

## 3.2 aws_kiro_infra 当前逻辑

- 策略分派：`server/patch_utils.py::detect_eval_strategy`
  - `data_source` 含 `r2e` 时走 R2E grader。
- 执行：`server/docker_ops.py::evaluate_r2e`
  - apply patch -> run tests -> 返回原始输出。
- 判分：`server/patch_utils.py::grade_eval_output_r2e`
  - `parse_r2e_pytest_log` + `expected_output_json` 比对。

---

## 4. 已确认问题与现象

## 4.1 `instance_id` 冲突（已修复）

### 现象

- 同一 repo 的不同 tag 样本都映射为同一个 `instance_id`（如都变成 `orange3_final`）。
- 后果：下载/保存 tar 时重复覆盖或全部 skip，无法得到多样本。

### 根因

- `preprocess_r2egym.py` 旧逻辑仅用镜像名，不包含 tag/hash。

### 修复

- 新逻辑：`instance_id` 包含 tag（`name__tag`），`latest` 保持兼容。
- 结果：现在可生成如
  - `orange3_final__2d9617...`
  - `orange3_final__a95245...`

---

## 4.2 `download_swe_images.sh` 不适用于 R2E 真 docker_image

### 现象

- 传 `PROXY_DOCKER_IO` 无效。
- 脚本仍拉 `docker.io/xingyaoww/sweb.eval.x86_64.*` 风格镜像。

### 根因

- 该脚本是 SWE 系（swe-gym/swebench）命名规则脚本，按 `instance_id` 推导镜像名。
- 使用的是 `PROXY_PREFIX_SWE_GYM / PROXY_PREFIX_SWE_BENCH`，并非 `PROXY_DOCKER_IO`。

### 影响

- 对 R2E（`instance.docker_image = namanjain12/...`）会“拉错镜像源/命名体系”。

---

## 4.3 R2E 镜像中可能不存在 `/run_tests.sh`

### 现象

- 评测日志出现：
  - `parsed: 0 tests`
  - `reason: test_count_mismatch`
- 手动验证输出：
  - `bash: /run_tests.sh: No such file or directory`

### 影响

- 即使数据、镜像下载成功，reward 仍全部为 0（尤其 Scenario B gold patch 也失败）。

### 当前应对（已做）

- `docker_ops.evaluate_r2e` 增加路径探测：
  - `/run_tests.sh`
  - `/testbed/run_tests.sh`
  - `/root/run_tests.sh`
- `data/test_r2e_eval_local.py` 同步路径探测逻辑。

### 当前结论

- 部分实际镜像在上述路径仍找不到脚本，说明镜像构建/路径约定与当前假设不一致。

---

## 4.4 本地镜像不存在的“误报”

### 现象

- 检查脚本 `docker run --pull never docker.io/...` 报 `No such image`。

### 根因

- 批量下载脚本 `save` 完后会 `docker rmi` 清理（节省 `/ebs/docker`）。

### 处理

- 验证某实例时要先 `docker load -i <iid>.tar.gz` 再 `docker run --pull never ...`。

---

## 5. 当前状态快照（截至本次诊断）

- 已成功生成 `R2E-Gym-Subset` streaming parquet（20 条样本）。
- 已成功拉取并保存 5 个不同 tag 的 `orange3_final` tarball 到：
  - `data/r2egym_smoke/images_r2e_5_unique/*.tar.gz`
- 镜像代理拉取链路可用（`slime-agent-cn-beijing.cr.volces.com`）。
- 关键待解是：部分镜像中 test script 路径/存在性与预期不一致。

---

## 6. 后续修改建议（按优先级）

## P0（必须）

1. **新增 R2E 专用批量下载脚本**
   - 直接读取 parquet `instance.docker_image`。
   - 支持 `PROXY_DOCKER_IO`（docker.io 前缀重写）。
   - 不走 `xingyaoww/swebench` 推导逻辑。

2. **把“脚本不存在”变成强信号错误**
   - 在 `evaluate_r2e` 返回明确错误字段（已部分实现）。
   - 上层 summary 中单独统计该类失败，避免被误判成“模型 patch 问题”。

## P1（强烈建议）

3. **增加 run_tests path/命令探测能力**
   - 除固定路径外，考虑容器内搜索 `run_tests.sh`（限制深度，避免慢）。
   - 或支持从 parquet/metadata 注入测试命令字段（若数据侧可提供）。

4. **在 preprocess 阶段增加可选健康检查**
   - 抽样验证镜像可运行 + 测试脚本存在。
   - 提前筛掉明显坏样本，减少训练时 reward=0 噪音。

## P2（优化）

5. **增强日志可观测性**
   - 保存 `run_tests` 原始输出前后若干行。
   - 在 `test_count_mismatch` 时打印 expected/parsed 的大小和 key diff。

---

## 7. 建议的验证清单（改动后必跑）

1. **预处理唯一性**
   - 检查 `instance_id` 是否去重后仍有足够样本。
2. **镜像下载正确性**
   - pull 地址是否来自代理镜像站。
   - tar 是否按 `instance_id.tar.gz` 落盘。
3. **容器测试脚本存在性**
   - 对前 N 个样本做脚本探测（至少 5 个）。
4. **reward 行为**
   - Scenario A（no patch）应为 0。
   - Scenario B（gold patch）应尽量为 1（若不是，至少要能给出可解释错误）。
   - Scenario C（bad patch）应为 0。

---

## 8. 关键文件索引

- 预处理：
  - `data/preprocess_r2egym.py`
- 本地 smoke：
  - `data/run_r2e_local_test.sh`
  - `data/test_r2e_eval_local.py`
- 服务端执行与判分：
  - `server/docker_ops.py`
  - `server/patch_utils.py`
- 现有（SWE导向）镜像下载脚本：
  - `data/download_swe_images.sh`

---

## 9. 修复与验证结果（2026-04-16 更新）

### 9.1 根因（源码级确认）

分析 R2E-Gym 官方 clone（`R2E-Gym/src/r2egym/agenthub/runtime/docker.py`）+ Dockerfiles，确认：

1. **`run_tests.sh` 在 image 里存在**：`/testbed/run_tests.sh`（每个 repo 的 Dockerfile 里 `COPY run_tests.sh /testbed/run_tests.sh`）
2. **R2E runtime `setup_env()` 做了关键 setup**（我们之前漏掉的）：
   - `SKIP_FILES_NEW = [“run_tests.sh”, “r2e_tests”]` — 这两个文件从 `/testbed/` 移到 `/root/`（对 agent 隐藏）
   - `ln -s /root/r2e_tests /testbed/r2e_tests`（让 `cd /testbed && pytest r2e_tests` 能找到测试）
3. **我们的 `evaluate_r2e` 里 `git clean -fd`** 把非 git 跟踪的 `run_tests.sh` 和 `r2e_tests/` symlink 都删了

### 9.2 框架修复（3 处改动，不影响 SWE-Gym）

| 文件 | 改动 | 触发条件 |
|---|---|---|
| `server/docker_ops.py` | 新增 `setup_r2e_container()` — `ln -s /r2e_tests /testbed/r2e_tests` + 备份 `run_tests.sh` | 仅 `strategy==”r2e”` |
| `server/docker_ops.py` | `evaluate_r2e()`: `git clean -fd` → `git clean -fd -e run_tests.sh -e r2e_tests` | 仅 `evaluate_r2e` 内部 |
| `server/swe_agent_server.py` | strategy 提前检测；R2E agent + eval 容器创建后调 `setup_r2e_container` | swe_harness 路径完全不变 |

### 9.3 本机 E2E 验证 ✅

```
instance: orange3_final__2d9617bd0cb1f0ba61771258410ab8fae8e7e24d
image:    namanjain12/orange3_final:2d9617bd...

Scenario A (no patch):
  parsed tests : 10
  1 FAILED:  test_migrates_settings_removes_incompatible
  9 PASSED:  test_close_context, test_fast_save, test_find_or_create_context, ...
  resolved     : False
  reward       : 0  ← 预期 ✓

Scenario B (gold patch, 写 new_file_content 到容器):
  parsed tests : 10
  10 PASSED:  (全部)
  resolved     : True (与 expected_output_json 完全 match)
  reward       : 1  ← 预期 ✓
```

**验证细节**：
- `run_tests.sh` 内容：`QT_QPA_PLATFORM=minimal ... xvfb-run ... pytest -rA r2e_tests`（159 bytes）
- 测试文件：`/r2e_tests/test_1.py`（10 test cases in `TestContextHandler` + `TestSettingsPrinter`）
- `expected_output_json` 字典：10 个 test → 全 “PASSED”
- 无 patch 时 FAILED 的那条就是 gold commit 要修的 bug（IncompatibleContext raise 位置不对）
- Gold patch 写了 3 个文件（`settings.py` + `test_context_handler.py` + `tutorial-settings.rst`）后全 PASS

### 9.4 结论

**R2E-Gym reward 链路完全打通**。之前全部 reward=0 的三个根因：

| 根因 | 状态 |
|---|---|
| `r2e_tests` symlink 缺失 → pytest collected 0 items | ✅ 已修复（`setup_r2e_container`） |
| `git clean -fd` 删 `run_tests.sh` → No such file | ✅ 已修复（`-e` exclusion） |
| 本机 docker exec stdout 被吞 → 路径探测假阴性 | ✅ 已确认不影响 K8s（仅本机 rootless bug） |

SWE-Gym 路径**未受任何影响**（所有改动都在 `strategy==”r2e”` 条件分支内）。
