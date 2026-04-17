# SWE-RL Training Data Format

本目录给出 aws_kiro_infra trainer（`scripts/run_swe_rl_{4b,30b}_kiro_trainer.sh`）预期的训练数据格式，以及从 HuggingFace 数据集一键预处理的脚本。

**两种 eval strategy 同时支持**（由 parquet 行的 `data_source` 字段分派）：
- **SWE-Gym / SWE-Bench** → `swe_harness` 策略（FAIL_TO_PASS / PASS_TO_PASS + 预渲染 eval_script + 官方 harness）
- **R2E-Gym** → `r2e` 策略（image 内 `/run_tests.sh` + `expected_output_json` 字典比对）

CPU agent pod 在 `server/patch_utils.py::detect_eval_strategy` 按 `data_source` 自动选路；mirror rllm 的 `_calculate_reward_*` 5 分支。同一次训练 run 可以混用两种数据源。

Kiro team 的 `preprocessed_sweap_*_v8.x_train.parquet` 是 swe-gym-style（`swe_harness` 路径）。

---

## 1. Parquet schema（trainer 输入）

文件：一个 `.parquet`，每行一个 instance。**必需 4 列**：

| 列名 | 类型 | 作用 |
|---|---|---|
| `prompt` | `list[dict]` | 传给 LLM 的 message 列表，通常 `[{"role":"user", "content":"<problem_statement>"}]`。`generate_with_swe_remote._build_payload` 会提取最后一个 user 消息作为 `problem_statement`。 |
| `data_source` | `str` | ★ 决定 eval strategy 和 image 命名 scheme。必须含 `swe-gym` / `swe-bench` / `r2e-gym` 之一（见下方分派表）。 |
| `ability` | `str` | 约定写 `"coding"`，slime 框架要求但我们不用。 |
| `instance` | `dict` | 数据源相关的 instance 元数据，**两种 schema 见 §2-§3**。 |

slime trainer 的 flag 就是这么读的：

```
--input-key     prompt
--metadata-key  instance
```

（`data_source` 和 `ability` 被放在 Sample 的 metadata，`instance` 被放在 `metadata.instance`，这和 `sample.metadata.get("instance", metadata)` 兼容。）

### `data_source` → 分派

`server/patch_utils.py::detect_eval_strategy` 按 `data_source` 字符串选分支：

| `data_source` 包含的关键词 | eval strategy | image 命名规则 | 必需 instance 字段 |
|---|---|---|---|
| `swe-gym` | `swe_harness` | `xingyaoww/sweb.eval.x86_64.<iid_s_>:latest` | FAIL_TO_PASS / PASS_TO_PASS / eval_script / ... |
| `swe-bench`（含 verified） | `swe_harness` | `swebench/sweb.eval.x86_64.<iid_1776_>:latest` | 同上 |
| `r2e-gym` / `r2e` | `r2e` | **优先用 `instance.docker_image` 字段** | expected_output_json / docker_image / ... |
| 其它（heuristic 回退） | 看 instance 里有没有 `expected_output_json` 且没有 `FAIL_TO_PASS` | - | - |

---

## 2. `instance` dict schema — SWE-Gym / SWE-Bench（`swe_harness` 路径）

| 键 | 类型 | 必需 | 作用 |
|---|---|---|---|
| `instance_id` | `str` | ✅ | 唯一 ID（`django__django-12345`）。用于 Docker image 命名 + trace 归档。 |
| `repo` | `str` | ✅ | GitHub repo（`django/django`）。harness 用。 |
| `base_commit` | `str` | ✅ | git hash。容器起来后的 HEAD。 |
| `problem_statement` | `str` | ✅ | 任务描述（GitHub issue body）。 |
| `patch` | `str` | ✅ | Gold patch（unified diff）。用作 reference + 供 30B 脚本 `--label-key patch` 读。 |
| `test_patch` | `str` | ✅ | 补测试文件的 patch。harness eval 时应用。 |
| `FAIL_TO_PASS` | `list[str]` | ✅ | 打了 patch 后应从 fail 变 pass 的测试 ID。harness grading 关键输入。 |
| `PASS_TO_PASS` | `list[str]` | ✅ | 应始终 pass 的测试 ID。harness 用来检查 patch 没破坏其他测试。 |
| `version` | `str` | ✅ | SWE 数据集的 version 标识（有时和 `base_commit` 同，但 harness `make_test_spec` 需要）。 |
| `eval_script` | `str` | **✅** | **预渲染的 bash 测试脚本**（从 `swegym.harness.test_spec.make_test_spec(inst).eval_script` 取）。CPU pod 的 `swe_agent_server` 会直接拿来跑 eval，**不再原地调 `make_test_spec`**。省掉每个 rollout 几十秒。 |
| `hints_text` | `str` | 可选 | GitHub 上的讨论提示，一般不喂给 agent。 |
| `created_at` | `str` | 可选 | 时间戳，仅元数据。 |
| `environment_setup_commit` | `str` | 可选 | 某些 instance 的环境 setup 用的 commit。 |

### ⚠️ `eval_script` 务必**预渲染**后写进 parquet

- CPU agent pod 镜像可能不装 `swegym.harness`（只装 `swebench.harness`）。预渲染后跑 eval 就**只需要 bash + docker**。
- 如果你本地跑预处理时不预渲染，agent pod 侧 `resolve_eval_script` 会尝试 `make_test_spec(...)`，**要求 pod 镜像里有 `swegym`/`swebench` 的 harness**，否则会报错 `ModuleNotFoundError`。

---

## 3. `instance` dict schema — R2E-Gym（`r2e` 路径）

和 SWE-Gym **不同的**字段需求（R2E-Gym 有自己的 eval 机制）：

| 键 | 类型 | 必需 | 作用 |
|---|---|---|---|
| `instance_id` | `str` | ✅ | 唯一 ID（由 preprocess 从 `docker_image` 衍生，如 `<repo>_<hash>`）。用于 tarball 文件名。 |
| `docker_image` | `str` | **✅** | ★ **显式 image 名**（如 `docker.io/r2egym/<repo>_<hash>:latest`）。R2E 的 image 不能从 instance_id 推出来，必须从 HF 数据集 `docker_image` 列带过来。 |
| `expected_output_json` | `str` (JSON) | **✅** | ★ **gold 测试状态字典**（形如 `{"test_a": "PASSED", "test_b": "FAILED"}`）。eval 时用来和 pytest 实际输出逐项比对。 |
| `repo` | `str` | ✅ | 仓库名。log 解析时按 repo 挑 parser。 |
| `base_commit` | `str` | ✅ | 容器里预 checkout 的 commit。 |
| `problem_statement` | `str` | ✅ | 任务描述，构造 user message 用。 |
| `modified_files` / `relevant_files` | `list[str]` | 可选 | 参考信息，log/debug 用。 |
| `parsed_commit_content` | `str` | 可选 | 上游 gold diff（不在 runtime 使用）。 |

### R2E-Gym **不需要**这些字段

- ❌ `FAIL_TO_PASS` / `PASS_TO_PASS` —— R2E 不按这个判定
- ❌ `eval_script` —— image 里自带 `/run_tests.sh`
- ❌ `patch` / `test_patch` —— 不在 runtime 消费

（我们的 `preprocess_r2egym.py` 依然会把这些字段填空字符串 / 空 list，保持 dict 形状一致便于统一日志。）

---

## 4. 用我们的脚本一键产出（推荐）

### 依赖

```bash
pip install datasets pandas pyarrow

# SWE-Gym 预处理要装 make_test_spec
pip install git+https://github.com/SWE-Gym/SWE-Gym@main
# 或（处理 SWE-Bench 数据时）
pip install git+https://github.com/princeton-nlp/SWE-bench.git

# R2E-Gym 预处理不需要 r2egym 包 (不调 make_test_spec,只读 HF 字段)
```

### SWE-Gym / SWE-Bench → parquet

```bash
cd /data_storage/wyj/jxl/slime/examples/aws_kiro_infra

# 默认:SumanthRH/SWE-Gym-Subset (100 instances, 2 repos, smoke 用)
python3 data/preprocess_swegym.py \
  --output-path /data/swegym_for_kiro/train.parquet

# SWE-Gym 全量 (2438 instances, 11 repos)
python3 data/preprocess_swegym.py \
  --data-source SWE-Gym/SWE-Gym --split train \
  --output-path /data/swegym_full/train.parquet

# SWE-Bench Verified (500 instances, 当 eval 集)
python3 data/preprocess_swegym.py \
  --data-source SumanthRH/SWE-bench_Verified \
  --split test \
  --data-source-tag swe-bench-verified \
  --output-path /data/swebench_verified/eval.parquet
```

### R2E-Gym → parquet

```bash
# R2E-Gym Lite (230 instances)
python3 data/preprocess_r2egym.py \
  --output-path /data/r2egym_lite/train.parquet

# R2E-Gym V1 full (4578 train)
python3 data/preprocess_r2egym.py \
  --data-source R2E-Gym/R2E-Gym-V1 --split train \
  --output-path /data/r2egym_v1/train.parquet

# Smoke test
python3 data/preprocess_r2egym.py --max-samples 10 \
  --output-path /tmp/r2e_smoke.parquet
```

### 混合训练（同时用两种数据源）

两份 parquet concat 就行（4 列 schema 一致；只是 `data_source` 和 `instance` 的内部结构不同）：

```python
import pandas as pd
a = pd.read_parquet("/data/swegym_full/train.parquet")
b = pd.read_parquet("/data/r2egym_v1/train.parquet")
mixed = pd.concat([a, b], ignore_index=True).sample(frac=1, random_state=0)  # shuffle
mixed.to_parquet("/data/mixed/train.parquet", index=False)
```

Trainer 跑起来后，每条 sample 的 `data_source` 字段会被 server 自动识别，走对应 eval 路径。

### 校验产出

```bash
python3 -c "
import pandas as pd
df = pd.read_parquet('train.parquet')
print('rows:', len(df))
print('data_source 分布:')
print(df['data_source'].value_counts().to_string())
print('first instance keys:', list(df.iloc[0]['instance'].keys()))
"
```

---

## 5. 给 teammate 的规格（如果他们用自己的管道产出）

Teammate 只需保证产出的 parquet 满足 §1 + §2。具体：

```python
# Pseudocode
from swegym.harness.test_spec import make_test_spec   # or swebench.harness.test_spec.test_spec

rows = []
for example in your_raw_data:
    inst = {
        "instance_id":       example["instance_id"],
        "repo":              example["repo"],
        "base_commit":       example["base_commit"],
        "problem_statement": example["problem_statement"],
        "patch":             example["patch"],
        "test_patch":        example["test_patch"],
        "FAIL_TO_PASS":      example["FAIL_TO_PASS"],      # or json.loads(...) if string
        "PASS_TO_PASS":      example["PASS_TO_PASS"],
        "version":           example["version"],
        "hints_text":        example.get("hints_text", ""),
        "created_at":        example.get("created_at", ""),
    }
    # ★ 预渲染 eval_script
    inst["eval_script"] = make_test_spec(inst).eval_script

    rows.append({
        "prompt":      [{"role": "user", "content": example["problem_statement"]}],
        "data_source": "swe-gym",                   # or "swe-bench", "swe-bench-verified"
        "ability":     "coding",
        "instance":    inst,
    })

pd.DataFrame(rows).to_parquet("train.parquet")
```

### 常见陷阱

1. **`FAIL_TO_PASS` / `PASS_TO_PASS`** 在 HF 上可能是 JSON 字符串而不是 list。要 `json.loads(...)`。
2. **`eval_script` 不能漏**（见 §2 的警告）。
3. **`data_source` 字符串要对**。`server/mini_swe_agent.py::get_docker_image_name` 里检查：
   - 含 `"swe-gym"`  → `docker.io/xingyaoww/sweb.eval.x86_64.<iid_with_s_replace>:latest`
   - 含 `"swe-bench"` → `docker.io/swebench/sweb.eval.x86_64.<iid_with_1776_replace>:latest`
   其它字符串会抛 `NotImplementedError`。
4. **instance_id 大小写敏感**：我们的代码会内部 `.lower()` 然后传给 harness（对齐 swebench/swegym 约定）。保留原样写进 parquet 即可。

---

## 6. 数据集规模参考

### SWE-Gym 系（`swe_harness` 路径）

| 数据集 | train instances | 覆盖 repo | 用途 |
|---|---|---|---|
| `SumanthRH/SWE-Gym-Subset` | 100 | 2（moto, mypy） | smoke test / dev |
| `SWE-Gym/SWE-Gym` | **2438** | 11 | 完整 SWE-Gym（学术论文基线） |
| `SWE-Gym/SWE-Gym-Lite` | 230 | 11 | 轻量版 |
| `SumanthRH/SWE-bench_Verified` | 500 | 12 | **评测集**（不能当训练） |
| Kiro 的 `preprocessed_sweap_582_v8.0` | 582 | -- | 内部，4B 默认 |
| Kiro 的 `preprocessed_sweap_filtered_301_v8.1` | 301 | -- | 内部，30B 默认 |

### R2E-Gym 系（`r2e` 路径）

| 数据集 | train instances | 用途 |
|---|---|---|
| `R2E-Gym/R2E-Gym-Subset` | **4578** | **★ DeepSWE 官方训练集**（过滤掉 SWE-Bench-Verified 同 repo），默认 |
| `R2E-Gym/R2E-Gym-Lite` | 4578 | 和 Subset 同量（镜像/别名） |
| `R2E-Gym/R2E-Gym-V1` | 8101 | 完整未过滤版（有 data leak 风险，**不建议训练**） |
| `R2E-Gym/SWE-Bench-Verified` | 500 | R2E 包装的 SWE-Bench Verified（仅作评测） |

### Kiro sweap vs SWE-Gym

Kiro 的 "sweap" 是在原始 SWE-Gym 之上做了**质量过滤**（剔除 flaky / env-setup 易坏的 instance）。要完全对齐，建议跟 Kiro team 要他们的 filter script；否则我们的 `preprocess_swegym.py` 用原始数据是安全 fallback。

---

## 7. 部署到 AWS

把预处理好的 parquet 放到 trainer YAML 指定的路径：

```yaml
# k8s/launch_trainer_4b.yaml
env:
- name: PROMPT_DATA
  value: "/mnt_out/jinxiaolong/data/preprocessed_sweap_582_v8.0_train.parquet"
```

即：
```bash
# 本地 → PVC
scp /data_storage/wyj/jxl/OpenClaw-RL/data/swegym_for_kiro/train.parquet \
    <hyperpod-bastion>:/mnt_out/jinxiaolong/data/preprocessed_sweap_582_v8.0_train.parquet
# 或放自己的路径并改 YAML 里的 PROMPT_DATA 值
```

如果你先上 Kiro team 的 sweap parquet，trainer YAML 里的默认路径就能直接跑。
