# SWE-RL AWS Kiro Infra — 运维手册

基于 **Kiro 生产 3-pool 拓扑**（对齐 `slime/examples/kiro_agent/` latest async disaggregated 部署），唯一差异：**agent scaffold 使用 mini-swe-agent**（bash backtick + submit sentinel），不用 Kiro 原生的 Strands + Qwen tool_call。

---

## 1. 概览

### 这是什么

一套在 AWS HyperPod / EKS 上跑 SWE-Bench RL 的全链路。包含：
- **算法**：GRPO（4B 用 DAPO 非对称 clip + aux LLM judge；30B 用 Kiro 对称 clip + TIS + async + per-token-loss）
- **模型**：Qwen3-4B-Instruct-2507 / Qwen3-Coder-30B-A3B-Instruct
- **Scaffold**：mini-swe-agent（LLM 输出 \`\`\`bash ... \`\`\` → 容器 exec → 下一 turn）
- **环境**：SWE-Bench 实例 docker 容器
- **评测**：官方 swebench/swegym harness

### 为什么 3 个 PyTorchJob

Kiro 生产发现单 job colocate 存在问题：
- Trainer OOM 会拖垮 SGLang engine（重启 4B/30B 模型要几分钟）
- Docker 密集的 agent workload 和 NCCL/EFA 密集的 trainer workload 在同节点会互相干扰
- GPU 节点跑 Docker 浪费钱（`ml.p5.48xlarge` 租金 ~ 15× `ml.m5.24xlarge`）

所以拆成 3 个独立 pool：**GPU SGLang / CPU Docker+agent / GPU trainer**。这是 Kiro team 的最终设计。

---

## 2. 架构

### 拓扑

```
┌───────────────────────────────────────────────────────────────────────────┐
│  Job 1 — SGLang 池（GPU）       k8s/launch_sglang_{4b,30b}.yaml           │
│                                                                           │
│   每 pod (ml.p5.48xlarge, 8 × H100):                                      │
│     - 1 × SGLang engine (Ray actor, TP=8)                                 │
│   Head pod 额外:                                                          │
│     - SGLang router :30000  (KV-cache aware LB)                           │
│   产出到共享 PVC:                                                         │
│     <OUT>/<SGLANG_RUN_NAME>/sglang_external_rollout.env                   │
└──────────────────────────────────┬────────────────────────────────────────┘
                                   │ 共享 PVC (.env 文件协调)
┌──────────────────────────────────┴────────────────────────────────────────┐
│  Job 2 — swe_agent 池（CPU）    k8s/launch_swe_agent_workers.yaml         │
│                                                                           │
│   每 pod (ml.m5.24xlarge, 96 vCPU / 384 GB, privileged):                  │
│     - dockerd           (跑 SWE-Bench 实例容器)                           │
│     - FastAPI :5000     (server.swe_agent_server:app)                     │
│         /healthz                                                          │
│         /generate_trajectory — 每 trajectory 一次调用,返回完整结果         │
│                                                                           │
│   In-process 跑 mini-swe-agent loop:                                       │
│     本地 tokenizer (MODEL_PATH) → apply_chat_template → input_ids         │
│     ↓                                                                      │
│     SGLang router /generate (收 output_token_ids + logprobs)               │
│     ↓                                                                      │
│     docker exec (本地 dockerd, SWE 实例容器)                               │
│     ↓                                                                      │
│     submit sentinel / max_steps → 退出循环                                 │
│     ↓                                                                      │
│     policy gate → eval harness (fresh 容器 apply patch + 跑测试)           │
│                                                                           │
│   产出到共享 PVC:                                                         │
│     <OUT>/<AGENT_RUN_NAME>/swe_agents.env   (每 pod flock-append 自己 URL) │
└──────────────────────────────────┬────────────────────────────────────────┘
                                   │ 共享 PVC (.env 文件协调)
┌──────────────────────────────────┴────────────────────────────────────────┐
│  Job 3 — trainer 池（GPU）     k8s/launch_trainer_{4b,30b}.yaml           │
│                                                                           │
│   等两个 .env 文件落到 PVC → 起自己的 Ray cluster                          │
│   跑 train.py (4B sync) 或 train_async.py (30B async),--rollout-external   │
│                                                                           │
│   generate_kiro.generate 每个 trajectory:                                  │
│     1. least-pending 选一个 swe_agent URL                                 │
│     2. POST /generate_trajectory (problem / instance / sampling)           │
│     3. 收到 per-turn token_ids / loss_mask / logprobs / patch / eval       │
│     4. 拼 Sample (swe-rl trajectory 模式)                                  │
│     5. (4B only) 调 swe_traj_judge → litellm → aux reward                 │
└───────────────────────────────────────────────────────────────────────────┘
```

### 单个 trajectory 端到端数据流

```
Trainer pod                CPU agent pod            GPU SGLang pod          Docker 实例容器
-----------                 -------------            ---------------         -----------------
generate_kiro.generate
     │
     │  POST /generate_trajectory
     ├──────────────────────▶
     │                       allocate SWE 容器
     │                             │
     │                             ├─ docker run ─────────────────────────▶ 启动 sleep inf
     │                             │                                        
     │  mini-swe-agent 主循环:
     │                       build messages
     │                       apply_chat_template → input_ids
     │                             │
     │                             │  POST /generate {input_ids}
     │                             ├───────────────▶
     │                             │                inference
     │                             │ ◀──{text, output_token_ids, logprobs}
     │                             │
     │                       parse ```bash...```
     │                             │
     │                             │  docker exec bash -lc ─────────────▶ 执行命令
     │                             │ ◀──{returncode, output}─────────────── 返回结果
     │                             │
     │                       render observation → messages.append(user)
     │                       loop ↑ 直到 submit/max_steps
     │
     │                       policy gate (touch tests? config?)
     │                       eval harness:
     │                             ├─ docker run (fresh) ────────────────▶ 启动
     │                             ├─ git apply + bash eval_script ──────▶ 跑测试
     │                             ├─ docker rm ─────────────────────────▶ 销毁
     │                       get_eval_report (swebench/swegym)
     │  ◀─────────────── per-turn data + patch + eval_result
     │
     _build_trajectory_sample
         prompt_tokens = system+user msgs 的 token_ids
         response_tokens = 后续所有 msgs 的 token_ids
         loss_mask = 每 msg 的 token_mask (assistant=1, env=0)
     │
     aux_judge (4B only, 单次 litellm) → aux_reward
     │
     sample.reward = {score, acc}
     return Sample
```

### Data 字段 → 容器操作映射

**核心问题**：parquet 里存的是 `instance_id` / `eval_script` / `FAIL_TO_PASS` ... 这些字段如何变成 CPU pod 上的一串 `docker run / docker exec / docker rm` 命令？

#### Parquet 字段消费表

`server/patch_utils.py::detect_eval_strategy` 按 `data_source` 字符串分派成两条消费路径：

**路径 A — `swe_harness`**（SWE-Gym / SWE-Bench）

| Parquet 字段 | 被哪个模块用 | 用来干什么 |
|---|---|---|
| `data_source` (`swe-gym` / `swe-bench`) | `detect_eval_strategy` + `get_docker_image_name` | 选命名 scheme：`swe-gym`→`xingyaoww/`，`swe-bench`→`swebench/` |
| `instance.instance_id` | `get_docker_image_name` | image 名核心（`getmoto__moto-7365` → `getmoto_s_moto-7365`） |
| `instance.problem_statement` | `run_agent_loop` | 第一条 user message |
| `instance.FAIL_TO_PASS` | `analyze_patch_policy` + `grade_eval_output` | policy gate scope + 最终 resolved 判定 |
| `instance.PASS_TO_PASS` | `grade_eval_output` → swebench harness | 检查 patch 没破坏其他测试 |
| `instance.eval_script` | `docker.evaluate` | ★ 直接 `bash <<EOF ... EOF` 扔进 fresh eval 容器 |
| `instance.patch`（gold） | **不消费** | reference only |
| `instance.test_patch` | 已包含在 `eval_script` 里 | 不单独用 |
| `instance.base_commit` | **镜像构建时已处理** | 运行时不用 |

**路径 B — `r2e`**（R2E-Gym）

| Parquet 字段 | 被哪个模块用 | 用来干什么 |
|---|---|---|
| `data_source` (`r2e-gym`) | `detect_eval_strategy` | 选 r2e 路径 |
| `instance.docker_image` | `get_docker_image_name` | ★ **显式 image 名**（R2E 的 image 不能从 iid 推出） |
| `instance.instance_id` | 仅 tarball 文件名 / trace 归档 | 不用于 image 命名 |
| `instance.problem_statement` | `run_agent_loop` | 第一条 user message |
| `instance.expected_output_json` | `grade_eval_output_r2e` | ★ **gold 测试状态字典**，逐项比对 pytest 输出 |
| `instance.FAIL_TO_PASS` / `PASS_TO_PASS` / `eval_script` | **不消费** | R2E 不需要（image 自带 `/run_tests.sh`） |

**通用（两条路径都用）**

| 字段 | 用途 |
|---|---|
| `prompt` (message list) | `_build_payload` 回退提取 `problem_statement` |
| `ability` | 不消费，slime 框架约定 |

#### 一个具体 instance 的完整 trace

拿 `getmoto__moto-7365`（SWE-Gym）举例，数据 → 动作的**精确映射**：

```
Parquet 行:
┌───────────────────────────────────────────────────────────────────────────┐
│ data_source   = "swe-gym"                                                 │
│ instance_id   = "getmoto__moto-7365"                                      │
│ problem_stmt  = "DynamoDB's `update_item` performs ..."                   │
│ base_commit   = "7f6c9cb1..."                                             │
│ FAIL_TO_PASS  = ["tests/test_dynamodb/...::test_update_item_add_float"]   │
│ PASS_TO_PASS  = [...]                                                     │
│ eval_script   = "#!/bin/bash\nset -xo pipefail\nsource /opt/.../          │
│                  activate testbed\ncd /testbed\ngit config --global ..."  │
│ patch         = "diff --git a/moto/dynamodb/models/dynamo_type.py ..."    │
│ ...                                                                       │
└───────────────────────────────────────────────────────────────────────────┘
                            │
                            │ HTTP POST /generate_trajectory (整行送)
                            ▼
                    swe_agent_server.generate_trajectory(req)

① 解析 image name ────────────────────────────────────────────────────────
   get_docker_image_name(instance, data_source)
     "swe-gym" 命中 → iid="getmoto__moto-7365".replace("__","_s_")
                    = "getmoto_s_moto-7365"
     →  docker.io/xingyaoww/sweb.eval.x86_64.getmoto_s_moto-7365:latest

② 如果 tarball 在 PVC 上,docker load ────────────────────────────────────
   if exists(${SWE_DOCKER_IMAGES_PATH}/getmoto__moto-7365.tar.gz):
       docker load -i /mnt_out/.../swe_images/getmoto__moto-7365.tar.gz

③ docker run 起 agent 容器 ───────────────────────────────────────────────
   DockerManager.create(image):
     docker run -d --init --name swe-<uuid> --pull never \
                -w /testbed --pids-limit 1024 --memory 8g \
                docker.io/xingyaoww/sweb.eval.x86_64.getmoto_s_moto-7365:latest \
                sleep infinity
     → container_id = <abc123...>
     容器里 /testbed 已经是 HEAD=7f6c9cb1... (SWE-Bench image 预 checkout 好的)

④ 主 agent 循环 ──────────────────────────────────────────────────────────
   messages = [
     {role: system, content: <system_template from swebench.yaml>},
     {role: user,   content: <Template(instance_template).render(
                                 task=problem_statement)>},
   ]
   for turn in 0..step_limit-1:
     input_ids = tokenizer.apply_chat_template(messages) + gen_prompt
     text, token_ids = SGLang.generate(input_ids=...)
     messages.append({role:assistant, ..., token_ids:...})

     bash_cmd = parse_bash_action(text)   # 从 ```bash...``` 抠

     if "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in bash_cmd:
         # submit 路径: 执行一次拿 patch 然后退出
         exec_result = docker.exec(container_id, bash_cmd, cwd="/testbed")
         git_patch = extract_patch_from_submission(exec_result.output)
         break
     else:
         # 普通 exec: 执行 + 把结果塞回下一轮 user message
         exec_result = docker.exec(container_id, bash_cmd, cwd="/testbed")
         messages.append({role:user, content: render_observation(exec_result)})

   # docker exec 具体命令:
   #   docker exec -w /testbed <abc123> bash -lc "<bash_cmd>"

⑤ Git diff 兜底(如果 LLM 没主动 submit) ─────────────────────────────────
   if git_patch is None:
     git_patch = docker.diff(container_id, cwd="/testbed")
     # docker exec <abc123> bash -lc "git add -A && git diff --cached"

⑥ Policy gate ──────────────────────────────────────────────────────────
   analyze_patch_policy(git_patch, instance):
     changed_files = re.findall(r"^diff --git a/(.+?) b/", git_patch)
     # 从 instance.FAIL_TO_PASS 抽 test 文件名
     if any(f 属于 FAIL_TO_PASS 对应的 test 文件 for f in changed_files):
         violated = True  → 跳过 eval, reward=0

⑦ 销毁 agent 容器 + 起 fresh eval 容器 ───────────────────────────────────
   docker.destroy(agent_container_id)
     # docker rm -f <abc123>

   eval_container = docker.create(image=<同一个 image>)
     # docker run -d ... → <def456>

⑧ 应用 patch + 跑 eval_script (★ parquet.eval_script 在这里终于被用) ──
   docker.evaluate(eval_container_id, patch=git_patch, eval_script=<parquet.instance.eval_script>):

     # 步骤 (8a) git reset 回 HEAD
     docker exec -w /testbed <def456> bash -lc \
       "git reset --hard HEAD && git clean -fd && \
        git apply <<'PATCH_<uuid>'
        <git_patch 原文>
        PATCH_<uuid>"

     # 步骤 (8b) 跑 parquet 里的 eval_script
     docker exec -w /testbed <def456> bash -lc \
       "bash <<'EVAL_<uuid>'
        #!/bin/bash
        set -xo pipefail
        source /opt/miniconda3/bin/activate
        conda activate testbed
        cd /testbed
        ...(parquet.instance.eval_script 的几 KB 原文)...
        python -m pytest tests/test_dynamodb/...::test_update_item_add_float
        EVAL_<uuid>"

     → {returncode, stdout, apply_output}

⑨ 官方 harness grading ──────────────────────────────────────────────────
   grade_eval_output(instance, git_patch, apply_output, eval_output):
     test_spec = make_test_spec(instance)     # 仍用 instance 里字段
     report = get_eval_report(test_spec,
                              prediction={patch, instance_id},
                              log_path=<tmp 文件 写 stdout>,
                              include_tests_status=True)
     # 根据 FAIL_TO_PASS/PASS_TO_PASS 对比 stdout 里的 PASSED/FAILED 行
     resolved = report["resolved"]   # True/False

⑩ 销毁 eval 容器 + HTTP 返回 ────────────────────────────────────────────
   docker.destroy(eval_container_id)

   return GenerateResponse(
     reward = 1 if resolved else 0,
     eval_result = {resolved, grading_report, ...},
     policy = {...},
     messages = [...],          # trainer 拼 Sample 用
     step_debug = [...],
     git_patch = "...",
   )
```

#### 为什么 `eval_script` 是 parquet 预渲染而不是 CPU pod 现场生成

CPU pod 启动时 `swe_agent_server.generate_trajectory` 会调 `patch_utils.resolve_eval_script`：

```python
def resolve_eval_script(instance):
    # 优先读 parquet 里存的预渲染 eval_script
    if instance.get("eval_script", "").strip():
        return instance["eval_script"]            # ← 99% 走这条,快速
    # 兜底:临时调 make_test_spec(instance).eval_script
    # 这里会 import swebench / swegym.harness,
    # pod 镜像没装就 ModuleNotFoundError
    ...
```

所以**预处理时把 `eval_script` 填好**是最稳的路径——CPU pod 镜像里不用装 `swegym`，只要有 `docker` CLI + `python3` + `fastapi` + `transformers` 就能跑。

#### 镜像从哪儿来（再强调一次）

CPU pod 刚启动时，**dockerd 缓存是空的**。`docker run --pull never` 只认本地已加载的 image。两条路：

| 方案 | 操作 | 时机 |
|---|---|---|
| **A. Tarball load**（我们默认） | `SWE_DOCKER_IMAGES_PATH/<iid>.tar.gz` 存在 → `docker load -i` | 首次遇到某 instance 时加载，之后 dockerd 缓存命中 |
| **B. Registry pull**（非默认） | 改 `docker_ops.py` `--pull never` → `--pull missing` | 每个 pod 启动 + 每个新 instance 从 registry 拉（慢） |

所以**准备 tarballs 是上 K8s 前的必做项**。数据准备顺序：
1. 先跑 `preprocess_swegym.py` 产出 parquet（含 `instance_id` 列表）
2. 用 `download_swe_images.sh` 读 parquet → 下载对应 image → 存 tarball
3. Tarball 和 parquet 一起 rsync 到 PVC

两份是**一套**：parquet 里哪些 instance，tarball 目录就得有哪些 `<iid>.tar.gz`。

---

## 3. 代码结构

共 **21 个文件，~1850 LOC Python / ~1200 LOC shell+YAML**。

### 目录

```
swe-rl/aws_kiro_infra/
├── __init__.py                          包文档（3-pool 描述）
├── RUNBOOK.md                           本文件
├── generate_kiro.py                     slime 入口（薄壳）
├── generate_with_swe_remote.py          Trainer 侧 HTTP 客户端 + Sample 拼接
├── swe_traj_judge.py                    Aux LLM judge（4B trainer 用,独立）
├── swebench.yaml                        mini-swe-agent scaffold 配置
├── server/                              ★ CPU agent pod 代码
│   ├── __init__.py
│   ├── swe_agent_server.py              FastAPI 主入口
│   ├── mini_swe_agent.py                Scaffold + 主 agent 循环
│   ├── patch_utils.py                   Patch 解析 + policy gate + eval harness
│   ├── docker_ops.py                    Docker CLI subprocess wrapper
│   └── sglang_client.py                 SGLang /generate async 客户端
├── data/                                ★ 数据 / 镜像准备脚本(预处理机器上跑)
│   ├── README.md                        Parquet schema 规范(含 SWE-Gym 和 R2E-Gym 两份 schema)
│   ├── preprocess_swegym.py             SWE-Gym / SWE-Bench HF 数据 → parquet(swe_harness 策略)
│   ├── preprocess_r2egym.py             R2E-Gym HF 数据 → parquet(r2e 策略)
│   └── download_swe_images.sh           Parquet 列表 → docker image tarballs(SWE-Gym & R2E-Gym 通用)
├── k8s/                                 5 个 PyTorchJob YAML
│   ├── launch_sglang_4b.yaml
│   ├── launch_sglang_30b.yaml
│   ├── launch_swe_agent_workers.yaml
│   ├── launch_trainer_4b.yaml
│   └── launch_trainer_30b.yaml
└── scripts/                             4 个 bash launcher
    ├── launch_sglang.sh                 GPU 节点用（SGLang engine + router）
    ├── launch_swe_agent_workers_cpu.sh  CPU 节点用（dockerd + FastAPI）
    ├── run_swe_rl_4b_kiro_trainer.sh    Trainer 节点用（4B）
    └── run_swe_rl_30b_kiro_trainer.sh   Trainer 节点用（30B）
```

### 各文件职责

| 文件 | LOC | 跑在哪 | 作用 |
|---|---|---|---|
| `generate_kiro.py` | 82 | Trainer | slime 注册的入口。调 `generate_with_swe_remote.generate_trajectory`；reward_func 组合 outcome + `aux_reward_coef × judge_score` |
| `generate_with_swe_remote.py` | 453 | Trainer | 选 server URL（least-pending）→ POST `/generate_trajectory` → 收 per-turn token → 拼 Sample → 调 aux judge → 保存 artifacts |
| `swe_traj_judge.py` | 354 | Trainer | litellm 调 GPT-4o-mini 对整条 trajectory 打分（`[-1, +1]`），返回 `{score, model, votes, status}` |
| `swebench.yaml` | — | 两边都用 | system prompt / instance template / observation 模板 / submit sentinel 字符串 |
| `server/swe_agent_server.py` | 347 | CPU agent | FastAPI app: `/healthz` + `/generate_trajectory`。启动时加载 tokenizer + 启 dockerd + 加载 swebench.yaml |
| `server/mini_swe_agent.py` | 371 | CPU agent | Scaffold 核心：chat_template token 计算、agent 主循环、observation 渲染 |
| `server/patch_utils.py` | 460 | CPU agent | Bash 解析、submit sentinel、patch 提取、policy gate、**双策略 eval**：`swe_harness`(swebench/swegym `get_eval_report`) + `r2e`(`parse_r2e_pytest_log` + `grade_eval_output_r2e`,~100 行内联了 r2egym 的 reward 逻辑) + 分派器 `detect_eval_strategy` |
| `server/docker_ops.py` | 320 | CPU agent | 基于 subprocess 的 `docker run`/`exec`/`diff`/`rm`，async wrapper + DockerManager。两个 evaluate 方法：`evaluate`（SWE-Gym,用 parquet.eval_script heredoc）+ `evaluate_r2e`（R2E-Gym,用镜像内置 `/run_tests.sh`） |
| `server/sglang_client.py` | 79 | CPU agent | httpx 异步调 SGLang `/generate`，解析 `output_token_logprobs` |
| `scripts/launch_sglang.sh` | 210 | GPU pod | 启 Ray cluster + 起 SGLang engines + router（调 Kiro 的 `launch_external_sglang.py`）+ 写 env 文件 |
| `scripts/launch_swe_agent_workers_cpu.sh` | 180 | CPU pod | 等 SGLang env → 启 dockerd → flock-append URL 到 swe_agents.env → 启 FastAPI |
| `scripts/run_swe_rl_4b_kiro_trainer.sh` | 270 | Trainer pod | 读 2 个 env → 起 trainer Ray → slime `train.py`（4B hyperparams）|
| `scripts/run_swe_rl_30b_kiro_trainer.sh` | 310 | Trainer pod | 读 2 个 env → 起 trainer Ray → slime `train_async.py`（Kiro 30B 对齐）|
| `data/preprocess_swegym.py` | 220 | 预处理机 | HF SWE-Gym / SWE-Bench → parquet, 预渲染 `eval_script`(swe_harness 策略) |
| `data/preprocess_r2egym.py` | 200 | 预处理机 | HF R2E-Gym → parquet, 复制 `docker_image` + `expected_output_json`(r2e 策略) |
| `data/download_swe_images.sh` | 180 | 预处理机 | 读 parquet instance 列表 → `docker pull` → tag canonical → `docker save` → `.tar.gz`（支持代理） |
| `data/README.md` | — | — | Parquet schema 规范(两种 schema + data_source 分派规则) |

---

## 4. 与 Kiro 对应关系

### 文件映射

| Kiro (`slime/examples/kiro_agent/`) | Ours (`aws_kiro_infra/`) | 差异 |
|---|---|---|
| `sglang_rollout_server/job_yamsl/launch_sglang_only_async.yaml` | `k8s/launch_sglang_{4b,30b}.yaml` | replicas / model path |
| `sglang_rollout_server/job_yamsl/launch_kos_only_async.yaml` | `k8s/launch_swe_agent_workers.yaml` | image / runs swe_agent_server 而非 KoS |
| `jobs/slime_train_async_debug_run.yaml` | `k8s/launch_trainer_{4b,30b}.yaml` | trainer script / 模型大小 |
| `sglang_rollout_server/launch_sglang_only.sh` | `scripts/launch_sglang.sh` | 一致（抄过来） |
| `sglang_rollout_server/launch_kos_only.sh` | `scripts/launch_swe_agent_workers_cpu.sh` | KoS server 替换成 FastAPI uvicorn |
| `sglang_rollout_server/launch_external_sglang.py` | **直接引用** | 无改动 |
| `run-qwen3-30b-kiro-kos-async-debug-run.sh` | `scripts/run_swe_rl_30b_kiro_trainer.sh` | 1:1 对齐 |
| `kiro_generate_with_kos.py` | `generate_kiro.py` + `generate_with_swe_remote.py` + `server/mini_swe_agent.py` | Trainer 端薄了（agent 逻辑搬到 server） |
| `Kiro-on-Strands/remote_rollout_server/sglang_remote_rollout.py` | `server/swe_agent_server.py` | FastAPI 形态一致，scaffold 从 Strands 换成 mini-swe-agent |

### 概念映射

| 概念 | Kiro 生产 | 我们 |
|---|---|---|
| Scaffold | Strands + Qwen tool_call（JSON tool_call） | mini-swe-agent（` ```bash ... ``` ` + `COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` 哨兵） |
| Agent 循环位置 | CPU pod（KoS server 内） | CPU pod（swe_agent_server 内）✓ 一致 |
| Docker 容器粒度 | 每 SWE-Bench 实例一个 docker | 每 SWE-Bench 实例一个 docker ✓ 一致 |
| `/generate_single` → trainer 侧返回 | Trajectory file + logprobs file（TITO） | Per-turn `messages[i].token_ids/token_mask/token_logprobs` inline JSON |
| Token 追踪 | TITO JSON 文件读取 + `response_mask` | 直接用 slime 的 per-turn token 方式（swe-rl 原生） |
| Coordination 文件 | `sglang_external_rollout.env` + `kos_servers.env` | `sglang_external_rollout.env` + `swe_agents.env` |
| Eval harness | KoS server 内 | swe_agent_server 内（`patch_utils.grade_eval_output`） |

### 算法 recipe 对比（4B Kiro-style vs 30B rllm/DeepSWE）

| 维度 | **30B（rllm/DeepSWE GRPO++,sync）** | **4B（Kiro-style 保留）** |
|---|---|---|
| 数据集 | **R2E-Gym-Subset 4578**（DeepSWE 官方过滤版） | SWE-Gym |
| train script | `train.py`（sync） | `train.py`（sync） |
| Clip | **非对称 0.2 / 0.28** (DAPO high) | DAPO 非对称 0.2 / 0.28 |
| TIS | 无（sync 不需要） | 无 |
| KL loss | **无**（不传 `--use-kl-loss`）| `--use-kl-loss --kl-loss-coef 0.0` |
| Length norm | **Dr.GRPO** via `swe_rl_loss_reducers.dr_grpo_length_norm_reducer` | 无 |
| Compact filter | **开** via `generate_kiro_rllm.generate`（只保留 voluntary submit） | 无 |
| Aux LLM judge | **无**（strict 0/1 ORM） | 开（`coef=0.5`） |
| Advantage estimator | `grpo` + `--grpo-std-normalization False` (≈ LOOP) | `grpo`（带 std） |
| Over-sampling | `--over-sampling-batch-size 16` | 无 |
| Rollout-batch × n-samples | **8 × 8 = 64**（DeepSWE 官方）| 4 × 4 = 16 |
| Response len | **32768**（DeepSWE 官方） | 4096 |
| Parallelism (TP/PP/CP/EP) | 4/2/8/8 | 4/1/1/1 |
| Entrypoint | `generate_kiro_rllm.generate` | `generate_kiro.generate` |
| Trainer script | `scripts/run_swe_rl_30b_rllm_trainer.sh` | `scripts/run_swe_rl_4b_kiro_trainer.sh` |

老 Kiro async 配方 (`scripts/run_swe_rl_30b_kiro_trainer.sh`) 保留作为 fallback，不是默认。

---

## 5. HTTP API 合约

### `POST /generate_trajectory`

**Request body**（`server/swe_agent_server.py::GenerateRequest`）：

```json
{
  "instance_id": "django__django-12345",
  "problem_statement": "When using ...",
  "instance": {
    "instance_id": "...",
    "base_commit": "...",
    "FAIL_TO_PASS": ["tests/test_foo.py::test_bar"],
    "PASS_TO_PASS": [...],
    "patch": "<gold patch for metadata>",
    "eval_script": "<optional pre-rendered eval>",
    ...
  },
  "data_source": "swe-gym",
  "sampling_params": {"temperature": 1.0, "top_p": 0.95, "max_new_tokens": 4096},
  "sglang_router_url": "http://10.0.1.5:30000",
  "max_context_len": 32768,
  "max_new_tokens": 4096,
  "step_limit": 30,
  "skip_eval": false,
  "eval_timeout": 300,
  "exec_timeout": 180,
  "strict_no_test": true,
  "strict_no_config": true,
  "test_patch_policy_scope": "eval_tests_only",
  "request_id": "django__django-12345__g0__i3__17123..."
}
```

**Response body**（`server/swe_agent_server.py::GenerateResponse`）：

```json
{
  "status": "success",
  "error": null,
  "messages": [
    {
      "role": "system",
      "content": "<system prompt>",
      "token_ids": [151644, 8948, 198, ...],
      "token_mask": [0, 0, 0, ...],
      "token_logprobs": [0.0, 0.0, 0.0, ...]
    },
    {
      "role": "user",
      "content": "<instance problem>",
      "token_ids": [...],
      "token_mask": [0, ...],
      "token_logprobs": [0.0, ...]
    },
    {
      "role": "assistant",
      "content": "```bash\nls\n```",
      "token_ids": [...],
      "token_mask": [0, 0, 1, 1, ..., 1, 0, 0],
      "token_logprobs": [0.0, 0.0, -0.12, -0.34, ...]
    },
    ...
  ],
  "step_debug": [
    {
      "step_idx": 0,
      "action": "ls",
      "returncode": 0,
      "output_len": 512,
      "output_head": "...",
      "output_tail": "...",
      "start_ts": 1720000000.0,
      "end_ts": 1720000001.5,
      "ok": true
    }
  ],
  "git_patch": "diff --git a/foo.py b/foo.py\n...",
  "patch_source": "submission",
  "exit_status": "submitted",
  "n_steps": 12,
  "reward": 1,
  "eval_result": {
    "ok": true,
    "resolved": true,
    "resolved_by": "harness_get_eval_report",
    "resolved_by_returncode": true,
    "grading_report": {...},
    "output": "<test stdout>",
    "apply_output": "..."
  },
  "policy": {
    "changed_files": ["foo.py"],
    "test_files": [],
    "violated": false,
    "reasons": []
  },
  "elapsed_seconds": 134.2
}
```

### `GET /healthz`

```json
{
  "ok": true,
  "docker_ok": true,
  "containers_running": "3",
  "tokenizer_loaded": true,
  "concurrent_slots_free": 13
}
```

---

## 6. Token 一致性说明

### swe-rl vs Kiro 的差异

- **Kiro** 采用 **TITO**（Token-In, Token-Out）模式：KoS server 把 full_token_ids / aligned_logprobs / response_mask 写到磁盘 `_logprobs.json`，trainer 读文件对齐
- **我们**（保留 swe-rl 做法）：server 在内存里逐 turn 收集 `token_ids` / `token_mask` / `token_logprobs`，JSON response 直接返回。完全不落盘

### 关键保证

1. **Tokenizer 在 CPU pod 本地加载**（从 `MODEL_PATH`）。apply_chat_template 结果和 trainer 一致，因为都是同一个 HF checkpoint。
2. **SGLang /generate 直接传 input_ids**（不传 messages），返回 `output_token_logprobs`（包含 token_id 和 logprob）。**不走 `/v1/chat/completions`，避免 SGLang 侧再 tokenize。**
3. **Assistant msg 的 token_ids** = `gen_prompt_tokens + output_token_ids + trailing_nl_tokens`（见 `server/mini_swe_agent.py::run_agent_loop` 的 line 230+）。`token_mask = [0]*|gen_prompt| + [1]*|output| + [0]*|trail|`
4. **Env-side msg**（system/user）token_ids 通过 `tokenizer.apply_chat_template([msg], tokenize=True)` 生成
5. **Trainer 侧 `_build_trajectory_sample`** 直接把 messages 里的 token_ids 拼起来，不再 tokenize

**监控**：slime 的 `rollout_probs_diff` 指标应接近 0。>10% 说明 tokenizer 不匹配（多半是 CPU pod `MODEL_PATH` 和 trainer `HF_CKPT` 指向的 checkpoint 不同）。

---

## 7. 双 eval strategy — SWE-Gym + R2E-Gym 同时支持

aws_kiro_infra 原生支持**两种 reward 判定机制**，按 parquet 行的 `data_source` 字段**运行时动态分派**。**同一个 trainer + 同一份 agent pod 池**可以同时处理混合数据集，mirror rllm 的 `_calculate_reward_*` 5 分支设计。

### 7.1 分派逻辑

`server/patch_utils.py::detect_eval_strategy`：

```python
def detect_eval_strategy(instance, data_source) -> Literal["swe_harness", "r2e"]:
    ds = (data_source or "").lower()
    # 1) 显式 tag 优先
    if "r2e-gym" in ds or "r2e_gym" in ds or ds == "r2e":
        return "r2e"
    if "swe-gym" in ds or "swe-bench" in ds:
        return "swe_harness"
    # 2) 字段启发式兜底: 有 expected_output_json 且没 FAIL_TO_PASS → R2E
    if instance.get("expected_output_json") and not instance.get("FAIL_TO_PASS"):
        return "r2e"
    return "swe_harness"
```

### 7.2 两种策略的判定流程对照

| 步骤 | `swe_harness` (SWE-Gym / SWE-Bench) | `r2e` (R2E-Gym) |
|---|---|---|
| ① Image 解析 | 从 `instance_id` 推导（`xingyaoww/sweb.eval.x86_64.<iid_s_>:latest`） | 直接用 `instance.docker_image` 字段 |
| ② 起 fresh eval 容器 | 同左 | 同左 |
| ③ Apply patch | `git reset --hard HEAD && git apply <<PATCH...` | 同左 |
| ④ 跑测试 | `bash <<EVAL\n<parquet.eval_script>\nEVAL` | `bash /run_tests.sh`（镜像内置） |
| ⑤ 解析输出 | swebench/swegym `get_logs_eval` → 状态字典 | `parse_r2e_pytest_log`（pytest short summary）→ 状态字典 |
| ⑥ Grading 逻辑 | `get_eval_report` 看 `FAIL_TO_PASS` / `PASS_TO_PASS` 是否全通 | 状态字典和 `instance.expected_output_json` **逐键比对** |
| ⑦ Reward | 全通 → 1，否则 0 | 字典完全 match → 1，否则 0 |

两条都是 0/1 稀疏 ORM，差异只在**谁定义"通过"**：
- SWE-Bench 风格：列出要变 pass 的测试（FAIL_TO_PASS）+ 不能破坏的测试（PASS_TO_PASS）
- R2E-Gym 风格：给出**所有测试的 gold 状态字典**（`{"test_a": "PASSED", "test_b": "FAILED", ...}`），agent 跑出来的字典必须**完全一致**

### 7.3 为什么 CPU pod 镜像**不需要**装 `r2egym` 包

这是一个非显而易见的优化。R2E-Gym 官方包（`r2egym/`）是一个**~20,000 LOC 的完整 agent 环境框架**，但我们只需要其中**一小撮 reward 判定逻辑**（~50 行纯 Python），已经**整段内联进 `server/patch_utils.py`**。

#### r2egym 包的分层

| r2egym 子模块 | 作用 | rllm 用到 | 我们用到 |
|---|---|---|---|
| `agenthub.environment.RepoEnv` | gym-style 环境抽象 | ✅ | ❌ 我们有 `server/swe_agent_server.py` |
| `agenthub.runtime.DockerRuntime` | 容器管理 + K8s 编排（~2000 LOC） | ✅ | ❌ 我们有 `server/docker_ops.py` |
| `agenthub.tools.*`（4 个原生 tool） | Strands-like scaffold | ✅ | ❌ 我们用 mini-swe-agent bash backtick |
| `agenthub.action.Action` | tool_call 解析 | ✅ | ❌ 我们用正则 `parse_bash_action` |
| `repo_analysis.execution_log_parser.parse_log_pytest` | 解析 pytest 输出（**35 行**） | ✅ | ✅ **已 copy 进 `patch_utils.py::parse_r2e_pytest_log`** |
| `DockerRuntime._calculate_reward_r2e` 里的字典比对 | reward 计算（**~20 行**） | ✅ | ✅ **已重写成 `grade_eval_output_r2e`** |

#### 装 r2egym 的 transitive 依赖

```
r2egym → gymnasium                 # gym 环境
       → docker-py ≥ 6.0           # Python docker SDK
       → boto3                     # AWS SDK (K8s runtime)
       → kubernetes ≥ 30           # K8s client
       → sweagent ≥ 0.3            # Princeton sweagent
       → litellm / anthropic / openai
       → datasets / transformers
       → (还有 20+ transitive 包)
```

镜像体积代价：**+500~800 MB**。

#### 我们的做法：内联 ~100 行 pure Python

```python
# server/patch_utils.py (~100 LOC 两个函数)
def parse_r2e_pytest_log(log) -> dict[str, str]:
    # 读 "short test summary info" tail → {test_name: "PASSED"|"FAILED"|"ERROR"}
    ...

def grade_eval_output_r2e(instance, eval_output) -> dict:
    expected = json.loads(instance["expected_output_json"])
    parsed = parse_r2e_pytest_log(eval_output)
    # 归一化 key + 逐键比对
    ...
```

**依赖**：`re` + `json`（标准库）。零额外包。

#### 我们从 R2E-Gym 生态仍然**需要**的

| 物料 | 我们要不要 |
|---|---|
| `r2egym` Python 包 | ❌ 不需要 |
| R2E-Gym Docker images | ✅ 需要（用 `aws_kiro_infra/data/download_swe_images.sh` 下 tarball） |
| R2E-Gym HF dataset（`R2E-Gym/*`） | ✅ 需要（`preprocess_r2egym.py` 读里面的 `docker_image` + `expected_output_json`） |

### 7.4 混合训练 / 单独训练

**只训 SWE-Gym**：
```bash
python3 data/preprocess_swegym.py --output-path /data/swegym.parquet
# parquet 每行 data_source="swe-gym" → 都走 swe_harness 路径
```

**只训 R2E-Gym**：
```bash
python3 data/preprocess_r2egym.py --output-path /data/r2egym.parquet
# parquet 每行 data_source="r2e-gym" → 都走 r2e 路径
```

**混合训练**（一个 trainer 两种 reward 机制）：
```python
import pandas as pd
a = pd.read_parquet("/data/swegym.parquet")
b = pd.read_parquet("/data/r2egym.parquet")
mixed = pd.concat([a, b], ignore_index=True).sample(frac=1, random_state=0)
mixed.to_parquet("/data/mixed.parquet", index=False)
```

每条 sample 独立分派——server 逐行判断走哪条路径。CPU agent pod 池**不需要**为两种数据分别起两套 pod。

---

## 8. 部署

### K8s 基础设施前置条件

1. **容器镜像**：`975050351917.dkr.ecr.us-east-2.amazonaws.com/q-codegen:swerl-aws-kiro-0.1.0`
   - 需要预装：`slime` + `Megatron-LM` + `transformers` + `docker` CLI + `fastapi` + `uvicorn` + `httpx` + `loguru` + `pyyaml` + `jinja2` + `swebench`（和/或 `swegym`）+ `litellm`（trainer 用）
2. **HyperPod namespace**：`hyperpod-ns-aladdin`（或改 YAML 里的 namespace）
3. **ServiceAccount**：`bedrock-sa`（或改 YAML）
4. **Kueue queue**：`hyperpod-ns-aladdin-localqueue` + `high-p5-priority` class
5. **PVCs**：
   - `private-model-rw`（挂到 `/mnt_out`）—— 存代码、checkpoints、logs、env 文件、SWE image 和数据
   - `private-data-rw`（挂到 `/mnt_private`）—— 可选，存额外只读数据
6. **K8s Secrets**（4B 用 aux judge）：
   - `openai-api-key`（或 `anthropic-api-key`）

---

### 数据 / 镜像 / Checkpoint 准备（PVC 内容）

**这是每次新环境上线最容易忽视的部分。3 样东西全部要在 PVC 上就位，trainer / agent pod 才能跑。**

全部可以**在任一能访问 docker hub 的预处理机器上**提前跑完，然后 `rsync` 到 PVC。

#### 目标 PVC 布局

```
/mnt_out/<USER>/
├── codebase/
│   ├── slime/                       # git clone
│   ├── OpenClaw-RL/swe-rl/          # 本项目
│   └── Megatron-LM/                 # git clone
├── models/
│   ├── Qwen3-4B-Instruct-2507/                   # HF format (4B)
│   ├── Qwen3-Coder-30B-A3B-Instruct/             # HF format (30B)
│   └── Qwen3-Coder-30B-A3B-Instruct-mcore/       # mcore format (30B ref_load)
├── data/
│   ├── <your-dataset>.parquet       # prompt 数据(trainer 读)
│   └── swe_images/                  # Docker image tarballs(CPU pod 读)
│       ├── getmoto__moto-7365.tar.gz
│       ├── django__django-12345.tar.gz
│       └── …
└── logs/swerl-aws-kiro/              # 自动生成(env 协调文件 + trajectories)
```

#### Step 1: 准备 Prompt parquet

详见 [`data/README.md`](data/README.md)。schema 速查：

| 列 | 类型 | 说明 |
|---|---|---|
| `prompt` | `list[dict]` | `[{"role":"user","content":"<problem_statement>"}]` |
| `data_source` | `str` | `"swe-gym"` 或 `"swe-bench"`（决定 Docker image 命名） |
| `ability` | `str` | `"coding"` 固定 |
| `instance` | `dict` | 含 `instance_id` / `repo` / `base_commit` / `FAIL_TO_PASS` / `PASS_TO_PASS` / `patch` / `test_patch` / `version` / `eval_script`（★**必须预渲染**） |

用我们脚本一键产出：

```bash
# 依赖(只在预处理机器上装)
pip install datasets pandas pyarrow
pip install git+https://github.com/SWE-Gym/SWE-Gym@main   # for make_test_spec

# SumanthRH/SWE-Gym-Subset, 293 instances
python3 aws_kiro_infra/data/preprocess_swegym.py \
  --output-path /data/swe_my_train.parquet

# 完整 SWE-Gym, ~2500 instances
python3 aws_kiro_infra/data/preprocess_swegym.py \
  --data-source SWE-Gym/SWE-Gym --split train \
  --output-path /data/swegym_full.parquet

# SWE-Bench Verified, 500 instances(当 eval 集)
python3 aws_kiro_infra/data/preprocess_swegym.py \
  --data-source SumanthRH/SWE-bench_Verified --split test \
  --data-source-tag swe-bench-verified \
  --output-path /data/swebench_verified.parquet

# Smoke test(10 条,快速验证格式)
python3 aws_kiro_infra/data/preprocess_swegym.py --max-samples 10 \
  --output-path /tmp/smoke.parquet

# ─── R2E-Gym(自动走 r2e eval 策略) ───
# 默认 = DeepSWE 官方 4578 条(R2E-Gym-Subset,已过滤 SWE-Bench-Verified 同 repo)
python3 aws_kiro_infra/data/preprocess_r2egym.py \
  --output-path /data/r2egym_subset/train.parquet

# 未过滤的 V1(8101 条,有 data leak 风险,不建议训练)
# python3 aws_kiro_infra/data/preprocess_r2egym.py \
#   --data-source R2E-Gym/R2E-Gym-V1 --output-path /data/r2egym_v1/train.parquet

# ─── 混合训练:两份 parquet concat ───
python3 -c "
import pandas as pd
a = pd.read_parquet('/data/swegym_full/train.parquet')
b = pd.read_parquet('/data/r2egym_subset/train.parquet')
mixed = pd.concat([a, b]).sample(frac=1, random_state=0)
mixed.to_parquet('/data/mixed/train.parquet', index=False)
print(mixed['data_source'].value_counts())
"
```

产出大小：**几 MB 一个 parquet**。

> **Teammate 自产同格式数据**：只要 parquet 满足 [`data/README.md`](data/README.md) §1-§3 schema 就行(不用我们脚本)。SWE-Gym/SWE-Bench 伪代码见 §5,R2E-Gym 用 `data_source="r2e-gym"` + `instance.docker_image` + `instance.expected_output_json`。

#### Step 2: 下载 SWE-Bench Docker image tarballs

**必须做**。`server/docker_ops.py::create` 用 `--pull never`，image 不提前放到 PVC，`docker run` 直接失败。

```bash
# 从 parquet 自动抽 instance_id + data_source, docker pull → docker save | gzip
bash aws_kiro_infra/data/download_swe_images.sh \
  --prompt-data /data/swe_my_train.parquet \
  --output-dir  /data/swe_images/ \
  --parallel 4                        # 4 个并行拉

# 只要前 10 个(配合 smoke parquet)
bash aws_kiro_infra/data/download_swe_images.sh \
  --prompt-data /tmp/smoke.parquet \
  --output-dir  /tmp/swe_images_10 \
  --max 10
```

脚本做的事：
1. 从 parquet / JSONL 抠出 `(instance_id, data_source)` 列表
2. 解析对应 Docker image 名字（与 `server/mini_swe_agent.get_docker_image_name` 同规则）
3. `docker pull` + 指数退避重试
4. `docker save <image> | gzip -1 > <iid>.tar.gz`
5. 清理本地 dockerd（避免磁盘爆）—— `KEEP_LOCAL_IMAGE=1` 可保留
6. 已存在的 tarball 跳过（支持断点续传）

**大小估计**：
- 单 instance tarball：~1-5 GB（Python env + 完整源码）
- 800 instances ≈ **2-4 TB**
- 完整 SWE-Gym 2500 instances ≈ **5-10 TB**
- 下载耗时：4 并发，800 instance 大约 **4-8 小时**（看 docker hub 下行速度）

#### Step 3: 准备 model checkpoint

**4B（HF format 就够）**:
```bash
pip install huggingface-hub
huggingface-cli download Qwen/Qwen3-4B-Instruct-2507 \
  --local-dir /data/models/Qwen3-4B-Instruct-2507
```

**30B（要 HF + mcore 两份）**：
```bash
# HF
huggingface-cli download Qwen/Qwen3-Coder-30B-A3B-Instruct \
  --local-dir /data/models/Qwen3-Coder-30B-A3B-Instruct

# mcore(trainer 的 --ref-load 用)
# 用 slime 提供的转换脚本(自己准备 slime repo)
bash slime/scripts/convert_hf_to_mcore.sh \
  --hf-checkpoint /data/models/Qwen3-Coder-30B-A3B-Instruct \
  --save-dir      /data/models/Qwen3-Coder-30B-A3B-Instruct-mcore
```

大小：**4B ~8 GB / 30B ~60 GB**。

#### Step 4: rsync 到 PVC

```bash
# parquet
rsync -avz /data/swe_my_train.parquet \
  <bastion>:/mnt_out/<USER>/data/

# tarballs(大! 建议用 tmux + --progress,断了继续)
rsync -avz --progress /data/swe_images/ \
  <bastion>:/mnt_out/<USER>/data/swe_images/

# 模型
rsync -avz /data/models/ \
  <bastion>:/mnt_out/<USER>/models/

# 代码
rsync -avz --exclude '__pycache__' --exclude '.git' \
  /data/OpenClaw-RL/swe-rl/ \
  <bastion>:/mnt_out/<USER>/codebase/OpenClaw-RL/swe-rl/
# slime / Megatron-LM 同理
```

#### 路径对照：YAML 变量 ↔ PVC 路径

|YAML 文件|变量|改成你的实际路径|
|---|---|---|
| `k8s/launch_trainer_4b.yaml` | `HF_CKPT` | `/mnt_out/<you>/models/Qwen3-4B-Instruct-2507` |
| 同 | `PROMPT_DATA` | `/mnt_out/<you>/data/<yourdataset>.parquet` |
| `k8s/launch_trainer_30b.yaml` | `HF_CKPT` | 30B HF 路径 |
| 同 | `REF_LOAD` | 30B **mcore** 路径 |
| 同 | `PROMPT_DATA` | `/mnt_out/<you>/data/<yourdataset>.parquet` |
| `k8s/launch_sglang_{4b,30b}.yaml` | `MODEL_PATH` | 与 `HF_CKPT` **相同路径** |
| `k8s/launch_swe_agent_workers.yaml` | `MODEL_PATH` | 与 `HF_CKPT` **相同路径**（CPU pod 本地加载 tokenizer） |
| 同 | `SWE_DOCKER_IMAGES_PATH` | `/mnt_out/<you>/data/swe_images/` |
| 同 | `SGLANG_RUN_NAME` | 必须匹配 `launch_sglang_{4b,30b}.yaml` 的 `RUN_NAME` |
| 所有 trainer/agent YAML | `SLIME_DIR` / `SWE_RL_DIR` / `MEGATRON_LM_PATH` | 你 clone 的代码路径 |

#### 不上 K8s 也能先冒烟（推荐首次）

```bash
# 1. 小样本 parquet + 对应 10 个 image tar
python3 aws_kiro_infra/data/preprocess_swegym.py --max-samples 10 \
  --output-path /tmp/smoke.parquet
bash aws_kiro_infra/data/download_swe_images.sh \
  --prompt-data /tmp/smoke.parquet --output-dir /tmp/swe_images_10

# 2. 本地单机起 SGLang + swe_agent_server
export MODEL_PATH=/data/models/Qwen3-4B-Instruct-2507
export SWE_DOCKER_IMAGES_PATH=/tmp/swe_images_10
python3 -m sglang.launch_server --model-path $MODEL_PATH --port 30000 --tp 1 &
python3 -m uvicorn aws_kiro_infra.server.swe_agent_server:app --port 5000 &

# 3. curl 发一个请求,确认 pipeline 跑通
#    示例 request 结构见 §5 HTTP API 合约
```

跑通了再上 K8s。

---

### 部署步骤：4B（smoke test 推荐先跑这个）

```bash
cd /data_storage/wyj/jxl/slime/examples/aws_kiro_infra

# Step 1: 确认 k8s/launch_swe_agent_workers.yaml 里 MODEL_PATH 是 4B 的 ckpt
#         确认 SGLANG_RUN_NAME 是 "swerl_4b_sglang"（匹配 Job 1）

# Step 2: 起 SGLang pool（GPU,~5 min 加载模型 + 启 Ray）
kubectl apply -f k8s/launch_sglang_4b.yaml

# Step 3: 起 CPU agent pool（可以和 Step 2 并行;会等 SGLang env 文件）
kubectl apply -f k8s/launch_swe_agent_workers.yaml

# Step 4: 等两个 env 文件落到 PVC
ls -la /mnt_out/jinxiaolong/logs/swerl-aws-kiro/swerl_4b_sglang/sglang_external_rollout.env
ls -la /mnt_out/jinxiaolong/logs/swerl-aws-kiro/swerl_swe_agents/swe_agents.env

# Step 5: 健康检查
kubectl get pods -n hyperpod-ns-aladdin | grep swerl
# 每个 CPU pod 的 /healthz 应返回 docker_ok=true, tokenizer_loaded=true

# Step 6: 起 trainer
kubectl apply -f k8s/launch_trainer_4b.yaml

# Step 7: 跟 trainer log
kubectl logs -f <trainer-pod-worker-0> -n hyperpod-ns-aladdin
```

### 部署步骤：30B（Kiro 生产）

```bash
# 1. 编辑 k8s/launch_swe_agent_workers.yaml:
#    MODEL_PATH     = Qwen3-Coder-30B-A3B-Instruct
#    SGLANG_RUN_NAME = swerl_30b_sglang
#    replicas: 80   (Kiro production 规模)

# 2. 同样三步:
kubectl apply -f k8s/launch_sglang_30b.yaml         # 8 × p5
kubectl apply -f k8s/launch_swe_agent_workers.yaml  # 80 × m5
# 等两个 env 文件
kubectl apply -f k8s/launch_trainer_30b.yaml        # 16 × p5
```

### 4B 和 30B 同时跑（不同 RUN_NAME）

需要部署**两份** `launch_swe_agent_workers.yaml`（因为 tokenizer 不同）：
- 复制文件，改 metadata.name 加后缀 `-4b` / `-30b`
- 改 MODEL_PATH / SGLANG_RUN_NAME / RUN_NAME
- Trainer 的 `SWE_AGENT_RUN_NAME` 指向对应的那份

### 停服（逆序）

```bash
kubectl delete pytorchjob swerl-{4b,30b}-trainer      -n hyperpod-ns-aladdin
kubectl delete pytorchjob swerl-swe-agent-workers     -n hyperpod-ns-aladdin
kubectl delete pytorchjob swerl-{4b,30b}-sglang       -n hyperpod-ns-aladdin
```

---

## 9. 环境变量参考

### SGLang pool（Job 1）
| 变量 | 4B 默认 | 30B 默认 | 作用 |
|---|---|---|---|
| `MODEL_PATH` | Qwen3-4B-Instruct-2507 | Qwen3-Coder-30B-A3B-Instruct | HF checkpoint |
| `TP_SIZE` | 8 | 8 | SGLang TP |
| `MEM_FRACTION` | 0.80 | 0.80 | GPU mem 预留比例 |
| `CHUNKED_PREFILL_SIZE` | -1 (关) | 8192 | 长 context 优化 |
| `ROUTER_PORT` | 30000 | 30000 | Router 端口 |
| `RUN_NAME` | `swerl_4b_sglang` | `swerl_30b_sglang` | env 文件路径片段（**trainer 要匹配**） |

### CPU agent pool（Job 2）
| 变量 | 默认 | 作用 |
|---|---|---|
| `MODEL_PATH` | — | 本地加载 tokenizer 用，**必须和 SGLang 加载的一致** |
| `AGENT_PORT` | 5000 | FastAPI 端口 |
| `AGENT_MAX_CONCURRENT` | 16 | 同 pod 内同时跑的 agent 数 |
| `SGLANG_RUN_NAME` | — | **必须匹配 Job 1 的 RUN_NAME** |
| `SWE_DOCKER_IMAGES_PATH` | — | `{iid}.tar.gz` 目录，离线加载 SWE 镜像 |
| `SWE_CONTAINER_MEMORY` | 8g | 单容器内存上限 |
| `SWE_CONTAINER_PIDS_LIMIT` | 1024 | 单容器进程数上限 |
| `SWE_CONFIG_PATH` | `swebench.yaml` | Scaffold 配置 |

### Trainer pool（Job 3）
| 变量 | 4B 默认 | 30B 默认 | 作用 |
|---|---|---|---|
| `SGLANG_RUN_NAME` | swerl_4b_sglang | swerl_30b_sglang | 必须匹配 Job 1 |
| `SWE_AGENT_RUN_NAME` | swerl_swe_agents | swerl_swe_agents | 必须匹配 Job 2 |
| `SGLANG_ENV_BASE` | `/mnt_out/jinxiaolong/logs/swerl-aws-kiro` | 同 | PVC 上的 env 目录 |
| `HF_CKPT` | Qwen3-4B-Instruct-2507 | 30B HF | 训练 checkpoint |
| `REF_LOAD` | 同 HF_CKPT | 30B **mcore** | 参考 model（30B 要预转 mcore） |
| `PROMPT_DATA` | 4B: SWE-Gym parquet | 30B: **`r2egym_subset_train.parquet`**（R2E-Gym-Subset 4578, DeepSWE 官方） | 数据集 |
| `ROLLOUT_BATCH_SIZE` | 4 | 16 | |
| `N_SAMPLES_PER_PROMPT` | 4 | 16 | |
| `ROLLOUT_MAX_RESPONSE_LEN` | 4096 | 73728 | |
| `TENSOR_MODEL_PARALLEL_SIZE` | 4 | 4 | |
| `PIPELINE_MODEL_PARALLEL_SIZE` | 1 | 2 | |
| `CONTEXT_PARALLEL_SIZE` | 1 | 8 | |
| `EXPERT_MODEL_PARALLEL_SIZE` | 1 | 8 | MoE, 30B 才用 |
| `MAX_TOKENS_PER_GPU` | 9216 | 32768 | |
| `UPDATE_WEIGHTS_INTERVAL` | — | 1 | async 用 |
| `OVER_SAMPLING_BATCH_SIZE` | — | 32 | 30B 用（动态 filter 用） |
| `SWE_MAX_CONCURRENT` | 128 | 1024 | Trainer 侧并发上限 |
| `SWE_ROLLOUT_TIMEOUT` | 1800 | 3600 | 单 trajectory 超时 |
| `SWE_EVAL_TIMEOUT` | 300 | 600 | Eval 超时 |
| `AUX_REWARD_COEF` | 0.5 | 不用 | 仅 4B |
| `AUX_JUDGE_MODEL` | `openai/gpt-4o-mini` | 不用 | 仅 4B，litellm 路径 |
| `OPENAI_API_KEY` | (from secret) | 不用 | 仅 4B |

### 跨 pool 共用
| 变量 | 作用 |
|---|---|
| `PET_NNODES` | PyTorchJob 自动设；控制 Ray cluster 期望节点数 |
| `NCCL_TIMEOUT` | 12000 (s) |
| `SWE_STRICT_NO_TEST_PATCH` | 1 = 禁止 patch 改 eval 测试文件（default） |
| `SWE_STRICT_NO_CONFIG_PATCH` | 1 = 禁止 patch 改 pyproject.toml 等 |
| `SWE_TEST_PATCH_POLICY_SCOPE` | `eval_tests_only` 或 `all_tests` |
| `SWE_SAVE_TRAJ_DIR` | Trainer 侧保存 rollout artifacts 路径 |
| `SWE_STEP_LIMIT` | Agent loop 最大 turn 数（默认 30） |

---

## 10. 故障排查

### `Waiting for sglang_external_rollout.env` 卡死

- 检查 SGLang pod：`kubectl logs <pod> | tail -100`
- 确认 `launch_external_sglang.py` 输出 `All servers healthy!` + `Router /health OK`
- 4B 模型加载 2-3 分钟，30B 5-8 分钟；若超过 15 分钟还没就绪，多半是 GPU OOM 或 Ray cluster 没起
- 如果 rank ≥ 1 worker 没 join Ray：检查节点之间 NCCL/EFA 连通

### `Waiting for swe_agents.env` 卡死

- 检查 CPU pod：`kubectl logs <pod> | tail -100`
- 常见问题：
  - **dockerd 起不来**：看 `/tmp/dockerd.log`，多半是 privileged 没加 / `/dev/shm` 不够大
  - **tokenizer 加载失败**：`MODEL_PATH` 路径不对，或 HF 依赖缺
  - **Router 不可达**：网络策略拦了 GPU pod 的 30000 端口
  - **flock 写 env 文件超时**：共享 PVC I/O 压力大，重启 pod 即可

### Trainer 起来但所有 rollout 失败

- 对每个 URL `curl <url>/healthz`：
  - `docker_ok=false` → 重启该 CPU pod
  - `tokenizer_loaded=false` → MODEL_PATH 路径/权限问题
  - `concurrent_slots_free=0` → 该 pod 已打满，增加 CPU pod replicas
- 看 trainer 日志里 `[SWE-KIRO] [<iid>] remote call failed: ...` 错误类型：
  - `ConnectError` → 网络问题
  - `ReadTimeout` → agent 循环超时，调大 `SWE_REMOTE_HTTP_TIMEOUT`
  - `ValidationError` → 请求 payload 格式不对

### 训练 loss 异常（NaN / 暴跌）

- 看 `rollout_probs_diff`（wandb）——  > 0.05 说明 tokenizer drift 了
- 确认 Job 1 和 Job 2 用的是**同一个** `MODEL_PATH`
- 确认 trainer 的 `HF_CKPT` 也是同一个
- 如果 SGLang 没返回 logprobs，server 会 fallback 到 re-tokenize（会打日志 `Router didn't return token ids`），这种情况 token_logprobs 都是 0.0，TIS / rollout_probs_diff 会废

### 30B eval harness `ModuleNotFoundError: swegym`

- CPU pod 镜像需要装 `swegym`（swe-gym 数据用）和 `swebench`（swe-bench 数据用）
- 我们的 `patch_utils.get_harness_tools` 会按数据类型 try-import；两个都要有

### SGLang OOM（4B / 30B）

- 降 `MEM_FRACTION`（0.80 → 0.70）
- 30B 如果在 `chunked_prefill` 阶段 OOM：降 `CHUNKED_PREFILL_SIZE`（8192 → 4096）

### `docker run` failed: image not found / `--pull never`

- `server/docker_ops.py::create` 用 `--pull never`，所以 `<iid>.tar.gz` 必须在 `${SWE_DOCKER_IMAGES_PATH}` 目录里
- 检查：`ls <SWE_DOCKER_IMAGES_PATH>/<instance_id>.tar.gz`
- 没有就先跑 `aws_kiro_infra/data/download_swe_images.sh`（见 §8 Step 2）
- 如果不想用 tarballs（测试环境），可以临时改 `docker_ops.py::create` 里的 `--pull never` → `--pull missing`（每次 rollout 会走 registry pull，慢但能跑）

### 训练启动但所有 rollout 显示 `eval_script unavailable`

- parquet 里的 `instance.eval_script` 字段为空 / 缺失
- 原因：预处理脚本跑 `--skip-eval-script` 或者 `make_test_spec` 在预处理机没装好
- 修复：重跑 `aws_kiro_infra/data/preprocess_swegym.py`（装好 `swegym` / `swebench`），不加 `--skip-eval-script`
- 验证：`python3 -c "import pandas as pd; df=pd.read_parquet('train.parquet'); print(df.iloc[0]['instance'].get('eval_script','')[:200])"` 应输出 bash 脚本头 `#!/bin/bash ...`

### 预处理机上 `docker exec` 返回空（嵌套容器 + rootless docker 的坑）

- 症状：`docker load` 成功、`docker run` 成功、`docker ps` 显示 Up，但 `docker exec <c> pwd` 返回 `output=[] exit=0`，所有 exec 无输出也无报错
- 根因：`docker info` 看一下 —— `Storage Driver: fuse-overlayfs` + `Cgroup Driver: none` + `Operating System: ...(containerized)`。这是 **rootless docker 跑在容器里的嵌套环境**，`docker exec` 的 stdio 管道在 fuse-overlayfs 下会被某层劫持断流（Ubuntu 20.04 containerized 常见 bug）
- 影响范围：**仅影响本机做冒烟测试**。**不影响 K8s CPU pod** —— HyperPod 的 privileged pod 跑的是标准 dockerd（overlay2 + cgroup v2）
- 验证 tarball 本身 OK：看 `docker run -d ...` 返回 container id + `docker ps` 显示 Up 就够了，就能上 PVC
- 真正的 `docker exec` 冒烟：SSH 进任一 CPU agent pod 后做（`kubectl exec -it <agent-pod> -- bash`）

### R2E-Gym: `expected_output_json missing (required for r2e strategy)`

- 症状：trainer log 里看到这条 error，对应的 rollout reward=0 且 `policy=None` / `eval_result.resolved=False`
- 根因：parquet 行的 `data_source` 含 `r2e-gym` → server 走 r2e 路径，但 `instance.expected_output_json` 字段为空 / 缺失
- 修复：确保 `preprocess_r2egym.py` 正确读了 HF dataset 的 `expected_output_json` 列。验证：
  ```bash
  python3 -c "
  import pandas as pd
  df = pd.read_parquet('r2egym.parquet')
  r = df.iloc[0]
  print('data_source:', r['data_source'])
  print('has expected_output_json:', bool(r['instance'].get('expected_output_json')))
  print('first 200 chars:', r['instance'].get('expected_output_json', '')[:200])
  "
  ```
- 若混合数据集（SWE-Gym + R2E-Gym 拼一起），**不同行走不同 eval 策略是预期行为**，不是错误

### R2E-Gym: `docker run` 报 `No such image` / `--pull never`

- R2E-Gym image **不能从 `instance_id` 推导**，`get_docker_image_name` 依赖 `instance.docker_image` 字段
- 常见漏洞：手搓 parquet 时忘了带 `docker_image` 字段 → `get_docker_image_name` 走 SWE-Gym 推导路径 → 推出错的 image 名（或直接 raise `ValueError` 因为带 `r2e` 标记没 docker_image）
- 验证：
  ```bash
  python3 -c "
  import pandas as pd
  df = pd.read_parquet('r2egym.parquet')
  for r in df.iloc[:3].to_dict('records'):
      print(r['instance'].get('docker_image'))
  "
  ```
  应输出非空字符串（e.g. `docker.io/r2egym/repo_commit:latest`）

### R2E-Gym: eval 全部 `test_count_mismatch`

- 症状：`eval_result.report.reason == "test_count_mismatch"`，`n_parsed != n_expected`
- 根因：pytest 输出的测试数量和 `expected_output_json` 不匹配。可能：
  - agent patch 导致某些测试 error（崩溃）而不是 FAILED，parse_r2e_pytest_log 可能没正确捕获
  - 镜像里 `/run_tests.sh` 版本不对（跑的测试集和构建 expected 时不一样）
- 修复：先拿 gold patch 跑一遍，如果 gold 都过不了 test_count check 说明 image / expected_output_json 不一致，要重新预处理数据

---

## 11. 本地开发 / 非 K8s 跑法

所有代码 K8s-agnostic。本地多机只需：

```bash
# Node A (GPU): SGLang
export PET_NNODES=2 PET_NODE_RANK=0 MASTER_ADDR=10.0.0.1
bash scripts/launch_sglang.sh \
  --model-path /data/models/Qwen3-4B-Instruct-2507 \
  --run-name dev_sglang \
  --output-dir /shared/logs \
  --slime-dir /data/slime

# Node B (GPU): SGLang worker
export PET_NNODES=2 PET_NODE_RANK=1 MASTER_ADDR=10.0.0.1
bash scripts/launch_sglang.sh --model-path ... --run-name dev_sglang --output-dir /shared/logs --slime-dir /data/slime

# Node C (CPU): swe_agent
bash scripts/launch_swe_agent_workers_cpu.sh \
  --sglang-env-file /shared/logs/dev_sglang/sglang_external_rollout.env \
  --model-path /data/models/Qwen3-4B-Instruct-2507 \
  --run-name dev_agents \
  --output-dir /shared/logs \
  --swe-rl-dir /data/OpenClaw-RL/swe-rl \
  --aws-kiro-dir /data/slime/examples/aws_kiro_infra

# Node D (GPU): trainer
export PET_NNODES=1 PET_NODE_RANK=0 MASTER_ADDR=<trainer-ip>
export SGLANG_ENV_BASE=/shared/logs SGLANG_RUN_NAME=dev_sglang SWE_AGENT_RUN_NAME=dev_agents
export HF_CKPT=... PROMPT_DATA=...
bash scripts/run_swe_rl_4b_kiro_trainer.sh
```

### 单机快速冒烟（只测 server）

如果只想测 `server/swe_agent_server.py` 而不跑训练：

```bash
# 1. 起一个 SGLang 单实例（进程级）
python3 -m sglang.launch_server \
  --model-path /data/models/Qwen3-4B-Instruct-2507 \
  --port 30000 --tp 1 &

# 2. 起 swe_agent_server
export MODEL_PATH=/data/models/Qwen3-4B-Instruct-2507
export PYTHONPATH=$(pwd)/..:$(pwd)/../.. 
python3 -m uvicorn aws_kiro_infra.server.swe_agent_server:app --host 0.0.0.0 --port 5000

# 3. 手动 POST 一个 trajectory
curl -X POST http://localhost:5000/generate_trajectory \
  -H "Content-Type: application/json" \
  -d @test_request.json
```

---

## 12. 扩展点

- **换 scaffold**（mini-swe-agent → 其他）：改 `server/mini_swe_agent.py::run_agent_loop` + `swebench.yaml` 模板。Trainer 侧无感。
- **换 agent 循环控制**（新增 tool / 不同退出条件）：改 `run_agent_loop` 的 for-loop 部分。HTTP API 不变。
- **改 reward 组合**（非 outcome + aux judge）：改 `generate_kiro.py::reward_func`。
- **加更多 policy**（禁某些文件模式）：改 `server/patch_utils.py::analyze_patch_policy`。
- **切 async → sync**（30B 实验性）：`run_swe_rl_30b_kiro_trainer.sh` 把 `train_async.py` 换成 `train.py`，删 `ASYNC_ARGS`。
- **加新数据源 / eval strategy**（SWE-Smith / SWE-Rebench / 自定义）：
  1. 在 `server/patch_utils.py::detect_eval_strategy` 加新 tag
  2. 实现 `grade_eval_output_<new>()` 函数（参考 rllm `DockerRuntime._calculate_reward_swesmith` / `_swerebench` 两个分支,都是 ~30 行纯 Python）
  3. 可选：在 `server/docker_ops.py` 加 `evaluate_<new>()` 如果 eval 脚本来源不一样
  4. 在 `server/swe_agent_server.py::generate_trajectory` 的 Step 4 加 `elif strategy == "<new>"` 分支
  5. 写对应 preprocess 脚本 `data/preprocess_<new>.py`
- **同时运行多种 reward 机制**：已原生支持（§7 双 eval strategy）。把不同 preprocess 脚本的 parquet concat 起来即可，server 逐行按 `data_source` 分派。

---

## 附：快速校验清单

部署第一次跑之前：

### K8s 基础设施
- [ ] 镜像 `q-codegen:swerl-aws-kiro-0.1.0` 已推到 ECR
- [ ] PVC `private-model-rw` 挂上（`/mnt_out`）
- [ ] K8s secret `openai-api-key`（4B aux judge 用）

### 代码就位
- [ ] `/mnt_out/<USER>/codebase/slime`（git clone）
- [ ] `/mnt_out/<USER>/codebase/slime/examples/aws_kiro_infra`（本项目）
- [ ] `/mnt_out/<USER>/codebase/Megatron-LM`（git clone）

### 数据 / 镜像（§8 数据准备）
- [ ] `PROMPT_DATA` parquet 存在（`python3 -c "import pandas as pd; print(len(pd.read_parquet('...')))"` 应 > 0）
- [ ] 确认 parquet 每行的 `data_source` 字段（分派给对应 eval strategy）
- [ ] **swe_harness 路径**（SWE-Gym / SWE-Bench）: `instance.eval_script` 非空 + `FAIL_TO_PASS` 非空
- [ ] **r2e 路径**（R2E-Gym）: `instance.docker_image` 非空 + `instance.expected_output_json` 非空且是 valid JSON
- [ ] `SWE_DOCKER_IMAGES_PATH` 目录下每个 `<instance_id>.tar.gz` 存在（`ls <dir>/*.tar.gz | wc -l` ≥ parquet 行数）
- [ ] 每个 tarball 不超过 ~10 GB（太大多半 `docker save` 出错）
- [ ] `docker load <tar>` 后 `docker images` 里有**canonical name**（`docker.io/xingyaoww/...` 或 `docker.io/r2egym/...`，不是代理 URL）

### 模型
- [ ] `HF_CKPT` 路径存在且可读（包含 `config.json` + 权重 shards）
- [ ] `REF_LOAD` (30B 必须) 指向 mcore-converted checkpoint（不是 HF）
- [ ] SGLang YAML `MODEL_PATH` = CPU worker YAML `MODEL_PATH` = Trainer YAML `HF_CKPT`（三处**同一路径**）

### 配置协调
- [ ] `SGLANG_RUN_NAME` 在 sglang YAML 和 agent worker YAML 里相同
- [ ] `SWE_AGENT_RUN_NAME`（trainer YAML）= CPU worker YAML 的 `RUN_NAME`
- [ ] `SGLANG_ENV_BASE` 三处一致（sglang 写、agent 写、trainer 读）

### 节点池容量
- [ ] 4B：2 + 4 + 2 = 8 pods（4 × p5.48xlarge + 4 × m5.24xlarge）
- [ ] 30B：8 + 80 + 16 = 104 pods（24 × p5.48xlarge + 80 × m5.24xlarge，Kiro production 规模）
