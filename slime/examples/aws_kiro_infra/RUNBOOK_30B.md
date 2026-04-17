# 30B 训练 — 部署手册（rllm / DeepSWE 配方）

从零到跑起 Qwen3-Coder-30B 的 RL 训练。**对齐 rllm/DeepSWE 论文配方**（`rllm/examples/swe/train_deepswe_32b.sh`）—— GRPO++ + R2E-Gym + sync。

---

## 0. 当前配置一览

### 训练方法（rllm / DeepSWE GRPO++，sync）

| 项 | 值 |
|---|---|
| 模型 | `Qwen/Qwen3-Coder-30B-A3B-Instruct` (HF) + mcore-converted (`--ref-load`) |
| 训练脚本 | `slime/train.py`（**sync**，不用 async pipeline） |
| 算法 | GRPO + **无 std 归一**（≈ LOOP / Dr.GRPO） |
| Clip | **非对称 `0.2 / 0.28`**（DAPO high clip） |
| KL | **完全无**（不传 `--use-kl-loss`） |
| TIS | **无**（sync 不需要 off-policy correction） |
| Entropy | 0 |
| Length norm | **Dr.GRPO strict**（`swe_rl_loss_reducers.dr_grpo_length_norm_reducer`） |
| **Compact filter** | ★ **开**（仅 voluntary submit 的 traj 进梯度,在 `generate_kiro_rllm.generate` 里） |
| Aux LLM judge | **关**（strict 0/1 ORM reward） |
| Oversample | `--over-sampling-batch-size 16` + `check_reward_nonzero_std` |
| Optimizer | `adam-beta2=0.98, lr=1e-6, constant` + cpu-offload |

### 数据集（默认 R2E-Gym-Subset = DeepSWE 官方）

| 项 | 默认值 | 备注 |
|---|---|---|
| `PROMPT_DATA` | `/mnt_out/jinxiaolong/data/r2egym_subset_train.parquet` | **R2E-Gym-Subset train, 4578 instances**（DeepSWE 论文用的训练集，已过滤掉 SWE-Bench-Verified 同 repo）|
| 数据策略 | `r2e`（R2E-Gym 原生 eval） | `data_source="r2e-gym"` → server 自动分派 |
| `SWE_DOCKER_IMAGES_PATH` | `/mnt_out/jinxiaolong/data/r2egym_images/` | R2E-Gym docker tarballs（`namanjain12/*`） |

**为什么是 Subset 不是 V1**：
- **R2E-Gym-Subset**（4578）= DeepSWE 官方**过滤版**，**排除了 sympy 等 SWE-Bench-Verified 重叠 repo**，防止评测污染
- R2E-Gym-V1（8101）= 完整版，有 data leak 风险，**不建议训练用**

DeepSWE blog §2.1: *"Our dataset contains 4.5K problems from a subset of R2E-Gym. To avoid data contamination during training, we filtered out problems that are derived from the same repositories as SWE-Bench-Verified."*

### Rollout 规模（DeepSWE 官方）

| 项 | 值 |
|---|---|
| `rollout-batch-size` | **8**（`data.train_batch_size=8`，DeepSWE 官方脚本） |
| `n-samples-per-prompt` | **8**（`rollout.n=8`，一个 prompt 8 rollouts 做 GRPO） |
| `rollout-max-response-len` | **32768 tokens** |
| `over-sampling-batch-size` | 16（oversample + filter）|
| `SWE_RL_MAX_RESPONSE_LENGTH` | 32768（Dr.GRPO length norm 常量） |
| `--apply-chat-template` | 开 |

### 集群拓扑（Kiro production）

| Pool | 节点型号 | replicas | GPU/pod | 作用 |
|---|---|---|---|---|
| SGLang | `ml.p5.48xlarge` | **8** | 8 H100 | 推理池（8 engines × TP=8 = 64 GPU） |
| swe_agent | `ml.m5.24xlarge` | **80** | 0（CPU） | Docker + mini-swe-agent 循环 |
| Trainer | `ml.p5.48xlarge` | **16** | 8 H100 | slime + Megatron（128 GPU 训练） |

**总计 104 pods，24 GPU 节点 + 80 CPU 节点**。

### 并行切分

```
TP=4, PP=2, CP=8, EP=8 (MoE), ETP=1
→ 64 GPU 组成一份 pipeline，128 GPU 里切 2 份
recompute-granularity=full, recompute-method=uniform, recompute-num-layers=1
max-tokens-per-gpu=32768
log-probs-max-tokens-per-gpu=65536
```

---

## 1. 前置条件

1. **K8s cluster**：AWS HyperPod，namespace `hyperpod-ns-aladdin`，已有 `bedrock-sa` service account 和 Kueue queue
2. **PVC**：
   - `private-model-rw` 挂到 `/mnt_out`（代码、模型、数据、logs）
   - `private-data-rw` 挂到 `/mnt_private`（额外只读数据，可选）
3. **容器镜像**：`975050351917.dkr.ecr.us-east-2.amazonaws.com/q-codegen:swerl-aws-kiro-0.1.0`
   - 预装：slime + Megatron-LM + transformers + docker CLI + fastapi + uvicorn + httpx + loguru + swebench/swegym + litellm

---

## 2. 数据准备（预处理机器上做，然后 rsync 上 PVC）

### 2.1 Prompt parquet

**R2E-Gym-Subset 4578 条**（DeepSWE 官方训练集，已过滤）：

```bash
cd /data_storage/wyj/jxl/slime/examples/aws_kiro_infra

# 依赖（preprocess 不用 r2egym 包,只要 pandas + datasets）
pip install datasets pandas pyarrow

# HF 代理（国内需要）
export HTTP_PROXY=http://100.68.168.184:3128
export HTTPS_PROXY=http://100.68.168.184:3128

# ★ 默认:R2E-Gym-Subset train (4578 条, DeepSWE 官方)
python3 data/preprocess_r2egym.py \
  --output-path /data/r2egym_subset/train.parquet
# 输出 ~200MB, 几分钟

# Smoke test (10 条, streaming 避免下载全集):
python3 data/preprocess_r2egym.py \
  --streaming \
  --max-samples 10 \
  --output-path /tmp/r2e_smoke.parquet
```

Parquet 里 `data_source` 自动填 `"r2e-gym"` → server 的 `detect_eval_strategy` 自动走 r2e 路径（用镜像内置 `/testbed/run_tests.sh` + `expected_output_json` 逐键比对）。

### 2.2 R2E-Gym Docker image tarballs

**必做**。CPU agent pod 用 `--pull never` 跑容器，镜像必须预下载。

R2E-Gym 镜像名在 parquet 的 `instance.docker_image` 字段（形如 `docker.io/namanjain12/<repo>_final:<tag>`）。

```bash
# 代理 rewrite（把 docker.io/ 换成你的 mirror host）
export HTTP_PROXY=http://100.68.168.184:3128
export HTTPS_PROXY=http://100.68.168.184:3128
PROXY_DOCKER_IO=slime-agent-cn-beijing.cr.volces.com \
  bash data/download_swe_images.sh \
    --prompt-data /data/r2egym_subset/train.parquet \
    --output-dir  /data/r2egym_subset/images \
    --parallel 4

# 不用代理
bash data/download_swe_images.sh \
  --prompt-data /data/r2egym_subset/train.parquet \
  --output-dir  /data/r2egym_subset/images \
  --parallel 4
```

**预计**：4578 instances × 平均 2-3 GB = **10-14 TB**，4 并发 **50-80 小时**。`nohup` + 断点续传。

**Smoke（1 条，先验证代理通 + image 能 load）**：
```bash
bash data/download_swe_images.sh \
  --prompt-data /tmp/r2e_smoke.parquet \
  --output-dir  /tmp/r2e_smoke_images \
  --max 1
```

**端到端验证**（已跑通，orange3 reward 0/1 切换正确）：
```bash
bash data/run_r2e_local_test.sh
```

### 2.3 模型 checkpoint

```bash
# HF 格式（SGLang 用 + trainer 初始化用）
pip install huggingface-hub
huggingface-cli download Qwen/Qwen3-Coder-30B-A3B-Instruct \
  --local-dir /data/models/Qwen3-Coder-30B-A3B-Instruct
# ~60 GB

# mcore 格式（trainer 的 --ref-load 用）
cd /data/codebase/slime
bash scripts/convert_hf_to_mcore.sh \
  --hf-checkpoint /data/models/Qwen3-Coder-30B-A3B-Instruct \
  --save-dir      /data/models/Qwen3-Coder-30B-A3B-Instruct-mcore
# 10-20 分钟
```

### 2.4 rsync 到 PVC

```bash
# 目标 PVC 路径（和 YAML 里的默认值对齐）
BASTION=<hyperpod-bastion>
USER=jinxiaolong

# 代码
rsync -avz --exclude __pycache__ --exclude .git \
  /data/codebase/{slime,Megatron-LM} \
  ${BASTION}:/mnt_out/${USER}/codebase/

rsync -avz --exclude __pycache__ \
  /data/codebase/OpenClaw-RL/swe-rl \
  ${BASTION}:/mnt_out/${USER}/codebase/OpenClaw-RL/

# 模型（60 GB HF + 60 GB mcore）
rsync -avz --progress \
  /data/models/Qwen3-Coder-30B-A3B-Instruct{,-mcore} \
  ${BASTION}:/mnt_out/${USER}/models/

# Parquet
rsync -avz /data/swegym_full/train.parquet \
  ${BASTION}:/mnt_out/${USER}/data/swegym_full_train.parquet

# Image tarballs（大! 开 tmux 跑,用 --partial 断点续传）
rsync -avz --progress --partial \
  /data/swegym_full/images/ \
  ${BASTION}:/mnt_out/${USER}/data/swe_images/
```

---

## 3. 配置 YAML（3 个 PyTorchJob）

所有 YAML 的默认 `/mnt_out/jinxiaolong/...` 路径换成你自己 PVC 上的实际路径。

### 3.1 `k8s/launch_sglang_30b.yaml`（GPU SGLang 池）

关键 env（默认值已对齐 Kiro 30B）：

```yaml
env:
- name: MODEL_PATH
  value: "/mnt_out/<你>/models/Qwen3-Coder-30B-A3B-Instruct"
- name: TP_SIZE
  value: "8"                         # 每 pod 一个 engine × TP=8
- name: MEM_FRACTION
  value: "0.80"
- name: CHUNKED_PREFILL_SIZE
  value: "8192"                      # Kiro 30B 默认
- name: RUN_NAME
  value: "swerl_30b_sglang"          # ★ 记这个名,trainer 要用
- name: PET_NNODES
  value: "8"
```

**replicas**: 已设 `8`（产出 8 engines，总 64 GPU）。

### 3.2 `k8s/launch_swe_agent_workers.yaml`（CPU agent 池）

关键 env：

```yaml
env:
- name: MODEL_PATH
  value: "/mnt_out/<你>/models/Qwen3-Coder-30B-A3B-Instruct"    # ★ 和 SGLang 同
- name: SGLANG_RUN_NAME
  value: "swerl_30b_sglang"                                      # ★ 匹配 3.1
- name: SWE_DOCKER_IMAGES_PATH
  value: "/mnt_out/<你>/data/swe_images/"                         # ★ 你上传的 tarball 目录
- name: RUN_NAME
  value: "swerl_30b_agents"                                      # ★ 记这个,trainer 要用
```

**replicas**: 默认 `4`。Kiro 30B 生产用 `80`，你想大规模就改 `replicas: 80`。

### 3.3 `k8s/launch_trainer_30b.yaml`（GPU Trainer 池）

关键 env（已配好 rllm GRPO++ + R2E-Gym + sync）：

```yaml
env:
- name: RUN_NAME
  value: "swerl_30b_rllm"
- name: SGLANG_RUN_NAME
  value: "swerl_30b_sglang"          # ★ 匹配 3.1
- name: SWE_AGENT_RUN_NAME
  value: "swerl_30b_agents"          # ★ 匹配 3.2
- name: SGLANG_ENV_BASE
  value: "/mnt_out/<你>/logs/swerl-aws-kiro"

# 模型
- name: HF_CKPT
  value: "/mnt_out/<你>/models/Qwen3-Coder-30B-A3B-Instruct"
- name: REF_LOAD
  value: "/mnt_out/<你>/models/Qwen3-Coder-30B-A3B-Instruct-mcore"  # mcore 格式

# 数据 (R2E-Gym-Subset = DeepSWE 官方 4578 条过滤版)
- name: PROMPT_DATA
  value: "/mnt_out/<你>/data/r2egym_subset_train.parquet"            # ← parquet 里 data_source=r2e-gym

# rllm GRPO++ rollout 参数
- name: ROLLOUT_BATCH_SIZE
  value: "8"                         # DeepSWE 官方
- name: N_SAMPLES_PER_PROMPT
  value: "8"
- name: OVER_SAMPLING_BATCH_SIZE
  value: "16"
- name: ROLLOUT_MAX_RESPONSE_LEN
  value: "32768"
- name: SWE_RL_MAX_RESPONSE_LENGTH
  value: "32768"                     # Dr.GRPO length norm

# 并行(Kiro 30B production values)
- name: TENSOR_MODEL_PARALLEL_SIZE
  value: "4"
- name: PIPELINE_MODEL_PARALLEL_SIZE
  value: "2"
- name: CONTEXT_PARALLEL_SIZE
  value: "8"
- name: EXPERT_MODEL_PARALLEL_SIZE
  value: "8"
- name: MAX_TOKENS_PER_GPU
  value: "32768"

# SYNC 模式: 不要 UPDATE_WEIGHTS_INTERVAL(async-only)

- name: PET_NNODES
  value: "16"
```

YAML 里的 `args` 调的是 `scripts/run_swe_rl_30b_rllm_trainer.sh`（rllm 变体，而不是老的 `_kiro_trainer.sh`）。

**replicas**: 默认 `16`。

### 3.4 三个 RUN_NAME 必须匹配

```
SGLang YAML    RUN_NAME=swerl_30b_sglang   ──┐
                                             ├→ Agent pool 的 SGLANG_RUN_NAME
                                             │  Trainer 的 SGLANG_RUN_NAME
Agent YAML     RUN_NAME=swerl_30b_agents   ──┤
                                             └→ Trainer 的 SWE_AGENT_RUN_NAME
```

trainer 通过 `SGLANG_ENV_BASE/<RUN_NAME>/` 找各自的 `.env` 协调文件。

---

## 4. 起 3 个 PyTorchJob（顺序无所谓，agent 会等 sglang）

```bash
cd /data_storage/wyj/jxl/slime/examples/aws_kiro_infra/k8s

# Step 1: SGLang (GPU, 8 pods, 加载 30B 模型约 3-5 min)
kubectl apply -f launch_sglang_30b.yaml

# Step 2: CPU agent pool (和 Step 1 并行起)
kubectl apply -f launch_swe_agent_workers.yaml

# Step 3: 等两个 .env 文件就位
ls -la /mnt_out/<你>/logs/swerl-aws-kiro/swerl_30b_sglang/sglang_external_rollout.env
ls -la /mnt_out/<你>/logs/swerl-aws-kiro/swerl_30b_agents/swe_agents.env

# 两个都有了再起 Trainer
kubectl apply -f launch_trainer_30b.yaml
```

### 监控

```bash
# Pod 状态
kubectl get pods -n hyperpod-ns-aladdin | grep -E "swerl-30b|swerl-swe-agent"

# SGLang 启动日志（找 "All services ready!"）
kubectl logs -n hyperpod-ns-aladdin <swerl-30b-sglang-worker-0> -f | grep -E "(ERROR|ready|engine)"

# CPU agent 健康检查（应该看到 docker_ok + tokenizer_loaded）
kubectl exec -n hyperpod-ns-aladdin <swerl-swe-agent-workers-worker-0> -- curl -s http://localhost:5000/healthz

# Trainer 主日志
kubectl logs -n hyperpod-ns-aladdin <swerl-30b-trainer-worker-0> -f
```

### 预期启动时间线

| 时间 | 事件 |
|---|---|
| T+0 | 三个 kubectl apply |
| T+2m | Pod 都 Running（从 ECR pull 镜像完成）|
| T+5m | SGLang 启动完成（load 30B 模型 + Ray cluster）→ 写 `sglang_external_rollout.env` |
| T+5m | CPU agent pod 启动 dockerd + FastAPI → 写 `swe_agents.env` |
| T+6m | Trainer 检测到两个 env 文件，启 Ray cluster |
| T+15m | 第一条 rollout 开始跑（从 PVC load instance tarball + 跑 agent 循环） |
| T+20-30m | 第一个 training step |

---

## 5. 停服（逆序）

```bash
# Trainer 先停
kubectl delete pytorchjob swerl-30b-trainer     -n hyperpod-ns-aladdin

# CPU agent pool
kubectl delete pytorchjob swerl-swe-agent-workers -n hyperpod-ns-aladdin

# SGLang 最后停
kubectl delete pytorchjob swerl-30b-sglang      -n hyperpod-ns-aladdin
```

---

## 6. 本地 smoke test（上 K8s 前验证）

### 6.1 只验证数据 pipeline

```bash
cd /data_storage/wyj/jxl/slime/examples/aws_kiro_infra

# 10 条 parquet + 1 条 tarball
python3 data/preprocess_swegym.py --max-samples 10 \
  --output-path /tmp/smoke.parquet

bash data/download_swe_images.sh \
  --prompt-data /tmp/smoke.parquet \
  --output-dir  /tmp/smoke_images \
  --max 1
```

### 6.2 数据格式检查

```bash
python3 -c "
import pandas as pd
df = pd.read_parquet('/tmp/smoke.parquet')
print('rows:', len(df))
print('data_source:', df.iloc[0]['data_source'])        # 'swe-gym'
inst = df.iloc[0]['instance']
print('instance keys:', sorted(inst.keys()))             # 含 eval_script 等
print('eval_script head:', inst['eval_script'][:100])   # '#!/bin/bash\nset -xo pipefail...'
print('FAIL_TO_PASS count:', len(inst['FAIL_TO_PASS']))
"
```

期望全部非空。

### 6.3 训练 pipeline 冒烟（可选，需 GPU）

如果有本地 GPU，可以先跑 4B 端到端验证 pipeline 再上 30B。30B 在本地跑不起（80 GB 显存不够）。

---

## 7. 常见问题

### Trainer 卡在 "Waiting for sglang_external_rollout.env"

- SGLang 30B 加载慢，可能要 5-8 min
- `kubectl logs` SGLang head pod 看 `launch_external_sglang.py` 是否成功
- 30B 初次加载如果 OOM（`CUDA out of memory`），降 `MEM_FRACTION` 从 `0.80` → `0.70`

### Trainer 卡在 "Waiting for swe_agents.env"

- CPU agent pod 可能在等 SGLang（它们会先等 sglang 再 register）
- 或者 dockerd 没起来 → `kubectl logs` CPU pod 看 `/tmp/dockerd.log`
- `PROMPT_DATA` 读到的 instance 对应 tarball 必须在 `SWE_DOCKER_IMAGES_PATH`，否则第一条 rollout 就 fail

### `ModuleNotFoundError: swegym` / `swebench` 在 agent pod

- CPU pod 镜像需要装 `swegym` 或 `swebench` 包（跑 `get_eval_report`）
- 重新 build 镜像加上 `pip install swegym swebench`

### OOM 在 trainer pod

- 降 `MAX_TOKENS_PER_GPU`（32768 → 16384）
- 或降 `N_SAMPLES_PER_PROMPT`（16 → 8）
- 或升级 `EXPERT_MODEL_PARALLEL_SIZE` / `CONTEXT_PARALLEL_SIZE`

### 训练 loss NaN / `rollout_probs_diff` 大

- tokenizer 不一致：SGLang 的 `MODEL_PATH` 和 Trainer 的 `HF_CKPT` 必须**同一路径**
- mcore 转换有问题：重跑 `convert_hf_to_mcore.sh`

---

## 8. 附：三种配方对比

| 维度 | DeepSWE 官方（`train_deepswe_32b.sh`） | **我们当前**（`run_swe_rl_30b_rllm_trainer.sh`） | Kiro 30B（`run-qwen3-30b-kiro-kos-async-debug-run.sh`） |
|---|---|---|---|
| 模型 | Qwen3-32B | Qwen3-Coder-30B-A3B | Qwen3-Coder-30B-A3B |
| Scaffold | DeepSWE 自定义（4 tools） | **mini-swe-agent（bash backtick）** | Strands + Qwen tool_call |
| Dataset | R2E-Gym-Subset 4578 ✓ | **R2E-Gym-Subset 4578** ✓ | Kiro 内部 sweap 301 |
| Train script | `train_agent_ppo.py`（sync） | **`slime/train.py`（sync）** | `slime/train_async.py` |
| Hybrid engine | 是（rollout+actor 共 GPU） | 否（3-pool 分离） | 否（3-pool 分离） |
| Advantage estimator | `loop` | `grpo` + `grpo-std-normalization False`（≈ loop） | `grpo`（带 std） |
| Clip | 非对称 0.2/0.28 ✓ | **非对称 0.2/0.28** ✓ | 对称 0.2/0.2 |
| KL loss | 无 ✓ | **无** ✓ | 无（coef=0） |
| Length norm | `seq-mean-token-sum` | **Dr.GRPO via reducer** ✓ | `calculate-per-token-loss` |
| TIS | 无（sync）✓ | **无** ✓ | 开（async 补偿） |
| **Compact filter** | `agent.overlong_filter=True` ✓ | **开**（`generate_kiro_rllm` 里） ✓ | 无 |
| Rollout batch × n | 8 × 8 ✓ | **8 × 8** ✓ | 16 × 16 |
| Response len | 32768 ✓ | **32768** ✓ | 73728 |
| Over-sampling | 无（用 rejection_sample） | 16 | 32 |
| Aux LLM judge | 无 ✓ | **无** ✓ | 无 |

**我们现在和 DeepSWE 高度对齐**（scaffold 替换成 mini-swe-agent 是刻意选择的差异）。与 Kiro 30B 的不同：
- 数据集：R2E-Gym（我们/DeepSWE） vs Kiro 内部 sweap
- 训练模式：sync（我们/DeepSWE） vs async（Kiro）
- Clip：非对称（我们/DeepSWE） vs 对称（Kiro）
- Compact filter：有（我们/DeepSWE） vs 无（Kiro）

---

## 9. 快速校验清单

部署第一次跑之前：

- [ ] `kubectl get nodes` 确认有 24 × p5.48xlarge + 80 × m5.24xlarge 可用
- [ ] 镜像 `q-codegen:swerl-aws-kiro-0.1.0` 已在 ECR
- [ ] 模型：`HF_CKPT` 目录含 `config.json` + 权重分片；`REF_LOAD` 是 mcore 格式
- [ ] Parquet：`instance.eval_script` 非空、`FAIL_TO_PASS` 非空
- [ ] Tarball：`SWE_DOCKER_IMAGES_PATH/*.tar.gz` 数量 ≥ parquet rows
- [ ] 三个 YAML 里的 RUN_NAME 互相匹配（SGLang↔Agent↔Trainer）
- [ ] `SGLANG_ENV_BASE` 在三个 YAML 一致
- [ ] `MODEL_PATH` 在 SGLang + Agent YAML 一致，等于 `HF_CKPT`

---

## 相关文档

- `RUNBOOK.md` — 完整运维手册（4B + 30B，所有章节）
- `data/README.md` — Parquet schema 规范
- `DATASET_REWARD_DIAGNOSIS_2026-04-16.md` — R2E-Gym eval 链路诊断
