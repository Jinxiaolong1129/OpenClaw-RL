# Teammate Quickstart — 一页纸上手 30B 训练

> **★ 代码位置**：`/data_storage/wyj/jxl/OpenClaw-RL/slime/examples/aws_kiro_infra/`（按 slime examples 惯例）
>
> **PVC 部署路径**：`/mnt_out/<USER>/codebase/slime/examples/aws_kiro_infra/`


从零到跑起 Qwen3-Coder-30B 的 RL 训练。**训练方法 = rllm / DeepSWE GRPO++ (sync)，数据集 = R2E-Gym-Subset (4578 条 DeepSWE 官方过滤版)**。

完整背景见 `RUNBOOK_30B.md`，这里只给**操作命令**。

---

## Step 0：三件前置

1. **K8s 集群**：HyperPod，namespace `hyperpod-ns-aladdin`，已配 `bedrock-sa` / Kueue queue
2. **PVC**：`private-model-rw` 挂 `/mnt_out`
3. **容器镜像**：`975050351917.dkr.ecr.us-east-2.amazonaws.com/q-codegen:swerl-aws-kiro-0.1.0`

---

## Step 1：预处理数据（预处理机器上，一次性）

```bash
cd /data_storage/wyj/jxl/slime/examples/aws_kiro_infra

# ---- 环境 ----
pip install datasets pandas pyarrow huggingface-hub
export HTTP_PROXY=http://100.68.168.184:3128
export HTTPS_PROXY=http://100.68.168.184:3128

# ---- 1a. Prompt parquet (R2E-Gym-Subset 4578 条,~200 MB) ----
python3 data/preprocess_r2egym.py \
  --output-path /data/r2egym_subset/train.parquet
# 验证
python3 -c "
import pandas as pd
df = pd.read_parquet('/data/r2egym_subset/train.parquet')
print('rows:', len(df))
print('data_source:', df.iloc[0]['data_source'])       # 'r2e-gym'
print('docker_image:', df.iloc[0]['instance']['docker_image'])
"

# ---- 1b. Docker image tarballs (★ 大,~10-14 TB,挂 nohup) ----
PROXY_DOCKER_IO=slime-agent-cn-beijing.cr.volces.com \
nohup bash data/download_swe_images.sh \
    --prompt-data /data/r2egym_subset/train.parquet \
    --output-dir  /data/r2egym_subset/images \
    --parallel 4 \
    > /data/r2egym_subset/download.log 2>&1 &
# 进度: watch -n 30 'ls /data/r2egym_subset/images/*.tar.gz | wc -l; du -sh /data/r2egym_subset/'

# ---- 1c. 30B 模型 (HF + mcore) ----
huggingface-cli download Qwen/Qwen3-Coder-30B-A3B-Instruct \
  --local-dir /data/models/Qwen3-Coder-30B-A3B-Instruct
# mcore 转换
cd /data/codebase/slime
bash scripts/convert_hf_to_mcore.sh \
  --hf-checkpoint /data/models/Qwen3-Coder-30B-A3B-Instruct \
  --save-dir      /data/models/Qwen3-Coder-30B-A3B-Instruct-mcore
```

---

## Step 2：本机 smoke test（上 K8s 前强烈建议跑）

验证整条 reward 链路能跑通（已在 orange3 上本地验证过）：

```bash
cd /data_storage/wyj/jxl/slime/examples/aws_kiro_infra

export HTTP_PROXY=http://100.68.168.184:3128
export HTTPS_PROXY=http://100.68.168.184:3128

# 一键 3 合 1: preprocess (10 条) + download 1 个 tarball + reward eval
PROXY_DOCKER_IO=slime-agent-cn-beijing.cr.volces.com \
  bash data/run_r2e_local_test.sh

# 或者用我们已经跑通的验证脚本(已保留 orange3 tarball)
bash test.sh
```

**期望输出**：
```
Scenario A (no patch):   resolved=False (reward=0)   ← 正确
Scenario B (gold patch): resolved=True  (reward=1)   ← 正确
```

---

## Step 3：rsync 到 PVC

```bash
BASTION=<hyperpod-bastion-host>
USER=jinxiaolong

# 代码
rsync -avz --exclude __pycache__ --exclude .git \
  /data/codebase/{slime,Megatron-LM} \
  ${BASTION}:/mnt_out/${USER}/codebase/
rsync -avz --exclude __pycache__ \
  /data/codebase/OpenClaw-RL/swe-rl \
  ${BASTION}:/mnt_out/${USER}/codebase/OpenClaw-RL/

# 模型 (HF ~60GB + mcore ~60GB)
rsync -avz --progress /data/models/ \
  ${BASTION}:/mnt_out/${USER}/models/

# Parquet
rsync -avz /data/r2egym_subset/train.parquet \
  ${BASTION}:/mnt_out/${USER}/data/r2egym_subset_train.parquet

# Tarballs (大! tmux 里跑,断了 --partial 续传)
rsync -avz --progress --partial \
  /data/r2egym_subset/images/ \
  ${BASTION}:/mnt_out/${USER}/data/r2egym_images/
```

---

## Step 4：检查 & 改 K8s YAML（3 个）

三个 YAML 里的 `/mnt_out/jinxiaolong/...` 路径改成你自己 PVC 路径。关键点：

### `k8s/launch_sglang_30b.yaml`
```yaml
- name: MODEL_PATH
  value: "/mnt_out/<你>/models/Qwen3-Coder-30B-A3B-Instruct"   # HF
- name: RUN_NAME
  value: "swerl_30b_sglang"        # ★ 记这个
- name: PET_NNODES
  value: "8"
```
默认 `replicas: 8`。

### `k8s/launch_swe_agent_workers.yaml`
```yaml
- name: MODEL_PATH
  value: "/mnt_out/<你>/models/Qwen3-Coder-30B-A3B-Instruct"   # 和 SGLang 同
- name: SGLANG_RUN_NAME
  value: "swerl_30b_sglang"        # ★ 匹配 SGLang YAML
- name: SWE_DOCKER_IMAGES_PATH
  value: "/mnt_out/<你>/data/r2egym_images/"
- name: RUN_NAME
  value: "swerl_30b_agents"        # ★ 记这个
```
CPU pool 规模看你想多高并发，生产可以 80 replicas。

### `k8s/launch_trainer_30b.yaml`
```yaml
- name: RUN_NAME
  value: "swerl_30b_rllm"
- name: SGLANG_RUN_NAME
  value: "swerl_30b_sglang"        # ★ 匹配 SGLang YAML
- name: SWE_AGENT_RUN_NAME
  value: "swerl_30b_agents"        # ★ 匹配 agent YAML
- name: HF_CKPT
  value: "/mnt_out/<你>/models/Qwen3-Coder-30B-A3B-Instruct"   # HF
- name: REF_LOAD
  value: "/mnt_out/<你>/models/Qwen3-Coder-30B-A3B-Instruct-mcore"  # ★ mcore!
- name: PROMPT_DATA
  value: "/mnt_out/<你>/data/r2egym_subset_train.parquet"
```
YAML 里的 `args` 已调用 `scripts/run_swe_rl_30b_rllm_trainer.sh`（rllm 变体）。
默认 `replicas: 16`（Kiro prod 规模）。

---

## Step 5：kubectl apply（按顺序）

```bash
cd /data_storage/wyj/jxl/slime/examples/aws_kiro_infra/k8s

# 1. GPU SGLang 池 (8 pods, 加载 30B 模型 ~5 min)
kubectl apply -f launch_sglang_30b.yaml

# 2. CPU agent 池 (会等 SGLang env file)
kubectl apply -f launch_swe_agent_workers.yaml

# 3. 等两个 .env 文件(~5-10 min)
watch -n 10 'ls /mnt_out/<你>/logs/swerl-aws-kiro/swerl_30b_sglang/sglang_external_rollout.env 2>/dev/null && \
  ls /mnt_out/<你>/logs/swerl-aws-kiro/swerl_30b_agents/swe_agents.env 2>/dev/null'
# 两个都有了,Ctrl-C 退 watch

# 4. 起 Trainer
kubectl apply -f launch_trainer_30b.yaml

# 5. 跟 log
kubectl logs -n hyperpod-ns-aladdin $(kubectl get pods -n hyperpod-ns-aladdin -o name | grep swerl-30b-trainer-worker-0) -f
```

---

## Step 6：监控

```bash
# Pod 状态
kubectl get pods -n hyperpod-ns-aladdin | grep -E "swerl-30b|swerl-swe-agent"

# CPU agent 健康
kubectl exec -n hyperpod-ns-aladdin <agent-pod> -- curl -s http://localhost:5000/healthz
# 期望: {"ok":true, "docker_ok":true, "tokenizer_loaded":true, ...}

# SGLang 健康
kubectl exec -n hyperpod-ns-aladdin <sglang-head-pod> -- \
  curl -s http://localhost:30000/health
```

预期时间线：
- T+0      kubectl apply 三个 YAML
- T+2m    所有 pod Running（ECR pull 完成）
- T+5m    SGLang 启动完 → 写 sglang_external_rollout.env
- T+5m    Agent pod 起 dockerd + FastAPI → 写 swe_agents.env
- T+6m    Trainer 检测到两个 env → 起 Ray
- T+15m   第一条 rollout 开始
- T+25m   第一个 training step

---

## Step 7：停服（逆序）

```bash
kubectl delete pytorchjob swerl-30b-trainer      -n hyperpod-ns-aladdin
kubectl delete pytorchjob swerl-swe-agent-workers -n hyperpod-ns-aladdin
kubectl delete pytorchjob swerl-30b-sglang       -n hyperpod-ns-aladdin
```

---

## Troubleshooting 速查

| 症状 | 原因 | 修复 |
|---|---|---|
| Trainer 卡在 `Waiting for sglang_external_rollout.env` | SGLang 还在加载 30B | 等 5-8 min；超时看 `kubectl logs <sglang-pod>` |
| Trainer 卡在 `Waiting for swe_agents.env` | CPU agent 没起好 | 看 `kubectl logs <agent-pod>`；dockerd fail 最常见 |
| rollout 全 reward=0 | R2E image 缺 `run_tests.sh` | 已修复（见 `DATASET_REWARD_DIAGNOSIS_2026-04-16.md`）|
| `docker run: No such image` | tarball 没到 PVC | `ls /mnt_out/<你>/data/r2egym_images/` 确认 |
| Training loss NaN / 很大 | tokenizer drift | SGLang `MODEL_PATH` 和 Trainer `HF_CKPT` 必须同路径 |
| SGLang OOM | 30B 显存占太多 | 降 `MEM_FRACTION` 从 0.80 → 0.70 |

---

## 速查文档索引

- **30B 训练细节** → `RUNBOOK_30B.md`（配置、算法对比、部署）
- **4B 训练**（如果要做小规模对照）→ `RUNBOOK.md` §部署 4B
- **R2E-Gym reward 机制** → `DATASET_REWARD_DIAGNOSIS_2026-04-16.md`
- **数据格式规范** → `data/README.md`
- **完整架构 + 代码结构** → `RUNBOOK.md`

---

## 关键文件速览

```
aws_kiro_infra/
├── k8s/
│   ├── launch_sglang_30b.yaml              # 8 × p5 SGLang
│   ├── launch_swe_agent_workers.yaml       # 4-80 × m5 CPU agent
│   └── launch_trainer_30b.yaml             # 16 × p5 trainer
├── scripts/
│   └── run_swe_rl_30b_rllm_trainer.sh      # ★ 默认的 30B 脚本 (rllm/DeepSWE)
├── generate_kiro_rllm.py                   # ★ compact filter 实现
├── data/
│   ├── preprocess_r2egym.py                # R2E-Gym HF → parquet
│   ├── download_swe_images.sh              # parquet → docker tarballs
│   └── run_r2e_local_test.sh               # 本地 smoke test
└── RUNBOOK_30B.md                          # 详细文档
```
