# SWE-RL — 系统架构详解

基于 [OpenClaw-RL](../README.md) 框架，使用强化学习训练 LLM 自动修复真实 GitHub Issue（SWE-Bench / SWE-Gym）。

---

## 目录

- [整体架构](#整体架构)
  - [逻辑架构视图](#逻辑架构视图)
  - [物理部署视图](#物理部署视图)
- [物理节点分配与 Placement Group 机制](#物理节点分配与-placement-group-机制)
  - [Placement Group 分配过程](#placement-group-分配过程)
  - [Megatron TP 组的物理映射](#megatron-tp-组的物理映射)
  - [Agent Scaffold 在哪里运行](#agent-scaffold-在哪里运行)
  - [节点间通信矩阵](#节点间通信矩阵)
- [组件详解](#组件详解)
  - [训练层 — Slime + Megatron](#训练层--slime--megatron)
  - [推理层 — SGLang Router](#推理层--sglang-router)
  - [Rollout 层 — generate_with_swe_remote](#rollout-层--generate_with_swe_remote)
  - [环境层 — Docker 服务器集群](#环境层--docker-服务器集群)
- [数据流（单次 rollout）](#数据流单次-rollout)
- [Agent 执行循环](#agent-执行循环)
- [Context 管理](#context-管理)
- [PRM（过程奖励模型）](#prm过程奖励模型)
- [目录结构](#目录结构)
- [关键超参数](#关键超参数)
- [快速开始](#快速开始)
- [迁移注意事项（迁移到云平台 / K8s 环境）](#迁移注意事项迁移到云平台--k8s-环境)

---

## 整体架构

### 逻辑架构视图

各组件的职责与交互关系（不区分物理节点）：

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                          OpenClaw-RL / SWE-RL                               ║
║                                                                              ║
║  ┌─────────────────────────────────────────────────────────────────────┐    ║
║  │                         GPU 节点集群                                 │    ║
║  │                                                                     │    ║
║  │  ┌─────────────────┐    ┌──────────────────┐    ┌───────────────┐  │    ║
║  │  │  Megatron Actor │    │  SGLang Router   │    │  Pool Server  │  │    ║
║  │  │  (GRPO训练)     │    │  (:30000/v1)     │    │  (:18090)     │  │    ║
║  │  │  TP=8 per node  │    │  负载均衡推理请求  │    │  租约调度     │  │    ║
║  │  └────────┬────────┘    └────────┬─────────┘    └───────┬───────┘  │    ║
║  │     权重同步(Ray)        LiteLLM acompletion      HTTP           │    ║
║  │           ▼                      ▼                       │          │    ║
║  │  ┌─────────────────────────────────────────┐             │          │    ║
║  │  │  RolloutManager (num_gpus=0, CPU only)  │             │          │    ║
║  │  │   generate_with_swe_remote.py           │◄────────────┘          │    ║
║  │  │   (多协程并发，SWE_MAX_CONCURRENT=128)   │                        │    ║
║  │  └─────────────────────────────────────────┘                        │    ║
║  └─────────────────────────────────────────────────────────────────────┘    ║
║                                     │                                        ║
║                              HTTP (:5000)                                    ║
║                                     │                                        ║
║  ┌──────────────┐  ┌──────────────┐ │ ┌──────────────┐  ┌──────────────┐   ║
║  │ ECS Node A   │  │ ECS Node B   │ │ │ ECS Node N   │  │ ...          │   ║
║  │ exec_server  │  │ exec_server  │◄┘ │ exec_server  │  │              │   ║
║  │ ┌──────────┐ │  │ ┌──────────┐ │  │ ┌──────────┐ │  │              │   ║
║  │ │Container │ │  │ │Container │ │  │ │Container │ │  │              │   ║
║  │ │(SWE img) │ │  │ │(SWE img) │ │  │ │(SWE img) │ │  │              │   ║
║  │ └──────────┘ │  │ └──────────┘ │  │ └──────────┘ │  │              │   ║
║  │  最多15容器   │  │  最多15容器   │  │  最多15容器   │  │              │   ║
║  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘   ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

### 物理部署视图

系统由三个物理层组成，通过 HTTP 连接：

```
╔══════════════════════════════════════════════════════════════════════════════════╗
║                     GPU 集群（8 物理节点）                                        ║
║                                                                                  ║
║  ┌─────────────────────────────────────┐  ┌──────────────────────────────────┐  ║
║  │   训练区  Node 0-3（专用 Megatron）   │  │  推理区  Node 4-7（专用 SGLang）  │  ║
║  │                                     │  │                                  │  ║
║  │  Node 0 ── GPU 0-7: Megatron Shard0 │  │  Node 4 ── GPU 0-7: Engine 0,1  │  ║
║  │  Node 1 ── GPU 0-7: Megatron Shard1 │  │  Node 5 ── GPU 0-7: Engine 2,3  │  ║
║  │  Node 2 ── GPU 0-7: Megatron Shard2 │  │  Node 6 ── GPU 0-7: Engine 4,5  │  ║
║  │  Node 3 ── GPU 0-7: Megatron Shard3 │  │  Node 7 ── GPU 0-7: Engine 6,7  │  ║
║  │                    ↑ NCCL/NVLink    │  │              ↑ TP=4 per engine   │  ║
║  │                                     │  │              ↑ SGLang Router     │  ║
║  │  Node 0 CPU:                        │  │                (:30000)          │  ║
║  │  ┌──────────────────────────────┐   │  └──────────────────────────────────┘  ║
║  │  │ Ray Head / RolloutManager    │   │              ▲ HTTP :30000              ║
║  │  │ (Agent Scaffold, num_gpus=0) │───┼──────────────┘                         ║
║  │  │ Pool Server (:18090)         │   │                                         ║
║  │  └──────────────────────────────┘   │                                         ║
║  └─────────────────────────────────────┘                                         ║
╚══════════════════════════════════════════════════════════════════════════════════╝
                              │ HTTP :5000
    ┌─────────────────────────┼────────────────────────────────┐
    ▼                         ▼                                ▼
  ECS Node A              ECS Node B              ...    ECS Node R
  exec_server             exec_server                    exec_server
  [容器×15]               [容器×15]                      [容器×15]
  (CPU-only, 无 GPU)      (CPU-only, 无 GPU)             (CPU-only, 无 GPU)
```

> **关键设计**：Megatron（训练）和 SGLang（推理）分布在**不同的物理节点**上，通过 Ray Object Store 同步权重；Agent Scaffold 是**零 GPU** 的纯 CPU 异步协程，运行在 Ray 调度到的 CPU 节点上（8 节点脚本中通常为 head 节点）；ECS Docker 节点完全独立，无 GPU。

---

## 物理节点分配与 Placement Group 机制

### Placement Group 分配过程

Slime 使用**单一 Ray Placement Group** 管理所有 GPU，通过排序后的 bundle 索引划分角色：

```
# slime/slime/ray/placement_group.py

# 1. 创建 64 个 bundle（每个 = 1 GPU + 1 CPU），策略 PACK
bundles = [{"GPU": 1, "CPU": 1} for _ in range(64)]
pg = placement_group(bundles, strategy="PACK")
# PACK → Ray 从第一个节点开始填满后再填下一个节点
# 结果: Node0 填满8个, Node1 填满8个, ..., Node7 填满8个

# 2. 用 InfoActor 探测每个 bundle 的实际 (node_ip, gpu_id)
# 3. 按 sort_key = (node_ip_parts, gpu_id) 升序排列所有 bundle
# 排列结果（节点优先，GPU次之）:
#   sorted[0]  = (Node0, GPU0)
#   sorted[1]  = (Node0, GPU1)
#   ...
#   sorted[7]  = (Node0, GPU7)
#   sorted[8]  = (Node1, GPU0)
#   ...
#   sorted[31] = (Node3, GPU7)   ← 最后一个 Actor GPU
#   sorted[32] = (Node4, GPU0)   ← 第一个 Rollout GPU
#   ...
#   sorted[63] = (Node7, GPU7)

# 4. 按 rollout_offset 切分
rollout_offset = actor_num_nodes × actor_num_gpus_per_node = 8 × 4 = 32

Actor  (Megatron) → sorted[0  .. 31] = Node 0-3 的全部 GPU（每节点 8 块）
Rollout (SGLang)  → sorted[32 .. 63] = Node 4-7 的全部 GPU（每节点 8 块）
```

**结论：Megatron 独占 Node 0-3，SGLang 独占 Node 4-7，两者物理隔离。**

> `ACTOR_GPUS_PER_NODE=4` 的含义是 Megatron 内部的逻辑拓扑（8 个"逻辑节点" × 4 GPU），并非每物理节点分4块 GPU。实际上每个物理训练节点的全部 8 块 GPU 都归 Megatron 使用。

---

### Megatron TP 组的物理映射

```
Actor GPUs 按排序后的逻辑 rank 分配（TP=8, PP=1）:

  逻辑 rank:  0   1   2   3   4   5   6   7  | 8   9  10  11  12  13  14  15  | ...
  物理节点:  ←────── Node 0: GPU 0-7 ────────→ | ←────── Node 1: GPU 0-7 ────────→
  TP Group:  ←──────── TP Group 0 ────────────→ | ←──────── TP Group 1 ─────────────→

  共 4 个 TP Group（各对应一个物理节点），每个 TP Group 的 8 块 GPU
  通过节点内 NVLink 高速通信，不跨节点 → 最优 all-reduce 带宽。

SGLang Engines（rollout_num_gpus_per_engine=4）:
  Engine 0 → Node 4: GPU 0-3  (TP=4)
  Engine 1 → Node 4: GPU 4-7  (TP=4)
  Engine 2 → Node 5: GPU 0-3
  ...
  Engine 7 → Node 7: GPU 4-7
  共 8 个引擎，统一由 SGLang Router (:30000) 负载均衡。
```

---

### Agent Scaffold 在哪里运行

```python
# slime/slime/ray/placement_group.py
rollout_manager = RolloutManager.options(
    num_cpus=1,
    num_gpus=0,   # ← 零 GPU，纯 CPU Ray Actor
).remote(args, pg, prm_pg)
```

`RolloutManager` 是一个 **CPU-only Ray Actor**，由 Ray 调度到可用 CPU 节点（8 节点脚本中通常为 head 节点）。
`generate_with_swe_remote.generate()` 作为异步协程运行在 `RolloutManager` 内部，完全不占用 GPU。

**Scaffold 的工作全是 I/O 和 Python 控制流：**

```
RolloutManager (Ray 调度到的 CPU 节点)
  └── generate_with_swe_remote.generate()   ← asyncio 协程，零 GPU
        │
        ├── await acompletion(...)
        │     └── HTTP → SGLang Router (:30000) → Node 4-7 GPU 推理
        │
        └── await env_client.exec(...)
              └── HTTP → Pool Server (:18090, 通常在 head CPU)
                    └── HTTP → Exec Server (:5000, ECS 节点)
                          └── docker exec → 容器内 bash
```

---

### 节点间通信矩阵

```
┌──────────────────────┬────────────────────────┬────────────────────────────────────┐
│ 发送方               │ 接收方                  │ 协议 / 用途                         │
├──────────────────────┼────────────────────────┼────────────────────────────────────┤
│ Megatron Worker      │ Megatron Worker        │ NCCL + NVLink（节点内）             │
│ Node 0-3             │ Node 0-3               │ TP all-reduce，在同一物理节点内完成  │
├──────────────────────┼────────────────────────┼────────────────────────────────────┤
│ Megatron Actor       │ SGLang Engines         │ Ray Object Store（跨节点）          │
│ Node 0-3             │ Node 4-7               │ update_weights() 触发，权重广播     │
├──────────────────────┼────────────────────────┼────────────────────────────────────┤
│ RolloutManager       │ SGLang Router          │ HTTP :30000                        │
│ CPU 节点（Ray 调度） │ Node 4-7 GPU           │ LiteLLM acompletion，每步 LLM 推理  │
├──────────────────────┼────────────────────────┼────────────────────────────────────┤
│ RolloutManager       │ Pool Server            │ HTTP :18090                        │
│ CPU 节点（Ray 调度） │ head CPU               │ allocate / exec / evaluate / close  │
├──────────────────────┼────────────────────────┼────────────────────────────────────┤
│ Pool Server          │ Exec Servers           │ HTTP :5000                         │
│ Node 0 CPU           │ ECS 节点 CPU           │ Docker 容器创建、命令执行、评估      │
├──────────────────────┼────────────────────────┼────────────────────────────────────┤
│ RolloutManager       │ Megatron Actor         │ Ray RPC + Object Store             │
│ Node 0 CPU           │ Node 0-3 GPU           │ async_train(data_ref) 传输训练数据  │
└──────────────────────┴────────────────────────┴────────────────────────────────────┘
```

---

## 组件详解

### 训练层 — Slime + Megatron

```
slime/train_async.py
│
├── create_placement_groups(args)            ← 单一 PACK PG，64 个 bundle
│   ├── Actor  PG → sorted[0..31]  → Node 0-3 全部 GPU（32 块）
│   └── Rollout PG → sorted[32..63] → Node 4-7 全部 GPU（32 块）
│
├── create_rollout_manager(args, pg)         ← num_gpus=0，CPU-only Ray Actor
│   ├── 启动 SGLang Router（:30000）
│   └── 初始化 8 个 SGLang Engine（各 TP=4，4 GPU）
│
├── create_training_models(args, pgs, ...)
│   ├── Actor Model  (Megatron, TP=8, 4 个 TP Group 各在单节点)
│   └── Ref Model    (同 Actor，用于 KL 散度计算)
│
└── 主训练循环 (async)
    ├── rollout_manager.generate.remote(rollout_id)    ← 异步采样（不阻塞训练）
    │     └── 调用 generate_with_swe_remote.generate()
    ├── actor_model.async_train(rollout_id, data_ref)  ← 异步训练
    │     └── GRPO loss + KL loss + entropy loss
    └── actor_model.update_weights()                   ← 权重同步到 SGLang Engine
```

**GRPO 训练目标：**

```
loss = -E[clip(ratio, 1-ε, 1+ε) × A] + β × KL(π_θ ∥ π_ref)

  A (advantage)  = (r - mean(r_group)) / std(r_group)  # 同组 n=8 条轨迹内归一化
  r (reward)     = outcome_reward  [+ prm_step_coef × mean(step_scores)]
  ε              = 0.2 (clip 下界), 0.28 (clip 上界)
  β              = 0.00 (KL coef，默认关闭)
```

---

### 推理层 — SGLang Router

```
SGLang Router (:30000)                         ← 运行在 RolloutManager 所在 CPU 节点
│
├── 接收 LiteLLM acompletion() 请求
├── 负载均衡到 8 个 SGLang Engine
│   ├── Engine 0 (Node 4: GPU 0-3, TP=4)
│   ├── Engine 1 (Node 4: GPU 4-7, TP=4)
│   ├── Engine 2 (Node 5: GPU 0-3, TP=4)
│   ├── Engine 3 (Node 5: GPU 4-7, TP=4)
│   └── ... Engine 4-7 (Node 6-7 同上)
│
└── 返回推理结果 (assistant_text)
```

每步推理调用（在 Agent Scaffold 内异步发起）：
```python
resp = await acompletion(
    model="openai/Qwen/Qwen3-32B",
    messages=ctx_messages,   # 经 context_manager 截断后的消息
    temperature=1.0,
    max_tokens=4096,
)
```

---

### Rollout 层 — generate_with_swe_remote

这是 SWE-RL 的核心，实现了 Slime 定义的两个接口：

```
generate_with_swe_remote.py  （运行在 RolloutManager CPU Actor 内）
│
├── generate(args, sample, sampling_params)
│   ├── [超时守护] asyncio.wait_for(..., timeout=1800s)
│   ├── Step 1: 初始化 PRM Agent（若启用）
│   ├── Step 2: 通过 SweEnvClient 分配 Docker 容器
│   ├── Step 3: 调用 _run_agent_remote() 执行多轮交互
│   ├── Step 4: 分配新评估容器，打 patch + 跑测试
│   └── Step 5: 构建 Sample(tokens, loss_mask, reward) 返回
│
│   输出模式 A — dynamic_history=True（推荐）:
│     每步生成一个 Sample，使用该步骤实际截断上下文
│     返回 List[Sample]，长度 = 实际执行步数（≤20）
│
│   输出模式 B — dynamic_history=False:
│     全程对话打包为一个 Sample
│     返回 Sample
│
└── reward_func(args, sample)
    ├── 若 PRM 未启用: reward = outcome_reward (+1/-1)
    └── 若 PRM 启用:   reward = outcome_reward + prm_step_coef × mean(step_scores)
```

---

### 环境层 — Docker 服务器集群

两个组件的**启动时机和所在节点完全不同**：

| 组件 | 跑在哪 | 何时启动 | GPU 节点上跑吗 |
|------|--------|---------|--------------|
| `swe_env_pool_server.py` | GPU 头节点（Node 0）CPU | 每次训练时由训练脚本自动启动 | ✅ 是（但只用 CPU） |
| `swe_exec_server.py` | ECS Docker 节点（独立机器） | **提前一次性部署**，注册为 systemd 服务，常驻运行 | ❌ 否，GPU 节点上从不运行 |

**部署顺序：**

```
第一步（一次性，每台 ECS 节点）:
  scp server/swe_exec_server.py  → ECS Node A
  scp server/setup_ecs_seed.sh   → ECS Node A
  ssh ECS Node A: bash setup_ecs_seed.sh
  # 安装 Docker + 注册 systemd 服务 + 预拉 SWE-Bench 镜像（需 2-5 小时）
  # 完成后 swe_exec_server.py 作为常驻服务开机自启

第二步（每次训练，GPU 头节点）:
  bash scripts/run_swe_rl_32b_remote_8nodes.sh
  # 脚本自动启动 swe_env_pool_server.py（连接已在线的 exec server 们）
  # GPU 节点上不运行任何 Docker 相关代码
```

**调用链（`swe_exec_server.py` 调的是它自己所在机器的本地 Docker）：**

```
SweEnvClient（GPU Node 0, CPU）
    │ HTTP :18090
    ▼
swe_env_pool_server.py（GPU Node 0, CPU）  ← 训练时自动启动
    │ HTTP :5000（跨机器）
    ├──▶ swe_exec_server.py（ECS Node A）  ← 预先部署，常驻
    │        └── subprocess: docker run/exec/rm（本机 Docker daemon）
    ├──▶ swe_exec_server.py（ECS Node B）
    └──▶ swe_exec_server.py（ECS Node R）

swe_exec_server.py 的 HTTP 接口:
    ├── POST /container/create    → docker run -d <swebench_image>
    ├── POST /container/exec      → docker exec <cmd> (cwd=/testbed)
    ├── POST /container/diff      → docker exec git diff
    ├── POST /container/evaluate  → git apply <patch> + bash eval_script
    └── POST /container/destroy   → docker rm -f <container>

每台 ECS 节点最多 SWE_MAX_CONTAINERS_PER_NODE=15 个并发容器
```

---

## 数据流（单次 rollout）

```
JSONL 数据集 (SWE-Bench / SWE-Gym)
  {"text": "<problem>", "metadata": {"instance": {...}, "data_source": "..."}}
         │
         ▼
Slime DataLoader → Sample(prompt, metadata)
         │
         ▼ [Node 0 CPU — RolloutManager 内异步执行]
┌────────────────────────────────────────────────────────────┐
│                   generate() 函数                           │
│                                                            │
│  1. 获取信号量 (SWE_MAX_CONCURRENT=128 并发上限)            │
│                                                            │
│  2. SweEnvClient.allocate(image, instance_id)             │
│     → Pool Server (:18090) → Exec Server (:5000)          │
│     → docker run -d <swebench_image> → lease_id           │
│                                                            │
│  3. _run_agent_remote() 多轮交互（最多 20 步）:             │
│     ┌─────────────────────────────────────────────┐       │
│     │ for step in range(20):                      │       │
│     │   ctx = context_manager.truncate(messages)  │  ← head+tail 截断（CPU）
│     │   resp = LiteLLM(ctx)                       │  ← HTTP → SGLang (Node4-7 GPU)
│     │   cmd  = parse_bash(resp.content)           │  ← 提取 ```bash``` 块（CPU）
│     │   out  = env_client.exec(cmd)               │  ← HTTP → ECS Docker
│     │   if PRM: submit_step_judge(step)           │  ← 异步 HTTP → PRM LLM
│     │   if is_submit: break                       │
│     │   messages.append(observation)              │
│     └─────────────────────────────────────────────┘       │
│                                                            │
│  4. 如有 patch:                                            │
│     env_client.close(lease_id)        ← 关闭交互容器       │
│     eval_lease = allocate(eval容器)    ← 新建干净容器       │
│     result = evaluate(patch, script)  ← git apply + 测试  │
│     reward = +1 if resolved else -1                       │
│                                                            │
│  5. 构建 Sample(s):                                        │
│     tokens    = tokenize(ctx_messages + response)         │
│     loss_mask = 1 for assistant tokens, 0 for others      │
│     reward    = outcome_reward + PRM_coef × step_mean     │
│                                                            │
└────────────────────────────────────────────────────────────┘
         │
         ▼
Sample / List[Sample]  (dynamic_history 模式: 每步一个)
         │
         ▼
reward_func() → 最终 reward dict
         │
         ▼  [Ray Object Store 传输到 Node 0-3]
Megatron Actor (Node 0-3 GPU) → GRPO 梯度更新
         │
         ▼
actor.update_weights() → Ray Object Store → SGLang Engines (Node 4-7 GPU)
```

---

## Agent 执行循环

```
初始对话:
  messages = [
    {"role": "system", "content": <system_template from swebench.yaml>},
    {"role": "user",   "content": <problem_statement>},
  ]

Step 0..19:
  ┌──────────────────────────────────────────────────────────────┐
  │                                                              │
  │  [context 截断]                                              │
  │   ctx = get_context_messages(messages, tokenizer,           │
  │             max_input_tokens, head_ratio=0.3)               │
  │   → head(30%) + tail(70%) 策略                              │
  │                                                              │
  │  [LLM 生成]  → HTTP → SGLang Router → SGLang Engine (GPU)   │
  │   assistant_text 包含:                                       │
  │   <think>...</think>  (思考过程，不训练)                      │
  │   ```bash\n<命令>\n```  (动作)                               │
  │                                                              │
  │  [解析命令]  parse_bash(assistant_text) → bash_cmd           │
  │   若解析失败 → observation = "No valid bash command found"   │
  │                                                              │
  │  [执行命令]  → HTTP → Pool Server → Exec Server → Docker     │
  │   exec_result = {returncode: 0/-1, output: "..."}           │
  │                                                              │
  │  [PRM 评分（若启用）]                                         │
  │   submit_step_judge(step_debug) → 异步 HTTP → judge LLM     │
  │   m-vote → \boxed{±1}（不阻塞主循环）                        │
  │                                                              │
  │  [提交判断]                                                   │
  │   若 bash_cmd 含 "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT":   │
  │     → 提取 patch → 退出循环                                   │
  │   否则:                                                       │
  │     obs = render_template(returncode, output)               │
  │     messages.append({"role": "user", "content": obs})       │
  │                                                              │
  └──────────────────────────────────────────────────────────────┘

若 20 步内未提交: git diff fallback → patch（可能为空）

评估:
  新容器 → git apply <patch> → bash eval_script → resolved?
  reward = +1 (resolved) / -1 (not resolved)
```

---

## Context 管理

长对话（超过 `rollout_max_context_len`）由 `swe_context_manager.py` 管理：

```
rollout_max_context_len = 16384 tokens
max_new_tokens          =  4096 tokens
max_input_tokens        = 12288 tokens (自动计算 = 16384 - 4096)
head_budget             ≈  3686 tokens (head_ratio=0.3, 30%)
tail_budget             ≈  8602 tokens (70%)

完整对话历史:
  [system]  [problem]  [turn_0] [obs_0] [turn_1] [obs_1] ... [turn_19] [obs_19]

截断后:
  [system]  [problem]  [turn_0] [obs_0]              ← head (早期探索, 30%)
             ... 省略中间 N 轮 ...
                        [turn_16] [obs_16] ... [turn_19]  ← tail (近期历史, 70%)
```

**dynamic_history 模式（推荐，训练/推理分布完全对齐）：**

```
每步生成一个训练 Sample，使用该步骤真实看到的截断上下文:

Step 0: Sample(ctx=managed_contexts[0], resp=assistant_texts[0], reward=outcome)
Step 1: Sample(ctx=managed_contexts[1], resp=assistant_texts[1], reward=outcome)
...
Step N: Sample(ctx=managed_contexts[N], resp=assistant_texts[N], reward=outcome)

优势: 模型训练时的 ctx 与推理时 ctx 完全一致，消除分布偏移
```

---

## PRM（过程奖励模型）

```
PRM 工作流（swe_prm.py）:

每步 exec 后异步触发（不阻塞 agent 主循环）:
  input = {
    problem_statement: <issue 描述>,
    history:           <最近 max_history_steps=8 步的操作+输出>,
    current_action:    <本步 bash 命令>,
    current_output:    <本步执行结果>,
  }
         │
         ▼  HTTP → PRM SGLang Router
  judge LLM × m=3 次独立推理
         │
         ▼
  解析 \boxed{+1} 或 \boxed{-1}
  step_score = majority_vote(3 votes)

汇总（agent 结束后 collect_step_results）:
  prm_step_scores = [s_0, s_1, ..., s_N]
  prm_step_mean   = mean(prm_step_scores)

最终 reward 合并:
  final_reward = outcome_reward + prm_step_coef × prm_step_mean

  示例（prm_step_coef=1.0）:
    resolved=1, step_mean=+0.7  →  +1.0 + 0.7 = +1.7
    resolved=0, step_mean=-0.3  →  -1.0 - 0.3 = -1.3
```

**PRM 专用 GPU（5节点脚本）：** 额外分配 8 块 GPU 专门运行 PRM judge 模型，不与 SGLang policy 引擎共享。在 placement group 中位于 `prm_offset = rollout_offset + rollout_num_gpus`。

---

## 目录结构

```
swe-rl/
│
├── generate_with_swe_remote.py   # Slime 入口: generate() + reward_func()
│                                 # 运行在 RolloutManager CPU Actor 内
│                                 # 实现多轮 bash agent 循环 + Sample 构建
│
├── swe_env_client.py             # 异步 HTTP 客户端
│                                 # 封装 allocate/exec/diff/evaluate/close
│
├── swe_context_manager.py        # 上下文窗口管理
│                                 # head+tail 截断策略
│
├── swe_prm.py                    # 过程奖励模型（可选）
│                                 # 逐步 judge + m-vote + 异步收集
│
├── swe_utils.py                  # Docker 镜像名映射
│                                 # data_source → docker.io/swebench/...
│
├── message_utils.py              # 多轮对话 tokenization
│                                 # 构建 response_ids + loss_mask
│
├── swebench.yaml                 # Agent 配置
│                                 # system_template, instance_template,
│                                 # action_observation_template,
│                                 # step_limit=20, cwd=/testbed, timeout=180s
│
├── litellm.json                  # LiteLLM 模型注册表
│                                 # 本地模型名 → cost/endpoint 映射
│
├── server/
│   ├── swe_exec_server.py        # ECS Docker 节点服务（Flask :5000）
│   │                             # 管理单节点容器生命周期（无 GPU）
│   ├── swe_env_pool_server.py    # GPU 头节点 CPU 进程（Flask :18090）
│   │                             # 跨节点租约调度 + 负载均衡
│   └── setup_ecs_seed.sh         # ECS 节点一键初始化脚本
│                                 # 安装 Docker + 注册 systemd + 拉取镜像
│
├── scripts/
│   ├── run_swe_rl_32b_remote_8nodes.sh         # Qwen3-32B, 8节点, 64 GPU
│   ├── run_swe_rl_32b_remote_8nodes_resume.sh  # 同上 + 断点续训
│   ├── run_swe_rl_32b_remote_4nodes.sh         # Qwen3-32B, 4节点, 32 GPU
│   ├── run_swe_rl_8b_remote_2nodes.sh          # Qwen3-8B,  2节点, 16 GPU
│   └── run_swe_rl_8b_prm_5nodes_remote.sh      # Qwen3-8B + PRM, 5节点
│
├── eval/
│   ├── eval_swe.py               # 独立评估脚本（无训练循环）
│   └── run_eval_swe.sh           # 启动 SGLang + Pool Server + 评估
│
├── data/
│   ├── preprocess_swe_dataset.py # HuggingFace → slime JSONL 格式转换
│   └── pull_swe_images.sh        # 向 ECS 节点批量拉取 Docker 镜像
│
├── mini-swe-agent/               # 第三方依赖 (pinned v1.12.0)
│   └── ...                       # 仅用于工具/测试，训练 rollout 未使用其 CLI
│
├── docs/
│   ├── en/
│   │   ├── SWE_REMOTE_DOCKER.md  # ECS 节点部署详情
│   │   ├── CONTEXT_MANAGEMENT.md # Context 管理算法说明
│   │   └── SWE_PRM.md            # PRM 设计文档
│   └── cn/                       # 中文文档
│
└── output/                       # 运行时产物（首次运行自动创建）
    ├── ckpt/                     # 训练 checkpoint
    │   └── swe-rl-32b-remote-8nodes_<timestamp>/
    │       └── iter_XXXXXX/      # Megatron torch_dist 格式
    ├── swe_rollouts/             # Rollout 轨迹产物
    │   └── swe-rl-32b-<timestamp>/
    │       └── <instance_id>__g<group>__i<idx>__<ts>/
    │           ├── traj.json     # 完整对话 + step_debug
    │           ├── patch.diff    # Agent 提交的 git patch
    │           └── meta.json     # 模型参数 + 采样配置
    └── eval_runs/                # 独立评估结果
        └── <run>/
            ├── summary.json      # resolve_rate, avg_steps, ...
            └── results.jsonl     # 每个 instance 的详细结果
```

---

## 关键超参数

> 本节默认值以 `scripts/run_swe_rl_32b_remote_8nodes.sh` 为准。  
> 其他脚本（4 节点 / 2 节点 / 5 节点 PRM）的默认参数在下方“脚本差异”表中单独列出。

### Rollout 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--rollout-max-context-len` | 16384 | 总 context 窗口（触发 context 截断） |
| `--rollout-max-response-len` | 4096 | 每步 LLM 最大生成 tokens |
| `--n-samples-per-prompt` | 8 | GRPO group size（每个 prompt 采样轨迹数） |
| `--dynamic_history` | 启用 | 每步生成一个 training sample |
| `--rollout-temperature` | 1.0 | 采样温度 |
| `--rollout-batch-size` | 8 | 每批次并发 instance 数 |
| `SWE_MAX_CONCURRENT` | 128 | 全局最大并发 Docker 容器数 |
| `SWE_MAX_CONTAINERS_PER_NODE` | 15 | 单 ECS 节点容器上限 |

### 脚本差异（与 8 节点默认值相比）

| 脚本 | `--rollout-max-context-len` | `SWE_MAX_CONCURRENT` | `SWE_MAX_CONTAINERS_PER_NODE` | 备注 |
|------|-----------------------------|----------------------|--------------------------------|------|
| `run_swe_rl_32b_remote_4nodes.sh` | 32768 | 2 | 8 | 较小规模调试配置 |
| `run_swe_rl_8b_remote_2nodes.sh` | 32768 | 2 | 8 | 8B 轻量配置 |
| `run_swe_rl_8b_prm_5nodes_remote.sh` | 32768 | 4 | 8 | 启用 PRM，`--advantage-estimator=step_wise` |

### 优化参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--advantage-estimator` | grpo | 优势估计算法 |
| `--lr` | 1e-6 | 学习率 |
| `--eps-clip` | 0.2 | PPO clip 下界 |
| `--eps-clip-high` | 0.28 | PPO clip 上界 |
| `--kl-loss-coef` | 0.00 | KL 散度系数（默认关闭） |
| `--entropy-coef` | 0.00 | 熵正则系数 |
| `--weight-decay` | 0.1 | 权重衰减 |

### 并行参数（Qwen3-32B，8节点）

| 参数 | 值 | 说明 |
|------|-----|------|
| `--tensor-model-parallel-size` | 8 | 张量并行度（=每物理训练节点 GPU 数） |
| `--pipeline-model-parallel-size` | 1 | 流水线并行度（不跨节点） |
| `--actor-num-nodes` | 8 | Megatron 逻辑节点数（实际物理节点 4 个） |
| `--actor-num-gpus-per-node` | 4 | Megatron 逻辑节点 GPU 数 |
| `--rollout-num-gpus` | 32 | SGLang 总 GPU 数 |
| `--rollout-num-gpus-per-engine` | 4 | 每个 SGLang 引擎 GPU 数（=TP=4） |

### PRM 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--prm-enable` | false | 启用 PRM |
| `--prm-model-path` | — | PRM 模型路径 |
| `--prm-m` | 3 | 投票次数 |
| `--prm-num-gpus` | 8 | PRM 专用 GPU 数（追加到 placement group 末尾） |
| `--prm-step-coef` | 1.0 | PRM 步骤奖励权重 |

---

## 快速开始

### 环境准备

```bash
pip install -r requirements.txt
pip install -e swe-rl/mini-swe-agent

cd swe-rl/
python data/preprocess_swe_dataset.py \
  --train-data-source SumanthRH/SWE-Gym-Subset \
  --train-split train \
  --output-dir ~/data/swe_gym_subset
```

### 配置环境变量

```bash
# swe-rl/.env.swe
export SWE_EXEC_SERVER_URLS="http://172.16.0.10:5000,http://172.16.0.11:5000,..."
export HF_CKPT=/path/to/Qwen3-32B
export PROMPT_DATA=/path/to/train.jsonl
export WANDB_API_KEY=your_key  # 可选
```

### 启动训练

| 规模 | 命令 | GPU 节点 | 总 GPU |
|------|------|----------|--------|
| 生产（32B） | `bash swe-rl/scripts/run_swe_rl_32b_remote_8nodes.sh` | 8 | 64 |
| 中等（32B） | `bash swe-rl/scripts/run_swe_rl_32b_remote_4nodes.sh` | 4 | 32 |
| 轻量（8B）  | `bash swe-rl/scripts/run_swe_rl_8b_remote_2nodes.sh`  | 2 | 16 |
| 含 PRM（8B）| `bash swe-rl/scripts/run_swe_rl_8b_prm_5nodes_remote.sh` | 5 | 40 |

### 独立评估

```bash
export SWE_EXEC_SERVER_URLS="http://172.16.0.10:5000,..."
export EVAL_MODEL_PATH=/path/to/model
export PROMPT_DATA=/path/to/data.jsonl
export OUTPUT_DIR=./output/eval_runs/my_run
bash swe-rl/eval/run_eval_swe.sh
```

---

## 关键依赖

| 类别 | 库 | 用途 |
|------|----|------|
| 分布式训练 | `ray`, `torch`, `megatron_core` | 异步 RL 训练框架 |
| 推理引擎 | `sglang`, `sglang-router` | 高效 LLM 推理 + 负载均衡 |
| RL 框架 | `slime` (editable) | 训练循环、Sample 类型、Placement Group |
| LLM 调用 | `litellm` | 统一 API，对接 SGLang |
| 环境通信 | `flask`, `requests` | Pool/Exec Server HTTP 服务 |
| 模型 | `transformers` | Tokenizer、权重格式 |
| 数据 | `datasets` (HuggingFace) | SWE-Bench/SWE-Gym 数据集 |
| 监控 | `wandb`, `loguru` | 实验跟踪、日志 |
| 模板 | `jinja2`, `pyyaml` | Prompt 模板渲染 |

---

## 迁移注意事项（迁移到云平台 / K8s 环境）

### 1. 多节点启动方式适配（改动量：小）

```bash
# 脚本期望的变量
MLP_ROLE_INDEX=0               # 本节点编号（0 = 头节点）
MLP_WORKER_0_HOST=<head_ip>    # 头节点 IP
MLP_WORKER_1_HOST=<worker1_ip>
...

```

---

### 2. Docker exec 节点（改动量：取决于平台）

这是迁移的核心问题。`swe_exec_server.py` 调用本机的 `docker` CLI，部署方式取决于平台能力：

#### 情况 A：平台能提供"有 Docker daemon 的 CPU 节点"（改动量：零）

直接把 `swe_exec_server.py` 部署到这些节点，填入 `SWE_EXEC_SERVER_URLS`：

```bash
export SWE_EXEC_SERVER_URLS="http://10.0.1.10:5000,http://10.0.1.11:5000,..."
```

#### 情况 B：纯 K8s，运行时为 Docker，允许挂载 socket（改动量：零，加 K8s YAML）

```yaml
# DaemonSet 部署 swe_exec_server，挂载宿主机 Docker socket
spec:
  containers:
  - name: swe-exec-server
    volumeMounts:
    - name: docker-sock
      mountPath: /var/run/docker.sock
  volumes:
  - name: docker-sock
    hostPath:
      path: /var/run/docker.sock
```

⚠️ 安全隐患：挂载 docker.sock = pod 拿到宿主机 root 权限，需平台团队审批。建议只在专用 nodegroup 上开放。

#### 情况 C：纯 K8s，运行时为 containerd / CRI-O（AWS EKS 1.24+ 默认）

**没有 docker.sock**，必须重写 `swe_exec_server.py`，用 K8s Pod API 替换 docker CLI：

```
改动范围：仅 swe_exec_server.py（~200行）
不需要改：generate_with_swe_remote.py、swe_env_client.py、所有训练代码

docker run  → kubernetes.create_namespaced_pod()
docker exec → kubernetes.connect_get_namespaced_pod_exec()  （websocket）
docker rm   → kubernetes.delete_namespaced_pod()
```

**确认方法（问 AWS team）：**
```bash
# 在任意节点上执行
kubectl get nodes -o wide   # 看 CONTAINER-RUNTIME 列
# docker://xx.x  → 情况 B
# containerd://  → 情况 C（EKS 1.24+ 默认）
```

---

### 3. SWE-Bench 镜像预安装

SWE-Bench 每个 instance 有独立 Docker 镜像，总量达数百 GB，**必须在训练前预拉到 exec 节点**，否则每次 rollout 都会卡在镜像下载上。

```bash
# 在每台 exec 节点上执行（需 2-5 小时）
TRAIN=~/train.jsonl bash data/pull_swe_images.sh

# 验证
curl http://localhost:5000/images | python3 -m json.tool | grep count
# "count": 500+  表示镜像已就绪
```

K8s 环境下可以用 DaemonSet 的 `initContainer` 或 Job 批量预热，但镜像缓存仍然依赖节点本地存储（不能依赖 registry 按需拉取，延迟无法接受）。

---



### 4. 改动量总结

| 组件 | 原始平台（ECS） | 云/K8s 平台 | 改动量 |
|------|--------------|------------|--------|
| 多节点启动脚本 | `MLP_ROLE_INDEX` 手动设置 | 映射云调度器变量 | 小（~10行） |
| `swe_env_pool_server.py` | GPU 头节点 CPU 进程 | 不变 | 零 |
| `swe_exec_server.py` | ECS 节点，直接跑 | 取决于运行时（见上） | 零 ～ 中 |
| 核心训练代码 | — | 完全不动 | 零 |
| 镜像预热 | `setup_ecs_seed.sh` | 需适配为 K8s Job/DaemonSet | 小 |
