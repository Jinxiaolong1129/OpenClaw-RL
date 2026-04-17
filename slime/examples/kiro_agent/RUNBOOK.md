# SLIME + KoS Multi-Turn RL Training Runbook

This runbook describes the end-to-end workflow for training multi-turn agent models using SLIME with external SGLang rollout servers and Kiro-on-Strands (KoS) remote rollout.

## Overview

The workflow uses two separate Kubernetes jobs that coordinate via a shared filesystem:

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Job 1: launch_sglang_and_kos_p5.yaml  (4 nodes × 8 GPUs)            │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  Each Node:                                                      │   │
│  │  ┌─────────────┐  ┌─────────────────────────┐                   │   │
│  │  │ SGLang      │  │ KoS Remote Rollout      │                   │   │
│  │  │ Engine      │  │ FastAPI Server (:5000)   │                   │   │
│  │  │ (Ray Actor) │  │ (Docker + Strands Agent) │                   │   │
│  │  └─────────────┘  └─────────────────────────┘                   │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│  Head node also runs: SGLang Router (:30000)                           │
│                                                                         │
│  Writes: <OUTPUT_DIR>/<RUN_NAME>/sglang_external_rollout.env           │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                    shared filesystem (.env file)
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Job 2: slime_train_smoke_test.yaml  (8 nodes × 8 GPUs)              │
│                                                                         │
│  Reads .env file → connects to external SGLang engines                 │
│  Auto-discovers KoS server URLs from engine IPs                        │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  Ray Cluster (8 nodes):                                          │   │
│  │  - Megatron-LM training (GRPO)                                   │   │
│  │  - Weight sync to external SGLang engines                        │   │
│  │  - Sends rollout requests to KoS servers via HTTP                │   │
│  │  - Receives trajectories + patches for reward computation        │   │
│  └──────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

## Data Flow

```
Training Loop (per step):
  1. SLIME picks a batch of SWE tasks from the parquet dataset
  2. Sends each task to a KoS FastAPI server (least-pending load balancing)
  3. KoS server runs the Strands agent in a Docker container:
     - Agent calls SGLang router for LLM inference (multi-turn)
     - Agent executes tools (read_file, write_file, execute_command) in Docker
     - Produces a trajectory (conversation) and a git patch
  4. SLIME receives the trajectory + patch
  5. Reward function scores the patch (binary: non-empty patch = 1, empty = 0)
  6. SLIME computes GRPO advantage and updates model weights
  7. Updated weights are synced to external SGLang engines
  8. Repeat
```

## Prerequisites

- Docker image with SLIME, SGLang, KoS, and dependencies pre-installed
- Model weights (HuggingFace format + Megatron-Core converted reference)
- Training data in parquet format
- SWE Docker images for task execution
- Workspace data for SWE tasks

## Step 1: Launch the Rollout Server (SGLang + KoS)

Submit the rollout server job. This starts SGLang inference engines and KoS FastAPI servers on all nodes.

```bash
kubectl apply -f examples/kiro_agent/sglang_rollout_server/job_yamsl/launch_sglang_and_kos_p5.yaml
```

### What this job does (on each node):

The YAML runs `launch_sglang_and_kos.sh`, which orchestrates a 4-step process:

1. **Ray cluster setup** — Head node starts Ray, workers join. All 4 nodes form a single Ray cluster.
2. **SGLang engine launch** — `launch_external_sglang.py` starts one SGLang engine per node as a Ray actor (TP=8, using all 8 GPUs per node). A router on the head node load-balances across engines.
3. **Health verification** — Waits for all engines + router to be healthy, then writes `sglang_external_rollout.env` to the shared filesystem.
4. **KoS server launch** — Each node starts a `sglang_remote_rollout.py` FastAPI server on port 5000. These servers handle Docker container management, Strands agent execution, and trajectory formatting. They all share the same SGLang router for LLM inference.

### Key environment variables to configure:

| Variable | Description | Default in YAML |
|----------|-------------|-----------------|
| `MODEL_PATH` | HuggingFace model path | `Qwen3-Coder-30B-A3B-Instruct` |
| `TP_SIZE` | Tensor parallel size per engine | `8` |
| `MEM_FRACTION` | GPU memory fraction for SGLang | `0.6` (lower because KoS Docker also uses the node) |
| `ROUTER_PORT` | SGLang router port | `30000` |
| `RUN_NAME` | Isolates config files per experiment | `smoke_test` |
| `KOS_WORKSPACE_PATH` | Path to SWE task workspaces | (set per experiment) |
| `KOS_SWE_DOCKER_IMAGES` | Path to pre-built SWE Docker images | (set per experiment) |
| `KOS_NUM_WORKERS` | Concurrent agent workers per node | `4` |
| `KOS_MAX_ITERATIONS` | Max tool-call iterations per agent run | `30` |
| `KOS_TIMEOUT` | Timeout per agent run (seconds) | `3600` |

### How to verify it's running:

```bash
# Check pods are running
kubectl get pods -n hyperpod-ns-aladdin | grep sglang-kos

# Check logs on head node (worker-0)
kubectl logs <pod-name-worker-0> -n hyperpod-ns-aladdin | tail -50

# Look for "All services running!" in the log
# The .env file should exist at: /mnt_out/myshang/logs/slime/<RUN_NAME>/sglang_external_rollout.env
```

### The .env file (coordination mechanism):

When SGLang is ready, `launch_external_sglang.py` writes this file:

```bash
SGLANG_ROUTER_IP=<head_node_ip>
SGLANG_ROUTER_PORT=30000
SGLANG_ENGINE_ADDRS="<ip1>:13140 <ip2>:13240 <ip3>:13340 <ip4>:13440"
SGLANG_NUM_ENGINES=4
SGLANG_TOTAL_GPUS=32
SGLANG_GPUS_PER_ENGINE=8
```

This file is the handshake between Job 1 and Job 2. The training job reads it to discover the SGLang endpoints.

## Step 2: Launch the Training Job

Once the rollout server is running (the .env file exists), submit the training job:

```bash
kubectl apply -f examples/kiro_agent/jobs/slime_train_smoke_test.yaml
```

### What this job does:

The YAML first verifies the `.env` file exists, then runs `run-qwen3-30b-kiro-kos.sh`, which:

1. **Sources the .env file** — Gets `SGLANG_ROUTER_IP`, `SGLANG_ENGINE_ADDRS`, etc.
2. **Auto-discovers KoS server URLs** — Extracts unique IPs from `SGLANG_ENGINE_ADDRS` and builds `KOS_REMOTE_URLS` (e.g., `http://<ip1>:5000,http://<ip2>:5000,...`).
3. **Sets up a Ray cluster** — 8 training nodes form their own Ray cluster (separate from the rollout Ray cluster).
4. **Submits the training job** via `ray job submit` with:
   - `--rollout-external` — Tells SLIME to connect to pre-launched SGLang engines (no colocated rollout).
   - `--rollout-external-engine-addrs` — The engine addresses from the .env file.
   - `--custom-generate-function-path examples.kiro_agent.kiro_generate_with_kos.generate` — The custom generate function that delegates to KoS servers.
   - `--custom-rm-path examples.kiro_agent.kiro_generate_with_kos.reward_func` — The reward function.

### Key environment variables to configure:

| Variable | Description | Default in YAML |
|----------|-------------|-----------------|
| `MODEL_PATH` | Must match the rollout server's model | `Qwen3-Coder-30B-A3B-Instruct` |
| `RUN_NAME` | Must match the rollout server's RUN_NAME | `smoke_test` |
| `SLIME_DIR` | Path to SLIME codebase | `/mnt_out/myshang/codebase/slime` |
| `KOS_TIMEOUT` | Timeout for remote KoS requests | `3600` |
| `TRAJECTORY_FOLDER` | Where to save trajectory logs | `/mnt_out/myshang/logs/slime/trajectory_folder` |

### Key training parameters (in `run-qwen3-30b-kiro-kos.sh`):

| Parameter | Value | Notes |
|-----------|-------|-------|
| `--rollout-batch-size` | 8 | Number of concurrent rollout requests |
| `--n-samples-per-prompt` | 2 | Rollouts per prompt (for GRPO advantage) |
| `--rollout-max-response-len` | 16384 | Max tokens per agent trajectory |
| `--global-batch-size` | 8 | Training batch size |
| `--tensor-model-parallel-size` | 4 | Training TP (different from rollout TP=8) |
| `--pipeline-model-parallel-size` | 2 | Training PP |
| `--context-parallel-size` | 2 | Training CP |
| `--expert-model-parallel-size` | 8 | MoE expert parallelism |
| `--lr` | 1e-6 | Learning rate |
| `--eps-clip` | 0.2 | PPO clip range |

## Step 3: Monitor Training

```bash
# Training logs
kubectl logs <training-pod-worker-0> -n hyperpod-ns-aladdin -f

# Or check the log file directly on shared storage:
# /mnt_out/myshang/logs/slime/train_kos_8nodes_<timestamp>.log

# Check KoS server logs (on rollout nodes):
# /mnt_out/myshang/logs/slime/kos/<RUN_NAME>/logs/rollout_logs/

# Check trajectories:
# /mnt_out/myshang/logs/slime/kos/<RUN_NAME>/trajectories/
```

## Customization Guide

### Changing the model

1. Update `MODEL_PATH` in both YAMLs (must be the same model).
2. Update `--hf-checkpoint` and `--ref-load` in `run-qwen3-30b-kiro-kos.sh`.
3. Update the model args source script (e.g., `scripts/models/qwen3-30B-A3B.sh`).
4. Adjust `TP_SIZE` / `MEM_FRACTION` if the model size changes.

### Changing the dataset

Update in `run-qwen3-30b-kiro-kos.sh`:
```bash
--prompt-data /path/to/your/data.parquet
--input-key your_prompt_column
--label-key your_label_column
```

The parquet must have at minimum a prompt column. For SWE tasks, it should also have `instance_id`, `base_commit`, and workspace metadata.

### Scaling nodes

- Rollout server: Change `replicas` in `launch_sglang_and_kos_p5.yaml`. More nodes = more SGLang engines + more KoS workers.
- Training: Change `replicas` in `slime_train_smoke_test.yaml`. Adjust parallelism args accordingly.

### Using a different instance type

Update `nodeSelector` in both YAMLs and adjust `NUM_GPUS_PER_NODE` / `TP_SIZE` accordingly.

## Troubleshooting

### Training job fails with "SGLANG_ENV_FILE not found"

The rollout server hasn't finished starting yet. Check:
- Are the rollout pods running? (`kubectl get pods`)
- Check rollout pod logs for errors.
- The .env file path must match between both jobs (controlled by `RUN_NAME` and `SGLANG_ENV_BASE`).

### KoS server returns errors

- Check Docker daemon is running on rollout nodes (required for SWE tasks).
- Verify `KOS_WORKSPACE_PATH` and `KOS_SWE_DOCKER_IMAGES` paths exist.
- Check KoS logs at `<OUTPUT_DIR>/kos/<RUN_NAME>/logs/rollout_logs/`.

### Weight sync failures

- Ensure `MODEL_PATH` is identical in both jobs.
- The training job's `--rollout-external-engine-addrs` must match the actual engine addresses.
- Check that the rollout Ray cluster is still healthy (`ray status` on rollout head node).

### Out of GPU memory on rollout nodes

- Lower `MEM_FRACTION` (e.g., from 0.6 to 0.5) to leave more memory for Docker containers.
- Reduce `KOS_NUM_WORKERS` to run fewer concurrent agent containers.

## File Reference

| File | Purpose |
|------|---------|
| `sglang_rollout_server/job_yamsl/launch_sglang_and_kos_p5.yaml` | K8s job: rollout server (SGLang + KoS) |
| `sglang_rollout_server/launch_sglang_and_kos.sh` | Orchestrates Ray + SGLang + KoS on each node |
| `sglang_rollout_server/launch_external_sglang.py` | Launches SGLang engines as Ray actors + router |
| `sglang_rollout_server/launch_external_sglang.sh` | Alternative: SGLang-only launch (no KoS) |
| `jobs/slime_train_smoke_test.yaml` | K8s job: SLIME training |
| `run-qwen3-30b-kiro-kos.sh` | Training script: Ray setup + SLIME train.py |
| `kiro_generate_with_kos.py` | Custom generate function (direct + remote KoS modes) |
| `custom_data_source.py` | Custom data source for parquet datasets |
| `Kiro-on-Strands/remote_rollout_server/sglang_remote_rollout.py` | KoS FastAPI server |
| `Kiro-on-Strands/model_config.json` | Model config (hostedsglang entry used by KoS) |
