# Kiro-on-Strands SLIME Integration

This example shows how to integrate your existing Strands agent workflows (with Docker environment interaction) into SLIME for GRPO training.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    SLIME Training Loop                          │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐ │
│  │   Trainer   │───▶│  Rollout    │───▶│  slime_generate.py  │ │
│  │  (Megatron) │    │  Manager    │    │  (your agent)       │ │
│  └─────────────┘    └─────────────┘    └─────────────────────┘ │
│        │                                        │               │
│        │ NCCL weight update                     │               │
│        ▼                                        ▼               │
│  ┌─────────────┐                    ┌─────────────────────────┐ │
│  │   SGLang    │◀───────────────────│  Strands Agent +       │ │
│  │   Server    │    HTTP /generate  │  Docker Environment    │ │
│  └─────────────┘                    └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## Key Benefits

| Feature | Your Current Setup (vLLM) | SLIME Integration (SGLang) |
|---------|---------------------------|---------------------------|
| Weight Updates | Restart vLLM server | NCCL in-place (fast) |
| Token Tracking | Retokenization needed | TITO (exact tokens) |
| Loss Masking | Manual | Automatic via strands-sglang |
| Environment | Docker containers ✓ | Docker containers ✓ |

## Prerequisites

1. **Install SLIME**
   ```bash
   cd /root/slime
   pip install -e . --no-deps
   ```

2. **Install strands-sglang**
   ```bash
   pip install strands-sglang @ git+https://github.com/horizon-rl/strands-sglang.git
   ```

3. **Set PYTHONPATH to include Kiro-on-Strands**
   ```bash
   export KIRO_ON_STRANDS_PATH=/path/to/Kiro-on-Strands
   export PYTHONPATH=${KIRO_ON_STRANDS_PATH}:${PYTHONPATH}
   ```

   This allows `slime_generate.py` to import your existing modules:
   - `utils.workspace_manager.WorkspaceManager`
   - `utils.docker_utils` (setup_aswe_workspace, stop_container, etc.)
   - `cli_utils.build_agent`
   - `remote_rollout_server.metrics_saver.MetricsSaver`
   - `prompts.*` (system prompts, instructions)

## Files

- `slime_generate.py` - Custom generate and reward functions that import from Kiro-on-Strands
- `run_slime_training.sh` - Training script with PYTHONPATH setup
- `run_workflow.py` - Your existing workflow (reference, copied from Kiro-on-Strands)
- `remote_rollout.py` - Your existing server (reference, copied from Kiro-on-Strands)

## Integration Steps

### 1. Set PYTHONPATH

The key to importing your Kiro-on-Strands code is setting PYTHONPATH:

```bash
# Option A: Environment variable
export KIRO_ON_STRANDS_PATH=/path/to/Kiro-on-Strands
export PYTHONPATH=${KIRO_ON_STRANDS_PATH}:${PYTHONPATH}

# Option B: In the training script (already configured in run_slime_training.sh)
KIRO_ON_STRANDS_PATH=/path/to/Kiro-on-Strands
PYTHONPATH=${KIRO_ON_STRANDS_PATH}:${PYTHONPATH} python train.py ...
```

### 2. How Imports Work

`slime_generate.py` imports environment utilities from your Kiro-on-Strands codebase:

```python
# These imports work because Kiro-on-Strands is in PYTHONPATH
from utils.workspace_manager import WorkspaceManager
from utils.docker_utils import setup_aswe_workspace, stop_container
from remote_rollout_server.metrics_saver import MetricsSaver
from prompts.maestro_system_prompt import MAESTRO_SYSTEM_PROMPT
```

**Note:** We do NOT import `build_agent` from `cli_utils` because it depends on the old Strands API (`strands.handlers.tool_handler`). Instead, we build the agent directly using `strands-sglang` and define tools inline that execute commands via `docker exec`.

### 3. Key Integration Points

The `slime_generate.py` reuses your existing code:

| Function | Uses From Kiro-on-Strands |
|----------|---------------------------|
| `setup_workspace_for_sample()` | `setup_aswe_workspace()`, `setup_public_container_workspace()` |
| `build_strands_agent_with_sglang()` | Tools execute via `docker exec` into container |
| `generate()` | `get_file_tree()`, `KIRO_BENCHMARKING_INSTRUCTION` |
| `reward_func()` | `MetricsSaver` |
| `cleanup_workspace()` | `stop_container()` |

**Docker Execution:** Since `WorkspaceManager` is just a dataclass (not an executor), tools are implemented using `docker exec` to run commands inside the container:

```python
# Tools execute commands via docker exec
def execute_in_container(container_id, command, workdir):
    subprocess.run(["docker", "exec", "-w", workdir, container_id, "bash", "-c", command])
```

### 4. What's Different from vLLM

**Replace vLLM with SGLang:**
```python
# Before (vLLM in remote_rollout.py)
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1")

# After (SGLang via strands-sglang in slime_generate.py)
from strands_sglang import SGLangClient, SGLangModel
client = SGLangClient.from_slime_args(args)
model = SGLangModel(client=client, tokenizer=tokenizer, ...)
```

**Extract tokens with TITO:**
```python
# strands-sglang tracks tokens automatically
tm = model.token_manager
sample.tokens = tm.token_ids
sample.loss_mask = tm.loss_mask[prompt_len:]
sample.rollout_log_probs = tm.logprobs[prompt_len:]
```

### 3. Prepare Dataset

Your dataset should be in CSV or parquet format with these columns:
- `instance_id` - Unique identifier
- `problem_statement` - The task/problem description
- `patch` (optional) - Target patch for reward computation

### 4. Convert Model Checkpoint

```bash
cd /root/slime
source scripts/models/qwen3-8B.sh

PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint /root/models/Qwen3-8B \
    --save /root/models/Qwen3-8B_torch_dist
```

### 5. Run Training

```bash
# Set the path to your Kiro-on-Strands directory
export KIRO_ON_STRANDS_PATH=/path/to/Kiro-on-Strands

# Run training
bash examples/kiro_on_strands/run_slime_training.sh
```

## Configuration

### GPU Allocation

For 8 GPUs with TP=4:
- Actor (training): 4 GPUs
- Rollout (SGLang): 4 GPUs

```bash
--actor-num-gpus-per-node 4
--rollout-num-gpus-per-engine 4
--tensor-model-parallel-size 4
--colocate  # Actor and rollout share the same node
```

### Multi-Node Setup

For larger models or more parallelism:
```bash
--actor-num-nodes 2
--actor-num-gpus-per-node 8
# Remove --colocate for separate rollout nodes
```

### Custom Rollout Function

Point SLIME to your custom functions:
```bash
--rollout-function-path examples.kiro_on_strands.slime_generate.generate
--reward-function-path examples.kiro_on_strands.slime_generate.reward_func
```

## Debugging

### Test Imports

```bash
# Verify Kiro-on-Strands imports work
export KIRO_ON_STRANDS_PATH=/path/to/Kiro-on-Strands
export PYTHONPATH=${KIRO_ON_STRANDS_PATH}:${PYTHONPATH}

python -c "
from utils.workspace_manager import WorkspaceManager
from utils.docker_utils import setup_aswe_workspace
from remote_rollout_server.metrics_saver import MetricsSaver
print('All imports successful!')
"
```

Note: We do NOT import `build_agent` from `cli_utils` because it uses the old Strands API.

### Test Generate Function Locally

```python
import asyncio
from slime.utils.types import Sample

# Create test sample
sample = Sample(
    index=0,
    prompt="Fix the bug in the login function",
    metadata={"instance_id": "test_001"}
)

# Mock args
class Args:
    sglang_router_ip = "127.0.0.1"
    sglang_router_port = 30000
    hf_checkpoint = "/root/models/Qwen3-8B"

args = Args()
sampling_params = {"max_new_tokens": 4096, "temperature": 1.0}

# Test
from examples.kiro_on_strands.slime_generate import generate
result = asyncio.run(generate(args, sample, sampling_params))
print(f"Status: {result.status}, Response length: {result.response_length}")
```

### Check SGLang Server

```bash
curl http://localhost:30000/health
curl http://localhost:30000/get_model_info
```

## Comparison with Other Examples

| Example | Use Case | Environment |
|---------|----------|-------------|
| `strands_sglang/` | Math with Python tool | Local subprocess |
| `tau-bench/` | Tool calling benchmark | Mock environment |
| `geo3k_vlm_multi_turn/` | VLM multi-turn | Custom env class |
| **`kiro_on_strands/`** | SWE tasks | Docker containers |

## Troubleshooting

### "Could not import Kiro-on-Strands modules"
```bash
# Make sure PYTHONPATH is set correctly
export KIRO_ON_STRANDS_PATH=/path/to/Kiro-on-Strands
export PYTHONPATH=${KIRO_ON_STRANDS_PATH}:${PYTHONPATH}

# Verify the path exists
ls ${KIRO_ON_STRANDS_PATH}/utils/workspace_manager.py
ls ${KIRO_ON_STRANDS_PATH}/cli_utils.py
```

### "strands-sglang not found"
```bash
pip install strands-sglang @ git+https://github.com/horizon-rl/strands-sglang.git
```

### Docker permission errors
Ensure Docker daemon is running and accessible:
```bash
docker ps
# If using Docker-in-Docker, set DOCKER_HOST
export DOCKER_HOST=tcp://localhost:2375
```

### NCCL timeout during weight update
Check network connectivity between trainer and SGLang nodes:
```bash
# On trainer node
nc -zv <sglang_node_ip> <nccl_port>
```

### Out of memory
Reduce batch sizes or increase GPU memory fraction:
```bash
--rollout-batch-size 4
--sglang-mem-fraction-static 0.6
--max-tokens-per-gpu 4096
```
