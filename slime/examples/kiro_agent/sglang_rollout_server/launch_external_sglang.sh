#!/bin/bash

# =============================================================================
# Multi-Node SGLang Server for SLIME External Rollout
# =============================================================================
#
# This script launches SGLang inference servers and a router within a Ray cluster,
# suitable for use with SLIME's --rollout-external mode.
#
# Supports:
#   - HyperPod/Kubernetes (auto-detects from PET_NNODES)
#   - Manual multi-node setup (via MASTER_ADDR and WORKER_IPS)
#   - Single node (default)
#
# Architecture:
#   ┌─────────────────────────────────────────────────────────────────────┐
#   │                         SGLang Router                               │
#   │                    (Load Balancer on Head Node)                     │
#   │                      http://<HEAD_IP>:<ROUTER_PORT>                 │
#   └─────────────────────────────────────────────────────────────────────┘
#                                    │
#          ┌─────────────────────────┼─────────────────────────┐
#          ▼                         ▼                         ▼
#   ┌─────────────┐          ┌─────────────┐          ┌─────────────┐
#   │  SGLang     │          │  SGLang     │          │  SGLang     │
#   │  Engine 0   │          │  Engine 1   │          │  Engine N   │
#   │  (Ray Actor)│          │  (Ray Actor)│          │  (Ray Actor)│
#   └─────────────┘          └─────────────┘          └─────────────┘
#
# Usage:
#   # HyperPod (auto-detects cluster from PET_NNODES):
#   bash examples/kiro_agent/launch_external_sglang.sh \
#       --model-path /path/to/model \
#       --num-engines 2 \
#       --tp-size 8
#
#   # Manual multi-node:
#   MASTER_ADDR=<head_ip> WORKER_IPS="<worker1_ip> <worker2_ip>" \
#       bash examples/kiro_agent/launch_external_sglang.sh \
#       --model-path /path/to/model \
#       --num-engines 2 \
#       --tp-size 8
#
# Environment Variables:
#   PET_NNODES        - Number of nodes (set by HyperPod/PyTorchJob)
#   MASTER_ADDR       - IP address of the head node (auto-detected on HyperPod)
#   WORKER_IPS        - Space-separated list of worker node IPs (manual mode)
#   NUM_GPUS_PER_NODE - GPUs per node (default: auto-detect)
#
# =============================================================================

set -e

SCRIPT_DIR=/mnt_out/myshang/codebase/slime/examples/kiro_agent

# =============================================================================
# Default Configuration
# =============================================================================
MODEL_PATH=""
NUM_ENGINES=0  # 0 = auto-calculate from cluster GPUs
TP_SIZE=8
DP_SIZE=1
MEM_FRACTION=0.6
ROUTER_PORT=30000
SERVER_BASE_PORT=13140
RAY_ADDRESS=""
RUN_NAME=${RUN_NAME:-"smoke_test"}
OUTPUT_DIR=/mnt_out/myshang/logs/slime

# Auto-detect GPUs per node
NUM_GPUS_PER_NODE=${NUM_GPUS_PER_NODE:-$(nvidia-smi -L 2>/dev/null | wc -l || echo 8)}

# Multi-node configuration from environment
MASTER_ADDR=${MASTER_ADDR:-""}
WORKER_IPS=${WORKER_IPS:-""}

# =============================================================================
# Parse Arguments
# =============================================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --num-engines)
            NUM_ENGINES="$2"
            shift 2
            ;;
        --tp-size)
            TP_SIZE="$2"
            shift 2
            ;;
        --dp-size)
            DP_SIZE="$2"
            shift 2
            ;;
        --num-gpus-per-node)
            NUM_GPUS_PER_NODE="$2"
            shift 2
            ;;
        --mem-fraction)
            MEM_FRACTION="$2"
            shift 2
            ;;
        --router-port)
            ROUTER_PORT="$2"
            shift 2
            ;;
        --server-base-port)
            SERVER_BASE_PORT="$2"
            shift 2
            ;;
        --ray-address)
            RAY_ADDRESS="$2"
            shift 2
            ;;
        --master-addr)
            MASTER_ADDR="$2"
            shift 2
            ;;
        --worker-ips)
            WORKER_IPS="$2"
            shift 2
            ;;
        --run-name)
            RUN_NAME="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 --model-path <path> [options]"
            echo ""
            echo "Options:"
            echo "  --model-path PATH       Path to HuggingFace model (required)"
            echo "  --num-engines N         Number of SGLang engines (default: auto)"
            echo "  --tp-size N             Tensor parallel size (default: 8)"
            echo "  --dp-size N             Data parallel size (default: 1)"
            echo "  --num-gpus-per-node N   GPUs per node (default: auto-detect)"
            echo "  --mem-fraction F        GPU memory fraction (default: 0.6)"
            echo "  --router-port PORT      Router port (default: 30000)"
            echo "  --server-base-port PORT Server base port (default: 13140)"
            echo "  --ray-address ADDR      Existing Ray cluster address (skip cluster setup)"
            echo "  --master-addr ADDR      Head node IP for Ray cluster"
            echo "  --worker-ips IPS        Space-separated worker node IPs"
            echo "  --run-name NAME         Run name for output directory isolation"
            echo "  --output-dir DIR        Base output directory (default: /mnt_out/myshang/logs/slime)"
            echo ""
            echo "Environment Variables:"
            echo "  PET_NNODES              Number of nodes (HyperPod/PyTorchJob)"
            echo "  MASTER_ADDR             Head node IP (alternative to --master-addr)"
            echo "  WORKER_IPS              Worker IPs (alternative to --worker-ips)"
            echo "  NUM_GPUS_PER_NODE       GPUs per node (alternative to --num-gpus-per-node)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# =============================================================================
# Validate Configuration
# =============================================================================
if [ -z "$MODEL_PATH" ]; then
    echo "Error: --model-path is required"
    exit 1
fi

# =============================================================================
# Auto-detect HyperPod/Kubernetes Environment
# =============================================================================
# Check if running on HyperPod (PET_NNODES is set by PyTorchJob)
if [ -n "${PET_NNODES:-}" ]; then
    echo "Detected HyperPod/PyTorchJob environment (PET_NNODES=${PET_NNODES})"
    WORLD_SIZE=${PET_NNODES}
    
    # Derive MASTER_ADDR from hostname pattern (e.g., job-name-worker-0)
    # PyTorchJob names workers as: <job-name>-worker-0, <job-name>-worker-1, etc.
    if [ -z "$MASTER_ADDR" ]; then
        MASTER_ADDR=$(hostname | sed -E 's/-worker-[0-9]+$/-worker-0/')
        echo "Auto-detected MASTER_ADDR: $MASTER_ADDR"
    fi
    
    MY_ADDR=$(hostname)
    
    if [[ "${MASTER_ADDR}" == "${MY_ADDR}" ]]; then
        IS_MASTER=true
        echo "This node is the HEAD node"
    else
        IS_MASTER=false
        echo "This node is a WORKER node (head: ${MASTER_ADDR})"
    fi
else
    # Manual mode or single node
    IS_MASTER=true
    
    if [ -z "$MASTER_ADDR" ]; then
        MASTER_ADDR=$(hostname -I | awk '{print $1}')
        echo "MASTER_ADDR not set, using detected IP: $MASTER_ADDR"
    fi
    
    # Count total nodes from WORKER_IPS
    WORKER_COUNT=0
    if [ -n "$WORKER_IPS" ]; then
        WORKER_COUNT=$(echo "$WORKER_IPS" | wc -w)
    fi
    WORLD_SIZE=$((1 + WORKER_COUNT))
fi

# Calculate total GPUs
GPUS_PER_ENGINE=$((TP_SIZE * DP_SIZE))
EXPECTED_GPUS=$((WORLD_SIZE * NUM_GPUS_PER_NODE))

# Auto-calculate NUM_ENGINES if not explicitly set (0 or default)
if [ "$NUM_ENGINES" -le 0 ]; then
    NUM_ENGINES=$((EXPECTED_GPUS / GPUS_PER_ENGINE))
    echo "Auto-calculated NUM_ENGINES: $NUM_ENGINES (from $EXPECTED_GPUS GPUs / $GPUS_PER_ENGINE GPUs per engine)"
fi

TOTAL_GPUS=$((NUM_ENGINES * GPUS_PER_ENGINE))

echo "=============================================="
echo "SLIME External Rollout - SGLang Server Setup"
echo "=============================================="
echo "Model Path:        $MODEL_PATH"
echo "Head Node:         $MASTER_ADDR"
echo "Total Nodes:       $WORLD_SIZE"
echo "GPUs per Node:     $NUM_GPUS_PER_NODE"
echo "Total GPUs:        $EXPECTED_GPUS"
echo ""
echo "SGLang Configuration:"
echo "  Num Engines:     $NUM_ENGINES"
echo "  TP Size:         $TP_SIZE"
echo "  DP Size:         $DP_SIZE"
echo "  GPUs per Engine: $GPUS_PER_ENGINE"
echo "  Memory Fraction: $MEM_FRACTION"
echo ""
echo "Network:"
echo "  Router Port:     $ROUTER_PORT"
echo "  Server Base Port:$SERVER_BASE_PORT"
echo "=============================================="

# Validate GPU count
if [ "$TOTAL_GPUS" -gt "$EXPECTED_GPUS" ]; then
    echo "ERROR: Requested $TOTAL_GPUS GPUs but only $EXPECTED_GPUS available"
    echo "       ($WORLD_SIZE nodes × $NUM_GPUS_PER_NODE GPUs/node)"
    exit 1
fi


# =============================================================================
# Step 1: Start Ray Cluster
# =============================================================================
if [ -z "$RAY_ADDRESS" ]; then
    echo ""
    echo "[Step 1/3] Setting up Ray cluster..."

    # Stop any existing Ray processes on this node
    echo "Stopping existing Ray processes..."
    ray stop --force 2>/dev/null || true
    sleep 2

    # Set Ray object store configuration
    export RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE=1

    if [ "$IS_MASTER" = true ]; then
        # =====================================================================
        # HEAD NODE: Start Ray head and wait for workers
        # =====================================================================
        echo "Starting Ray head node at $MASTER_ADDR..."
        ray start --head \
            --node-ip-address="$MASTER_ADDR" \
            --num-gpus="$NUM_GPUS_PER_NODE" \
            --object-store-memory=200000000000 \
            --disable-usage-stats \
            --dashboard-host=0.0.0.0 \
            --dashboard-port=8265

        # For manual multi-node setup, start workers via SSH
        if [ -n "$WORKER_IPS" ]; then
            echo "Starting Ray worker nodes via SSH..."
            for WORKER_IP in $WORKER_IPS; do
                echo "  - Connecting worker: $WORKER_IP"
                ssh -o StrictHostKeyChecking=no "$WORKER_IP" "
                    ray stop --force 2>/dev/null || true
                    sleep 2
                    export RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE=1
                    ray start --address=${MASTER_ADDR}:6379 \
                        --num-gpus=${NUM_GPUS_PER_NODE} \
                        --node-ip-address=\$(hostname -I | awk '{print \$1}') \
                        --object-store-memory=200000000000 \
                        --disable-usage-stats
                " &
            done
            wait
            echo "All worker nodes started via SSH"
        fi

        # Wait for all nodes to join the cluster
        echo ""
        echo "Waiting for all $WORLD_SIZE nodes to join the Ray cluster..."
        MAX_WAIT=300
        WAITED=0
        while true; do
            # Count nodes using Python for reliability
            NODE_COUNT=$(python3 -c "
import ray
try:
    ray.init(address='auto', ignore_reinit_error=True)
    nodes = ray.nodes()
    alive_nodes = [n for n in nodes if n.get('Alive', False)]
    print(len(alive_nodes))
except:
    print(0)
" 2>/dev/null || echo "0")
            
            if [ "$NODE_COUNT" -ge "$WORLD_SIZE" ]; then
                echo "All $NODE_COUNT / $WORLD_SIZE nodes have joined the cluster."
                break
            fi
            
            if [ "$WAITED" -ge "$MAX_WAIT" ]; then
                echo "ERROR: Timeout waiting for nodes. Only $NODE_COUNT / $WORLD_SIZE joined."
                ray status
                exit 1
            fi
            
            echo "  Waiting for nodes... ($NODE_COUNT / $WORLD_SIZE joined, ${WAITED}s elapsed)"
            sleep 5
            WAITED=$((WAITED + 5))
        done

    else
        # =====================================================================
        # WORKER NODE: Join the Ray cluster
        # =====================================================================
        echo "Waiting for head node at ${MASTER_ADDR}:6379..."
        MAX_WAIT=300
        WAITED=0
        until ray health-check --address="${MASTER_ADDR}:6379" 2>/dev/null; do
            if [ "$WAITED" -ge "$MAX_WAIT" ]; then
                echo "ERROR: Timeout waiting for head node at ${MASTER_ADDR}:6379"
                exit 1
            fi
            sleep 5
            WAITED=$((WAITED + 5))
        done
        echo "Head node is ready, joining cluster..."

        ray start \
            --address="${MASTER_ADDR}:6379" \
            --node-ip-address="$(hostname -I | awk '{print $1}')" \
            --num-gpus="$NUM_GPUS_PER_NODE" \
            --object-store-memory=200000000000 \
            --disable-usage-stats

        echo "Joined Ray cluster. Worker node will now sleep."
        echo "The head node will manage SGLang servers via Ray actors."
        
        # Worker nodes just keep alive - the Python module handles them via Ray
        sleep infinity
    fi
else
    echo ""
    echo "[Step 1/3] Using existing Ray cluster at $RAY_ADDRESS"
fi

# =============================================================================
# Step 2: Verify Ray Cluster (Head Node Only)
# =============================================================================
echo ""
echo "[Step 2/3] Verifying Ray cluster..."

# Check cluster status
echo "Ray cluster status:"
ray status

# Verify GPU count using Python
ACTUAL_GPUS=$(python3 -c "
import ray
ray.init(address='auto', ignore_reinit_error=True)
gpus = ray.cluster_resources().get('GPU', 0)
print(int(gpus))
" 2>/dev/null || echo "0")

echo ""
echo "Cluster GPU verification:"
echo "  Expected GPUs: $EXPECTED_GPUS"
echo "  Detected GPUs: $ACTUAL_GPUS"

if [ "$ACTUAL_GPUS" -lt "$TOTAL_GPUS" ]; then
    echo "ERROR: Cluster has fewer GPUs ($ACTUAL_GPUS) than required ($TOTAL_GPUS)"
    exit 1
fi
echo "  ✓ Cluster has sufficient GPUs"

# =============================================================================
# Step 3: Launch SGLang Servers via Python Script
# =============================================================================
echo ""
echo "[Step 3/3] Launching SGLang servers..."

# Build Python command
CMD="python3 ${SCRIPT_DIR}/sglang_rollout_server/launch_external_sglang.py"
CMD="$CMD --model-path $MODEL_PATH"
CMD="$CMD --num-engines $NUM_ENGINES"
CMD="$CMD --tp-size $TP_SIZE"
CMD="$CMD --dp-size $DP_SIZE"
CMD="$CMD --mem-fraction-static $MEM_FRACTION"
CMD="$CMD --router-port $ROUTER_PORT"
CMD="$CMD --server-base-port $SERVER_BASE_PORT"

if [ -n "$RAY_ADDRESS" ]; then
    CMD="$CMD --ray-address $RAY_ADDRESS"
fi

if [ -n "$RUN_NAME" ]; then
    CMD="$CMD --run-name $RUN_NAME"
fi

if [ -n "$OUTPUT_DIR" ]; then
    CMD="$CMD --output-dir $OUTPUT_DIR"
fi

echo "Executing: $CMD"
echo ""

# =============================================================================
# Cleanup Handler
# =============================================================================
cleanup() {
    echo ""
    echo "Shutting down SGLang external rollout servers..."
    
    # Kill Python process
    pkill -9 -f "launch_external_sglang.py" 2>/dev/null || true
    
    # Kill SGLang processes
    pkill -9 -f "sglang" 2>/dev/null || true
    
    # Stop Ray on workers (manual mode only)
    if [ -n "$WORKER_IPS" ] && [ -z "$RAY_ADDRESS" ]; then
        for WORKER_IP in $WORKER_IPS; do
            echo "Cleaning up worker node $WORKER_IP..."
            ssh -o StrictHostKeyChecking=no "$WORKER_IP" "
                pkill -9 -f sglang 2>/dev/null || true
                ray stop --force 2>/dev/null || true
            " &
        done
        wait
    fi
    
    # Stop local Ray (if we started it)
    if [ -z "$RAY_ADDRESS" ]; then
        ray stop --force 2>/dev/null || true
    fi
    
    echo "Cleanup complete"
    exit 0
}

trap cleanup SIGINT SIGTERM

# Execute the Python launcher
exec $CMD
