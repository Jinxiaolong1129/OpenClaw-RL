#!/usr/bin/env python3
"""
Launch multi-node SGLang server(s) for SLIME external rollout.

This script launches SGLang inference servers and a router within a Ray cluster,
suitable for use with SLIME's --rollout-external mode.

Reuses SLIME's existing sglang_engine infrastructure.

Usage:
    python examples/kiro_agent/launch_external_sglang.py \
        --model-path /path/to/model \
        --tp-size 8

    # Then use the output config with SLIME training:
    source /mnt_out/myshang/logs/slime/sglang_external_rollout.env
    python train.py --rollout-external ...
"""

import argparse
import json
import logging
import multiprocessing
import os
import socket
import time

import ray
import requests
from sglang.srt.server_args import ServerArgs

from slime.backends.sglang_utils.sglang_engine import launch_server_process

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def get_node_ip() -> str:
    """Get the IP address of the current node, resolving hostname if needed."""
    ip = ray.util.get_node_ip_address()
    if ip and not ip[0].isdigit():
        try:
            ip = socket.gethostbyname(ip)
        except socket.gaierror:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                s.connect(("8.8.8.8", 80))
                ip = s.getsockname()[0]
            finally:
                s.close()
    return ip


def launch_router(host: str, port: int, worker_urls: list[str] = None) -> multiprocessing.Process:
    """Launch the SGLang router with pre-configured workers.
    
    The new sglang_router (sgl-model-gateway Rust version) uses HTTP/2/gRPC for 
    health checks when workers are added dynamically. To avoid compatibility issues
    with SGLang servers (HTTP/1.1 only), we pass worker_urls at startup time.
    
    Note: Router binds to 0.0.0.0 to accept connections from any interface,
    but we track the actual IP in `host` for external access.
    """
    from sglang_router.launch_router import RouterArgs
    from slime.utils.http_utils import run_router

    router_args = RouterArgs(
        host="0.0.0.0",  # Bind to all interfaces (allows localhost access)
        port=port,
        balance_abs_threshold=16,  # Tolerate imbalance to favor cache-aware routing
        log_level="warn",
        worker_urls=worker_urls or [],
        worker_startup_timeout_secs=1200,  # 20 min - servers should already be ready
    )

    logger.info(f"Launching SGLang router at 0.0.0.0:{port} (accessible via {host}:{port})")
    logger.info(f"  Workers: {worker_urls}")

    proc = multiprocessing.Process(target=run_router, args=(router_args,))
    proc.daemon = True
    proc.start()
    time.sleep(3)

    if not proc.is_alive():
        raise RuntimeError("Router process died during startup")

    logger.info(f"Router launched at 0.0.0.0:{port}")
    return proc


def wait_router_ready(router_host: str, router_port: int, timeout: int = 120):
    """Wait for router to be ready and healthy."""
    router_url = f"http://{router_host}:{router_port}"
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"{router_url}/health", timeout=5)
            if response.status_code == 200:
                logger.info(f"Router is ready at {router_url}")
                return True
        except requests.RequestException:
            pass
        time.sleep(2)
    
    raise TimeoutError(f"Router not ready after {timeout}s")


def _wait_server_healthy(server_url: str, timeout: int = 600):
    """Wait for a server to be healthy."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"{server_url}/health_generate", timeout=5)
            if response.status_code == 200:
                logger.info(f"Server {server_url} is healthy")
                return
        except requests.RequestException:
            pass
        time.sleep(5)
    raise TimeoutError(f"Server {server_url} not healthy after {timeout}s")



@ray.remote
class SGLangLauncher:
    """
    Ray actor that launches SGLang server on a specific node.
    
    The actual server runs as a subprocess (via launch_server_process),
    but we need to clear CUDA_VISIBLE_DEVICES first since Ray restricts it.
    """

    def __init__(self, node_index: int, local_engine_idx: int = 0):
        self.node_index = node_index
        self.local_engine_idx = local_engine_idx
        self.process = None
        self.server_url = None

    def launch(
        self,
        model_path: str,
        port: int,
        tp_size: int,
        dp_size: int = 1,
        mem_fraction_static: float = 0.88,
        chunked_prefill_size: int = -1,
        context_length: int = None,
    ) -> str:
        """Launch SGLang server and return its URL."""
        # Get actual IP of this node
        host = get_node_ip()
        self.server_url = f"http://{host}:{port}"

        # Assign specific GPUs to this engine based on its local index on the node.
        # E.g., with TP=2: engine 0 gets GPUs 0,1; engine 1 gets GPUs 2,3; etc.
        gpu_start = self.local_engine_idx * tp_size * dp_size
        gpu_end = gpu_start + tp_size * dp_size
        visible_gpus = ",".join(str(g) for g in range(gpu_start, gpu_end))
        os.environ["CUDA_VISIBLE_DEVICES"] = visible_gpus
        logger.info(f"[Node {self.node_index}] CUDA_VISIBLE_DEVICES={visible_gpus}")

        server_args_kwargs = dict(
            model_path=model_path,
            trust_remote_code=True,
            enable_memory_saver=True,
            host=host,
            port=port,
            tp_size=tp_size,
            dp_size=dp_size,
            mem_fraction_static=mem_fraction_static,
            chunked_prefill_size=chunked_prefill_size,
        )
        if context_length is not None:
            server_args_kwargs["context_length"] = context_length

        server_args = ServerArgs(**server_args_kwargs)

        logger.info(f"[Node {self.node_index}] Launching SGLang at {host}:{port} with TP={tp_size}, DP={dp_size}")

        # Use SLIME's existing launch_server_process
        self.process = launch_server_process(server_args)

        logger.info(f"[Node {self.node_index}] Server started at {self.server_url}")
        return self.server_url

    def get_url(self) -> str:
        return self.server_url

    def health_check(self) -> bool:
        if not self.server_url:
            return False
        try:
            # Check both endpoints - router uses /health, SLIME uses /health_generate
            response1 = requests.get(f"{self.server_url}/health", timeout=5)
            response2 = requests.get(f"{self.server_url}/health_generate", timeout=5)
            return response1.status_code == 200 and response2.status_code == 200
        except requests.RequestException:
            return False

    def shutdown(self):
        if self.process and self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=10)
            if self.process.is_alive():
                self.process.kill()


def launch_multi_node_sglang(
    model_path: str,
    num_engines: int | None,
    tp_size: int,
    dp_size: int = 1,
    mem_fraction_static: float = 0.88,
    chunked_prefill_size: int = -1,
    context_length: int = None,
    router_port: int = 30000,
    server_base_port: int = 13140,
    output_dir: str = "/mnt_out/myshang/logs/slime",
) -> dict:
    """
    Launch multiple SGLang engines across nodes.
    
    The workflow is:
    1. Start all SGLang servers first
    2. Wait for all servers to be healthy
    3. Start router with pre-configured worker URLs (avoids HTTP/2 vs HTTP/1.1 issues)
    
    Returns dict with 'launcher_actors' that MUST be kept alive to prevent server shutdown.
    """
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    gpus_per_engine = tp_size * dp_size
    nodes = [n for n in ray.nodes() if n.get("Alive")]
    cluster_gpus = int(ray.cluster_resources().get("GPU", 0))

    # Auto-calculate num_engines
    if num_engines is None or num_engines <= 0:
        num_engines = cluster_gpus // gpus_per_engine
        if num_engines <= 0:
            raise ValueError(f"Not enough GPUs. Have {cluster_gpus}, need {gpus_per_engine} per engine")
        logger.info(f"Auto: num_engines={num_engines} ({cluster_gpus} GPUs / {gpus_per_engine} per engine)")

    total_gpus = num_engines * gpus_per_engine
    router_host = get_node_ip()

    logger.info(f"Launching {num_engines} engines, {gpus_per_engine} GPUs each (across {len(nodes)} nodes)")

    # Step 1: Launch SGLang servers, round-robin across nodes (supports multiple engines per node)
    launcher_actors = []  # Keep references to prevent GC
    engine_addrs = []

    # Track how many engines are assigned to each node for GPU slicing
    node_engine_count = {}
    for i in range(num_engines):
        node = nodes[i % len(nodes)]
        node_id = node["NodeID"]
        server_port = server_base_port + i * 100

        # Which engine is this on the same node? (0th, 1st, 2nd, ...)
        local_engine_idx = node_engine_count.get(node_id, 0)
        node_engine_count[node_id] = local_engine_idx + 1

        logger.info(f"Engine {i}: node={node_id[:8]}..., port={server_port}, local_idx={local_engine_idx}")

        launcher = SGLangLauncher.options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node_id, soft=False),
            num_gpus=0.01,  # Minimal - just to pin to node
            num_cpus=1,
        ).remote(node_index=i, local_engine_idx=local_engine_idx)

        launcher_actors.append((launcher, server_port))

    # Start all servers in parallel
    launch_futures = [
        launcher.launch.remote(
            model_path=model_path,
            port=port,
            tp_size=tp_size,
            dp_size=dp_size,
            mem_fraction_static=mem_fraction_static,
            chunked_prefill_size=chunked_prefill_size,
            context_length=context_length,
        )
        for launcher, port in launcher_actors
    ]

    server_urls = ray.get(launch_futures)
    logger.info(f"All servers started: {server_urls}")

    # Step 2: Wait for all servers to be healthy (in parallel for speed)
    logger.info("Waiting for all servers to be healthy...")
    health_futures = [launcher.health_check.remote() for launcher, _ in launcher_actors]
    
    # Poll until all healthy
    start_time = time.time()
    timeout = 600
    while time.time() - start_time < timeout:
        results = ray.get(health_futures)
        if all(results):
            break
        time.sleep(5)
        health_futures = [launcher.health_check.remote() for launcher, _ in launcher_actors]
    else:
        raise TimeoutError(f"Not all servers healthy after {timeout}s")
    
    logger.info("All servers healthy!")
    
    # Extra wait to ensure servers are fully ready for connections
    time.sleep(5)

    # Extract engine addresses
    for url in server_urls:
        # Parse http://host:port
        parts = url.replace("http://", "").split(":")
        host, port = parts[0], int(parts[1])
        engine_addrs.append(f"{host}:{port}")

    # Step 3: Start router with pre-configured worker URLs
    # This avoids the HTTP/2 vs HTTP/1.1 compatibility issue when adding workers dynamically
    logger.info(f"Starting router with workers: {server_urls}")
    launch_router(router_host, router_port, worker_urls=server_urls)
    wait_router_ready(router_host, router_port)

    # Build result
    result = {
        "router_ip": router_host,
        "router_port": router_port,
        "engine_addrs": engine_addrs,
        "num_engines": num_engines,
        "tp_size": tp_size,
        "dp_size": dp_size,
        "total_gpus": total_gpus,
        "gpus_per_engine": gpus_per_engine,
        "launcher_actors": [launcher for launcher, _ in launcher_actors],  # Keep alive!
    }

    # Write config files
    config_file = os.path.join(output_dir, "sglang_external_rollout.env")
    os.makedirs(output_dir, exist_ok=True)

    with open(config_file, "w") as f:
        f.write(f"SGLANG_ROUTER_IP={router_host}\n")
        f.write(f"SGLANG_ROUTER_PORT={router_port}\n")
        f.write(f"SGLANG_ENGINE_ADDRS=\"{' '.join(engine_addrs)}\"\n")
        f.write(f"SGLANG_NUM_ENGINES={num_engines}\n")
        f.write(f"SGLANG_TOTAL_GPUS={total_gpus}\n")
        f.write(f"SGLANG_GPUS_PER_ENGINE={gpus_per_engine}\n")
        f.write(f"SGLANG_DP_SIZE={dp_size}\n")
    logger.info(f"Config: {config_file}")

    json_file = os.path.join(output_dir, "sglang_external_rollout.json")
    # Exclude launcher_actors from JSON (not serializable)
    json_result = {k: v for k, v in result.items() if k != "launcher_actors"}
    with open(json_file, "w") as f:
        json.dump(json_result, f, indent=2)

    # Print usage
    logger.info("=" * 60)
    logger.info("SGLang External Rollout Ready!")
    logger.info("=" * 60)
    logger.info(f"Router: {router_host}:{router_port}")
    logger.info(f"Engines: {engine_addrs}")
    logger.info("")
    logger.info("SLIME args:")
    logger.info(f"  --rollout-external \\")
    logger.info(f"  --rollout-external-engine-addrs {' '.join(engine_addrs)} \\")
    logger.info(f"  --sglang-router-ip {router_host} \\")
    logger.info(f"  --sglang-router-port {router_port} \\")
    logger.info(f"  --rollout-num-gpus {total_gpus} \\")
    logger.info(f"  --rollout-num-gpus-per-engine {gpus_per_engine}")
    logger.info("")
    logger.info(f"Or: source {config_file}")
    logger.info("=" * 60)

    return result



def main():
    parser = argparse.ArgumentParser(description="Launch multi-node SGLang for SLIME external rollout")
    parser.add_argument("--model-path", type=str, required=True, help="Path to HuggingFace model")
    parser.add_argument("--num-engines", type=int, default=None, help="Number of engines (default: auto)")
    parser.add_argument("--tp-size", type=int, default=8, help="Tensor parallel size (default: 8)")
    parser.add_argument("--dp-size", type=int, default=1, help="Data parallel size (default: 1)")
    parser.add_argument("--mem-fraction-static", type=float, default=0.88, help="Memory fraction (default: 0.88)")
    parser.add_argument("--chunked-prefill-size", type=int, default=-1, help="Chunked prefill size (default: -1, disabled)")
    parser.add_argument("--context-length", type=int, default=None, help="Context length (default: use model config)")
    parser.add_argument("--router-port", type=int, default=30000, help="Router port (default: 30000)")
    parser.add_argument("--server-base-port", type=int, default=13140, help="Server base port (default: 13140)")
    parser.add_argument("--ray-address", type=str, default=None, help="Ray cluster address")
    parser.add_argument("--run-name", type=str, default=None,
                        help="Run name for output directory. Config files are written to "
                             "<output-dir>/<run-name>/sglang_external_rollout.env")
    parser.add_argument("--output-dir", type=str, default="/mnt_out/myshang/logs/slime",
                        help="Base output directory for config files (default: /mnt_out/myshang/logs/slime)")

    args = parser.parse_args()

    # Initialize Ray
    if args.ray_address:
        ray.init(address=args.ray_address)
    elif not ray.is_initialized():
        ray.init(address="auto")

    logger.info(f"Ray cluster: {ray.cluster_resources()}")

    output_dir = args.output_dir
    if args.run_name:
        output_dir = os.path.join(output_dir, args.run_name)

    result = launch_multi_node_sglang(
        model_path=args.model_path,
        num_engines=args.num_engines,
        tp_size=args.tp_size,
        dp_size=args.dp_size,
        mem_fraction_static=args.mem_fraction_static,
        chunked_prefill_size=args.chunked_prefill_size,
        context_length=args.context_length,
        router_port=args.router_port,
        server_base_port=args.server_base_port,
        output_dir=output_dir,
    )

    # IMPORTANT: Keep the script running to keep Ray actors alive
    # If this script exits, the SGLang server processes will be killed
    logger.info("Servers running. Press Ctrl+C to shutdown.")
    launcher_actors = result.get("launcher_actors", [])
    try:
        while True:
            # Periodic health check
            time.sleep(600)
            for i, actor in enumerate(launcher_actors):
                try:
                    healthy = ray.get(actor.health_check.remote(), timeout=10)
                    if not healthy:
                        logger.warning(f"Engine {i} health check failed")
                except Exception as e:
                    logger.warning(f"Engine {i} health check error: {e}")
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        for actor in launcher_actors:
            try:
                ray.get(actor.shutdown.remote(), timeout=30)
            except Exception:
                pass

    return result


if __name__ == "__main__":
    main()
