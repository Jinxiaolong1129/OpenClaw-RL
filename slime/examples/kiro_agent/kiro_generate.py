"""
Multi-turn Generation with Tool Use for SWE-bench Tasks

This module provides a custom generate function for SLIME that enables
multi-turn agent interactions with tool calling in Docker environments,
specifically designed for SWE-bench style tasks.

The implementation handles Docker-in-Docker (DinD) execution where:
1. The SLIME job runs inside a Docker container
2. That container runs Docker daemon (dockerd)
3. SWE-bench instance images are loaded and run as nested containers
4. Tools execute commands inside those nested containers via `docker exec`

Usage:
    --custom-generate-function-path examples.kiro_on_strands.kiro_generate:generate
    --custom-rm-path examples.kiro_on_strands.kiro_generate:reward_func

Environment Variables:
    SWE_DOCKER_IMAGES_PATH: Path to directory containing SWE docker images (tar.gz files)
    IS_SWE_TASK: Set to "true" to enable SWE-specific docker image loading
    WORKSPACE_BASE_PATH: Base path for workspace directories (default: /tmp/workspace)
    REPO_BASE_PATH: Base path for repository files
    MAX_ITERATIONS: Maximum agent iterations (default: 30)
    TIMEOUT_SECONDS: Per-instance timeout (default: 1800)
    DOCKER_BASE_IMAGE: Base docker image for fallback (default: strands_v0.1.4_base_image)
    TRAJECTORY_FOLDER: Path to save trajectory logs
"""

import asyncio
import json
import logging
import os
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.types import Sample

logger = logging.getLogger(__name__)

# Try to import Docker utilities from Kiro-on-Strands
# These provide robust container management with docker SDK
_kiro_strands_path = Path(__file__).parent.parent.parent / "docker" / "Kiro-on-Strands"
if _kiro_strands_path.exists() and str(_kiro_strands_path) not in sys.path:
    sys.path.insert(0, str(_kiro_strands_path))

try:
    from utils.docker_utils import (
        stop_container as _stop_container_sdk,
        preload_swe_docker_images,
        get_patch_from_swe_container,
        start_public_container_workspace,
        setup_public_container_workspace,
        cleanup_swe_docker_resources,
        MAX_DOCKER_CONCURRENCY,
    )
    DOCKER_UTILS_AVAILABLE = True
    logger.info("Loaded docker_utils from Kiro-on-Strands")
except ImportError as e:
    DOCKER_UTILS_AVAILABLE = False
    MAX_DOCKER_CONCURRENCY = 4
    logger.warning(f"Could not import docker_utils from Kiro-on-Strands: {e}")
    logger.warning("Using fallback subprocess-based Docker operations")

# ============================================================================
# Kiro-on-Strands Tool Integration
# ============================================================================

# Try to import tools from Kiro-on-Strands
KIRO_TOOLS_AVAILABLE = False
KIRO_TOOLS = {}

try:
    from tools.executeBash import executeBash, TOOL_SPEC as EXECUTE_BASH_SPEC
    from tools.readFile import readFile, TOOL_SPEC as READ_FILE_SPEC
    from tools.readMultipleFiles import readMultipleFiles, TOOL_SPEC as READ_MULTIPLE_FILES_SPEC
    from tools.fsWrite import fsWrite, TOOL_SPEC as FS_WRITE_SPEC
    from tools.fsAppend import fsAppend, TOOL_SPEC as FS_APPEND_SPEC
    from tools.strReplace import strReplace, TOOL_SPEC as STR_REPLACE_SPEC
    from tools.listDirectory import listDirectory, TOOL_SPEC as LIST_DIRECTORY_SPEC
    from tools.grepSearch import grepSearch, TOOL_SPEC as GREP_SEARCH_SPEC
    from tools.fileSearch import fileSearch, TOOL_SPEC as FILE_SEARCH_SPEC
    from tools.deleteFile import deleteFile, TOOL_SPEC as DELETE_FILE_SPEC
    
    # Tool registry mapping name -> (function, spec)
    KIRO_TOOLS = {
        "executeBash": (executeBash, EXECUTE_BASH_SPEC),
        "readFile": (readFile, READ_FILE_SPEC),
        "readMultipleFiles": (readMultipleFiles, READ_MULTIPLE_FILES_SPEC),
        "fsWrite": (fsWrite, FS_WRITE_SPEC),
        "fsAppend": (fsAppend, FS_APPEND_SPEC),
        "strReplace": (strReplace, STR_REPLACE_SPEC),
        "listDirectory": (listDirectory, LIST_DIRECTORY_SPEC),
        "grepSearch": (grepSearch, GREP_SEARCH_SPEC),
        "fileSearch": (fileSearch, FILE_SEARCH_SPEC),
        "deleteFile": (deleteFile, DELETE_FILE_SPEC),
    }
    KIRO_TOOLS_AVAILABLE = True
    logger.info(f"Loaded {len(KIRO_TOOLS)} tools from Kiro-on-Strands: {list(KIRO_TOOLS.keys())}")
except ImportError as e:
    logger.warning(f"Could not import tools from Kiro-on-Strands: {e}")
    logger.warning("Using fallback simple tool implementations")

# Try to import system prompts from Kiro-on-Strands
KIRO_SYSTEM_PROMPT = None
KIRO_INSTRUCTION_TEMPLATE = None
_kiro_get_file_tree_from_container = None
try:
    from prompts.vibe.system_prompts import get_kiro_prod_base_message, VIBE_SYSTEM_PROMPT
    from prompts.vibe.instruction import KIRO_BENCHMARKING_INSTRUCTION
    from utils.shard import get_file_tree_from_container as _kiro_get_file_tree_from_container
    KIRO_SYSTEM_PROMPT = VIBE_SYSTEM_PROMPT  # Use VIBE_SYSTEM_PROMPT for SWE tasks
    KIRO_INSTRUCTION_TEMPLATE = KIRO_BENCHMARKING_INSTRUCTION
    logger.info("Loaded system prompts and utilities from Kiro-on-Strands")
except ImportError as e:
    logger.warning(f"Could not import system prompts from Kiro-on-Strands: {e}")
    logger.warning("Using fallback system prompt")


def get_kiro_tool_definitions() -> list[dict]:
    """Get OpenAI-format tool definitions for all Kiro tools."""
    if not KIRO_TOOLS_AVAILABLE:
        return []
    
    tools = []
    for name, (_, spec) in KIRO_TOOLS.items():
        tools.append({
            "type": "function",
            "function": {
                "name": spec["name"],
                "description": spec["description"],
                "parameters": spec["inputSchema"],
            },
        })
    return tools


def execute_kiro_tool(
    tool_name: str,
    arguments: dict,
    container_id: str,
    container_workspace: str,
) -> dict:
    """
    Execute a Kiro-on-Strands tool.
    
    Args:
        tool_name: Name of the tool (e.g., "executeBash", "readFile")
        arguments: Tool arguments from model output
        container_id: Docker container ID
        container_workspace: Workspace path inside container (e.g., "/testbed")
    
    Returns:
        ToolResult dict with status and content
    """
    if tool_name not in KIRO_TOOLS:
        return {
            "toolUseId": str(uuid.uuid4()),
            "status": "error",
            "content": [{"text": f"Unknown tool: {tool_name}"}],
        }
    
    tool_func, _ = KIRO_TOOLS[tool_name]
    
    # Build ToolUse structure with injected container info
    tool_use = {
        "toolUseId": str(uuid.uuid4()),
        "name": tool_name,
        "input": {
            **arguments,
            "container_id": container_id,
            "container_workspace": container_workspace,
        },
    }
    
    # Execute the tool
    try:
        result = tool_func(tool_use)
        return result
    except Exception as e:
        logger.error(f"Tool execution error for {tool_name}: {e}")
        return {
            "toolUseId": tool_use["toolUseId"],
            "status": "error",
            "content": [{"text": f"Tool execution error: {str(e)}"}],
        }


def format_kiro_tool_result(result: dict) -> str:
    """Extract text from Kiro tool result for model consumption."""
    content = result.get("content", [])
    status = result.get("status", "unknown")
    
    if not content:
        return f"Tool returned status: {status}"
    
    parts = []
    for item in content:
        if "text" in item:
            parts.append(item["text"])
        elif "json" in item:
            # Format JSON output (e.g., from executeBash, grepSearch)
            json_data = item["json"]
            if "stdout" in json_data:
                if json_data["stdout"]:
                    parts.append(json_data["stdout"])
                if json_data.get("stderr"):
                    parts.append(f"STDERR: {json_data['stderr']}")
                if json_data.get("exit_status", 0) != 0:
                    parts.append(f"Exit code: {json_data['exit_status']}")
            else:
                parts.append(json.dumps(json_data, indent=2))
    
    return "\n".join(parts) if parts else f"Status: {status}"


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class KiroConfig:
    """Configuration for Kiro multi-turn generation."""
    # Generation limits
    max_turns: int = 15
    max_iterations: int = 30
    tool_timeout: float = 120.0  # seconds per tool call
    timeout_seconds: int = 1800  # per-instance timeout

    # System prompt - use Kiro-on-Strands prompt if available
    system_prompt: str = ""  # Will be set after class definition

    # Instruction template for SWE tasks
    instruction_template: str = ""  # Will be set after class definition

    # Docker settings
    docker_base_image: str = "strands_v0.1.4_base_image"
    docker_timeout: float = 7200.0  # container lifetime (2 hours)
    docker_load_timeout: float = 600.0  # timeout for loading docker images
    max_docker_concurrency: int = 4

    # SWE Docker image preloading
    swe_docker_images_path: str | None = None
    is_swe_task: bool = True

    # Workspace settings
    workspace_base_path: str = "/tmp/workspace"
    repo_base_path: str | None = None

    # Reward settings
    format_reward: float = 0.1
    tool_use_reward: float = 0.05
    max_tool_bonus: float = 0.3

    # Trajectory logging settings
    trajectory_folder: str | None = None
    save_trajectories: bool = True
    save_detailed_logs: bool = True


# Fallback system prompt if Kiro-on-Strands prompts not available
_FALLBACK_SYSTEM_PROMPT = """You are a helpful AI assistant that can interact with a computer to solve tasks.

<ROLE>
Your primary role is to assist users by executing commands, modifying code, and solving technical problems effectively. You should be thorough, methodical, and prioritize quality over speed.
</ROLE>

<EFFICIENCY>
* Each action you take is somewhat expensive. Wherever possible, combine multiple actions into a single action.
* When exploring the codebase, use efficient tools like find, grep, and git commands with appropriate filters.
</EFFICIENCY>

<FILE_SYSTEM_GUIDELINES>
* When a user provides a file path, first explore the file system to locate the file before working on it.
* If asked to edit a file, edit the file directly, rather than creating a new file with a different filename.
</FILE_SYSTEM_GUIDELINES>

<CODE_QUALITY>
* Write clean, efficient code with minimal comments.
* When implementing solutions, focus on making the minimal changes needed to solve the problem.
* Before implementing any changes, first thoroughly understand the codebase through exploration.
</CODE_QUALITY>

<PROBLEM_SOLVING_WORKFLOW>
1. EXPLORATION: Thoroughly explore relevant files and understand the context before proposing solutions
2. ANALYSIS: Consider multiple approaches and select the most promising one
3. IMPLEMENTATION: Make focused, minimal changes to address the problem
4. VERIFICATION: Test your implementation thoroughly
</PROBLEM_SOLVING_WORKFLOW>
"""

_FALLBACK_INSTRUCTION_TEMPLATE = """
Make the source file and test file updates for the following issue:

<GITHUB_ISSUE>
{pr_description}
</GITHUB_ISSUE>
"""

CONFIG = KiroConfig()

# Set system prompt - use Kiro-on-Strands prompt if available
CONFIG.system_prompt = KIRO_SYSTEM_PROMPT if KIRO_SYSTEM_PROMPT else _FALLBACK_SYSTEM_PROMPT
CONFIG.instruction_template = KIRO_INSTRUCTION_TEMPLATE if KIRO_INSTRUCTION_TEMPLATE else _FALLBACK_INSTRUCTION_TEMPLATE

# Initialize from environment variables
CONFIG.swe_docker_images_path = os.environ.get("SWE_DOCKER_IMAGES_PATH")
CONFIG.is_swe_task = os.environ.get("IS_SWE_TASK", "true").lower() in ("true", "1", "yes")
CONFIG.workspace_base_path = os.environ.get("WORKSPACE_BASE_PATH", "/tmp/workspace")
CONFIG.repo_base_path = os.environ.get("REPO_BASE_PATH")
CONFIG.max_iterations = int(os.environ.get("MAX_ITERATIONS", "30"))
CONFIG.timeout_seconds = int(os.environ.get("TIMEOUT_SECONDS", "1800"))
CONFIG.docker_base_image = os.environ.get("DOCKER_BASE_IMAGE", "strands_v0.1.4_base_image")
CONFIG.max_docker_concurrency = int(os.environ.get("MAX_DOCKER_CONCURRENCY", "4"))
CONFIG.trajectory_folder = os.environ.get("TRAJECTORY_FOLDER")
CONFIG.save_trajectories = os.environ.get("SAVE_TRAJECTORIES", "true").lower() in ("true", "1", "yes")
CONFIG.save_detailed_logs = os.environ.get("SAVE_DETAILED_LOGS", "true").lower() in ("true", "1", "yes")

# Track loaded docker images to avoid reloading
_loaded_docker_images: set[str] = set()
_docker_images_lock = threading.Lock()

# Semaphore for Docker concurrency control
_docker_semaphore: Optional[threading.Semaphore] = None


# ============================================================================
# Trajectory Logging
# ============================================================================

@dataclass
class TurnMetrics:
    """Metrics for a single turn in the conversation."""
    turn_idx: int
    role: str  # "assistant" or "tool"
    tokens_generated: int = 0
    tool_calls_made: int = 0
    tool_names: list = None
    duration_seconds: float = 0.0
    finish_reason: str = ""
    
    def __post_init__(self):
        if self.tool_names is None:
            self.tool_names = []
    
    def to_dict(self) -> dict:
        return {
            "turn_idx": self.turn_idx,
            "role": self.role,
            "tokens_generated": self.tokens_generated,
            "tool_calls_made": self.tool_calls_made,
            "tool_names": self.tool_names,
            "duration_seconds": round(self.duration_seconds, 3),
            "finish_reason": self.finish_reason,
        }


@dataclass
class TrajectoryMetrics:
    """Aggregate metrics for a complete trajectory."""
    instance_id: str
    rollout_idx: int = 0
    total_turns: int = 0
    total_tokens: int = 0
    prompt_tokens: int = 0
    response_tokens: int = 0
    total_tool_calls: int = 0
    unique_tools_used: set = None
    total_duration_seconds: float = 0.0
    status: str = ""
    patch_size: int = 0
    patch_lines: int = 0
    error_message: str = ""
    
    def __post_init__(self):
        if self.unique_tools_used is None:
            self.unique_tools_used = set()
    
    def to_dict(self) -> dict:
        return {
            "instance_id": self.instance_id,
            "rollout_idx": self.rollout_idx,
            "total_turns": self.total_turns,
            "total_tokens": self.total_tokens,
            "prompt_tokens": self.prompt_tokens,
            "response_tokens": self.response_tokens,
            "total_tool_calls": self.total_tool_calls,
            "unique_tools_used": list(self.unique_tools_used),
            "total_duration_seconds": round(self.total_duration_seconds, 3),
            "status": self.status,
            "patch_size": self.patch_size,
            "patch_lines": self.patch_lines,
            "error_message": self.error_message,
        }


class TrajectoryLogger:
    """
    Logger for saving trajectories and metrics during generation.
    
    Saves:
    - Full conversation messages (system, user, assistant, tool results)
    - Per-turn metrics (tokens, tool calls, duration)
    - Overall trajectory metrics
    - Final patch
    """
    
    def __init__(
        self,
        instance_id: str,
        rollout_idx: int = 0,
        trajectory_folder: str | None = None,
    ):
        self.instance_id = instance_id
        self.rollout_idx = rollout_idx
        self.trajectory_folder = trajectory_folder or CONFIG.trajectory_folder
        
        # Conversation history
        self.messages: list[dict] = []
        
        # Per-turn metrics
        self.turn_metrics: list[TurnMetrics] = []
        
        # Aggregate metrics
        self.metrics = TrajectoryMetrics(
            instance_id=instance_id,
            rollout_idx=rollout_idx,
        )
        
        # Timing
        self.start_time: float | None = None
        self.turn_start_time: float | None = None
    
    def start(self):
        """Mark the start of generation."""
        self.start_time = time.time()
        logger.info(f"[TRAJECTORY] Starting generation for {self.instance_id} (rollout {self.rollout_idx})")
    
    def add_system_message(self, content: str):
        """Add system message to trajectory."""
        self.messages.append({
            "role": "system",
            "content": content,
            "timestamp": time.time(),
        })
    
    def add_user_message(self, content: str):
        """Add user message to trajectory."""
        self.messages.append({
            "role": "user",
            "content": content,
            "timestamp": time.time(),
        })
    
    def start_turn(self):
        """Mark the start of a new turn."""
        self.turn_start_time = time.time()
    
    def add_assistant_message(
        self,
        content: str,
        tokens: list[int],
        log_probs: list[float],
        finish_reason: str = "",
        tool_calls: list[dict] | None = None,
    ):
        """Add assistant message with metrics."""
        turn_duration = time.time() - self.turn_start_time if self.turn_start_time else 0.0
        
        # Create message entry
        message = {
            "role": "assistant",
            "content": content,
            "timestamp": time.time(),
            "tokens_count": len(tokens),
            "finish_reason": finish_reason,
        }
        
        if tool_calls:
            message["tool_calls"] = tool_calls
        
        self.messages.append(message)
        
        # Create turn metrics
        turn_idx = len(self.turn_metrics)
        tool_names = [tc["name"] for tc in tool_calls] if tool_calls else []
        
        turn_metric = TurnMetrics(
            turn_idx=turn_idx,
            role="assistant",
            tokens_generated=len(tokens),
            tool_calls_made=len(tool_calls) if tool_calls else 0,
            tool_names=tool_names,
            duration_seconds=turn_duration,
            finish_reason=finish_reason,
        )
        self.turn_metrics.append(turn_metric)
        
        # Update aggregate metrics
        self.metrics.response_tokens += len(tokens)
        self.metrics.total_tool_calls += len(tool_calls) if tool_calls else 0
        for name in tool_names:
            self.metrics.unique_tools_used.add(name)
        
        logger.debug(
            f"[TRAJECTORY] Turn {turn_idx}: {len(tokens)} tokens, "
            f"{len(tool_calls) if tool_calls else 0} tool calls, "
            f"{turn_duration:.2f}s"
        )
    
    def add_tool_result(self, tool_name: str, tool_id: str, result: str, duration: float = 0.0):
        """Add tool execution result."""
        self.messages.append({
            "role": "tool",
            "tool_name": tool_name,
            "tool_call_id": tool_id,
            "content": result[:10000] if len(result) > 10000 else result,  # Truncate very long results
            "timestamp": time.time(),
            "duration_seconds": round(duration, 3),
        })
    
    def set_prompt_tokens(self, count: int):
        """Set the prompt token count."""
        self.metrics.prompt_tokens = count
    
    def finalize(
        self,
        status: str,
        patch: str = "",
        error_message: str = "",
    ):
        """Finalize the trajectory with final metrics."""
        self.metrics.total_duration_seconds = time.time() - self.start_time if self.start_time else 0.0
        self.metrics.total_turns = len([m for m in self.messages if m["role"] == "assistant"])
        self.metrics.total_tokens = self.metrics.prompt_tokens + self.metrics.response_tokens
        self.metrics.status = status
        self.metrics.patch_size = len(patch)
        self.metrics.patch_lines = len(patch.split("\n")) if patch else 0
        self.metrics.error_message = error_message
        
        logger.info(
            f"[TRAJECTORY] Finalized {self.instance_id}: "
            f"status={status}, turns={self.metrics.total_turns}, "
            f"tokens={self.metrics.total_tokens}, tools={self.metrics.total_tool_calls}, "
            f"patch_lines={self.metrics.patch_lines}, duration={self.metrics.total_duration_seconds:.1f}s"
        )
    
    def save(self) -> str | None:
        """
        Save trajectory to JSONL file.
        
        Returns:
            Path to saved file, or None if saving is disabled
        """
        if not CONFIG.save_trajectories:
            return None
        
        if not self.trajectory_folder:
            logger.warning("[TRAJECTORY] No trajectory folder configured, skipping save")
            return None
        
        # Create output directory
        output_dir = Path(self.trajectory_folder)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Build trajectory record
        trajectory = {
            "instance_id": self.instance_id,
            "rollout_idx": self.rollout_idx,
            "metrics": self.metrics.to_dict(),
            "turn_metrics": [tm.to_dict() for tm in self.turn_metrics],
            "messages": self.messages if CONFIG.save_detailed_logs else [],
            "timestamp": time.time(),
        }
        
        # Save to JSONL file
        output_file = output_dir / f"{self.instance_id}_rollout{self.rollout_idx}.jsonl"
        
        try:
            with open(output_file, "w") as f:
                f.write(json.dumps(trajectory) + "\n")
            
            logger.info(f"[TRAJECTORY] Saved to {output_file}")
            return str(output_file)
            
        except Exception as e:
            logger.error(f"[TRAJECTORY] Failed to save: {e}")
            return None
    
    def get_summary(self) -> dict:
        """Get a summary of the trajectory for logging."""
        return {
            "instance_id": self.instance_id,
            "rollout_idx": self.rollout_idx,
            "status": self.metrics.status,
            "total_turns": self.metrics.total_turns,
            "total_tokens": self.metrics.total_tokens,
            "total_tool_calls": self.metrics.total_tool_calls,
            "unique_tools": list(self.metrics.unique_tools_used),
            "duration_seconds": round(self.metrics.total_duration_seconds, 2),
            "patch_lines": self.metrics.patch_lines,
        }



def aggregate_trajectory_metrics(trajectory_folder: str) -> dict:
    """
    Aggregate metrics from all trajectory files in a folder.
    
    Args:
        trajectory_folder: Path to folder containing trajectory JSONL files
        
    Returns:
        Dictionary with aggregated statistics
    """
    from pathlib import Path
    
    folder = Path(trajectory_folder)
    if not folder.exists():
        return {"error": f"Folder not found: {trajectory_folder}"}
    
    # Collect all trajectory files
    trajectory_files = list(folder.glob("*.jsonl"))
    
    if not trajectory_files:
        return {"error": "No trajectory files found", "folder": trajectory_folder}
    
    # Aggregate metrics
    total_trajectories = 0
    status_counts = {}
    total_tokens = 0
    total_tool_calls = 0
    total_turns = 0
    total_duration = 0.0
    patch_counts = {"with_patch": 0, "empty_patch": 0}
    tool_usage = {}
    
    for traj_file in trajectory_files:
        try:
            with open(traj_file, "r") as f:
                for line in f:
                    if not line.strip():
                        continue
                    traj = json.loads(line)
                    metrics = traj.get("metrics", {})
                    
                    total_trajectories += 1
                    
                    # Status counts
                    status = metrics.get("status", "UNKNOWN")
                    status_counts[status] = status_counts.get(status, 0) + 1
                    
                    # Token counts
                    total_tokens += metrics.get("total_tokens", 0)
                    
                    # Tool calls
                    total_tool_calls += metrics.get("total_tool_calls", 0)
                    for tool in metrics.get("unique_tools_used", []):
                        tool_usage[tool] = tool_usage.get(tool, 0) + 1
                    
                    # Turns
                    total_turns += metrics.get("total_turns", 0)
                    
                    # Duration
                    total_duration += metrics.get("total_duration_seconds", 0)
                    
                    # Patch stats
                    if metrics.get("patch_size", 0) > 0:
                        patch_counts["with_patch"] += 1
                    else:
                        patch_counts["empty_patch"] += 1
                        
        except Exception as e:
            logger.warning(f"Error reading {traj_file}: {e}")
    
    if total_trajectories == 0:
        return {"error": "No valid trajectories found"}
    
    return {
        "total_trajectories": total_trajectories,
        "status_distribution": status_counts,
        "avg_tokens_per_trajectory": round(total_tokens / total_trajectories, 1),
        "avg_tool_calls_per_trajectory": round(total_tool_calls / total_trajectories, 2),
        "avg_turns_per_trajectory": round(total_turns / total_trajectories, 2),
        "avg_duration_seconds": round(total_duration / total_trajectories, 2),
        "total_duration_seconds": round(total_duration, 2),
        "patch_stats": patch_counts,
        "tool_usage": tool_usage,
        "trajectory_folder": trajectory_folder,
    }


def _get_docker_semaphore() -> threading.Semaphore:
    """Get or create the Docker concurrency semaphore."""
    global _docker_semaphore
    if _docker_semaphore is None:
        _docker_semaphore = threading.Semaphore(CONFIG.max_docker_concurrency)
    return _docker_semaphore

# ============================================================================
# Workspace Management
# ============================================================================

@dataclass
class Workspace:
    """Represents a Docker workspace for agent interaction."""
    container_id: str | None = None
    container_name: str = ""
    workdir: str = "/testbed"
    local_workspace: str = ""
    instance_id: str = ""
    is_active: bool = False
    image_preloaded: bool = False


def start_swe_container(
    instance_id: str,
    local_workspace: Path,
    repo_path: str | None = None,
    base_commit: str | None = None,
) -> str:
    """
    Start a Docker container for SWE-bench task.
    
    Args:
        instance_id: Unique identifier for the instance
        local_workspace: Local path to mount as workspace
        repo_path: Path to the repository to copy into container
        base_commit: Git commit to checkout (for public SWE tasks)
        
    Returns:
        Container ID
    """
    container_name = f"swe.strands.{instance_id}_{uuid.uuid4().hex[:8]}"
    
    # Stop any existing container with similar name
    subprocess.run(
        ["docker", "rm", "-f", f"swe.strands.{instance_id}"],
        capture_output=True,
        timeout=30,
    )
    
    # Determine which image to use
    # Check if instance-specific image exists (preloaded)
    check_result = subprocess.run(
        ["docker", "images", "-q", f"{instance_id}:latest"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    
    if check_result.stdout.strip():
        image_name = f"{instance_id}:latest"
        logger.info(f"Using preloaded image: {image_name}")
    else:
        image_name = CONFIG.docker_base_image
        logger.info(f"Using base image: {image_name}")
    
    # Prepare workspace directory
    local_workspace.mkdir(parents=True, exist_ok=True)
    
    # Copy repo to workspace if provided
    if repo_path and os.path.exists(repo_path):
        subprocess.run(
            ["cp", "-r", f"{repo_path}/.", str(local_workspace)],
            capture_output=True,
            timeout=120,
        )
        logger.info(f"Copied repo from {repo_path} to {local_workspace}")
    
    # Build docker run command
    docker_cmd = [
        "docker", "run", "-d",
        "--name", container_name,
        "-v", f"{local_workspace}:/testbed",
    ]
    
    # Git initialization command
    if base_commit:
        init_cmd = (
            f"cd /testbed && "
            f"git config --global user.email a && "
            f"git config --global user.name a && "
            f"git config --global --add safe.directory /testbed && "
            f"git reset --hard && git checkout {base_commit} && "
            f"sleep {int(CONFIG.docker_timeout)}"
        )
    else:
        init_cmd = (
            f"cd /testbed && "
            f"git init && "
            f"git config --global user.email a && "
            f"git config --global user.name a && "
            f"git config --global --add safe.directory /testbed && "
            f"git add . && git commit --allow-empty -am kiro-on-strands && "
            f"sleep {int(CONFIG.docker_timeout)}"
        )
    
    docker_cmd.extend([image_name, "bash", "-c", init_cmd])
    
    result = subprocess.run(
        docker_cmd,
        capture_output=True,
        text=True,
        timeout=60,
    )
    
    if result.returncode != 0:
        logger.error(f"Failed to start container: {result.stderr}")
        raise RuntimeError(f"Failed to start container: {result.stderr}")
    
    container_id = result.stdout.strip()
    
    # Wait for container to be ready
    time.sleep(5)
    
    logger.info(f"Started container {container_id[:12]} for {instance_id}")
    return container_id


def stop_container(container_id: str):
    """
    Stop and remove a Docker container.
    
    Uses _stop_container_sdk from Kiro-on-Strands if available,
    otherwise falls back to subprocess.
    """
    # Use Kiro-on-Strands utility if available (uses docker SDK)
    if DOCKER_UTILS_AVAILABLE:
        try:
            _stop_container_sdk(container_id)
            return
        except Exception as e:
            logger.warning(f"_stop_container_sdk failed, using fallback: {e}")
    
    # Fallback to subprocess
    try:
        subprocess.run(
            ["docker", "stop", container_id],
            capture_output=True,
            timeout=30,
        )
        subprocess.run(
            ["docker", "rm", "-f", container_id],
            capture_output=True,
            timeout=30,
        )
        logger.debug(f"Stopped container {container_id[:12]}")
    except Exception as e:
        logger.warning(f"Failed to stop container {container_id}: {e}")


async def setup_workspace(sample: Sample, rollout_idx: int = 0) -> Workspace:
    """
    Setup Docker workspace for the sample.
    
    Args:
        sample: Sample with metadata (instance_id, repo, base_commit, etc.)
        rollout_idx: Index of the rollout for this sample
        
    Returns:
        Workspace object with container info
    """
    instance_id = sample.metadata.get("instance_id", f"instance_{sample.index}")
    repo = sample.metadata.get("repo", "")
    base_commit = sample.metadata.get("base_commit")
    
    workspace = Workspace(
        instance_id=instance_id,
        workdir="/testbed",
    )
    
    # Preload Docker image if SWE task
    if CONFIG.is_swe_task and CONFIG.swe_docker_images_path:
        workspace.image_preloaded = preload_swe_docker_images(instance_id)
    
    # Setup local workspace path
    local_workspace = Path(CONFIG.workspace_base_path) / instance_id / f"rollout_{rollout_idx}"
    workspace.local_workspace = str(local_workspace)
    
    # Determine repo path
    repo_path = None
    if CONFIG.repo_base_path and repo:
        if base_commit:
            repo_path = os.path.join(CONFIG.repo_base_path, repo, base_commit)
        else:
            repo_path = os.path.join(CONFIG.repo_base_path, repo)
        
        if not os.path.exists(repo_path):
            repo_path = os.path.join(CONFIG.repo_base_path, repo)
    
    # Start container with semaphore for concurrency control
    semaphore = _get_docker_semaphore()
    with semaphore:
        try:
            container_id = start_swe_container(
                instance_id=instance_id,
                local_workspace=local_workspace,
                repo_path=repo_path,
                base_commit=base_commit,
            )
            workspace.container_id = container_id
            workspace.is_active = True
        except Exception as e:
            logger.error(f"Failed to setup workspace for {instance_id}: {e}")
            workspace.is_active = False
    
    return workspace


async def cleanup_workspace(workspace: Workspace):
    """Cleanup Docker workspace after generation."""
    if workspace.container_id:
        stop_container(workspace.container_id)


def cleanup_all_swe_containers():
    """
    Cleanup all SWE-related Docker containers.
    
    Useful for cleaning up after batch processing or on error recovery.
    Uses cleanup_swe_docker_resources from Kiro-on-Strands if available.
    """
    if DOCKER_UTILS_AVAILABLE:
        try:
            cleanup_swe_docker_resources()
            return
        except Exception as e:
            logger.warning(f"cleanup_swe_docker_resources failed: {e}")
    
    # Fallback: list and remove containers with swe.strands prefix
    try:
        result = subprocess.run(
            ["docker", "ps", "-a", "--format", "{{.Names}}"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        for name in result.stdout.strip().split("\n"):
            if name and "swe.strands" in name:
                subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=30)
                logger.debug(f"Removed container {name}")
    except Exception as e:
        logger.error(f"Failed to cleanup containers: {e}")


# ============================================================================
# Docker Command Execution
# ============================================================================

async def docker_exec(workspace: Workspace, command: str, timeout: float = None) -> str:
    """
    Execute a command inside the Docker container.
    
    Args:
        workspace: Workspace with container info
        command: Shell command to execute
        timeout: Timeout in seconds (default: CONFIG.tool_timeout)
        
    Returns:
        Command output (stdout + stderr)
    """
    if not workspace.is_active or not workspace.container_id:
        return "Error: Workspace not available"
    
    timeout = timeout or CONFIG.tool_timeout
    
    try:
        proc = await asyncio.create_subprocess_exec(
            "docker", "exec",
            "-w", workspace.workdir,
            workspace.container_id,
            "bash", "-c", command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(),
            timeout=timeout,
        )
        
        output = stdout.decode("utf-8", errors="replace")
        if stderr:
            stderr_text = stderr.decode("utf-8", errors="replace")
            if stderr_text.strip():
                output += f"\nSTDERR: {stderr_text}"
        
        # Truncate very long outputs
        if len(output) > 50000:
            output = output[:50000] + "\n... [output truncated]"
        
        return output
        
    except asyncio.TimeoutError:
        return f"Error: Command timed out after {timeout}s"
    except Exception as e:
        return f"Error: {type(e).__name__}: {str(e)}"


def get_patch_from_container(workspace: Workspace) -> str:
    """
    Extract git diff patch from the container's workspace.
    
    Uses get_patch_from_swe_container from Kiro-on-Strands if available,
    otherwise falls back to subprocess.
    
    Returns:
        Git diff as a string
    """
    if not workspace.container_id:
        return ""
    
    # Use Kiro-on-Strands utility if available (uses docker SDK)
    if DOCKER_UTILS_AVAILABLE:
        try:
            return get_patch_from_swe_container(workspace.container_id, workspace.workdir)
        except Exception as e:
            logger.warning(f"get_patch_from_swe_container failed, using fallback: {e}")
    
    # Fallback to subprocess
    try:
        cmd = (
            f"cd {workspace.workdir} && "
            f"git add -A && "
            f"git --no-pager diff -U5 --no-color --cached HEAD"
        )
        
        result = subprocess.run(
            ["docker", "exec", "-w", workspace.workdir, workspace.container_id, "bash", "-c", cmd],
            capture_output=True,
            text=True,
            timeout=60,
        )
        
        if result.returncode != 0:
            logger.warning(f"Git diff failed: {result.stderr}")
            return ""
        
        return result.stdout
        
    except Exception as e:
        logger.error(f"Failed to get patch: {e}")
        return ""


def get_file_tree_from_container(workspace: Workspace, max_depth: int = 3) -> str:
    """
    Get file tree from the container's workspace.
    
    Uses Kiro-on-Strands implementation if available (returns structured XML format),
    otherwise falls back to simple find command.
    
    Returns:
        File tree as a string (XML format if Kiro tools available, plain text otherwise)
    """
    if not workspace.container_id:
        return ""
    
    # Use Kiro-on-Strands implementation if available (uses Docker SDK, returns XML format)
    if _kiro_get_file_tree_from_container is not None:
        try:
            return _kiro_get_file_tree_from_container(
                container_id=workspace.container_id,
                workdir=workspace.workdir,
                target=500
            )
        except Exception as e:
            logger.warning(f"Kiro get_file_tree_from_container failed, using fallback: {e}")
    
    # Fallback to subprocess-based implementation
    try:
        cmd = f"find {workspace.workdir} -maxdepth {max_depth} -type f -o -type d | head -500"
        
        result = subprocess.run(
            ["docker", "exec", workspace.container_id, "bash", "-c", cmd],
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        return result.stdout if result.returncode == 0 else ""
        
    except Exception as e:
        logger.warning(f"Failed to get file tree: {e}")
        return ""


# ============================================================================
# Tool Definitions and Execution
# ============================================================================

# Fallback simple tool definitions (used when Kiro tools not available)
_FALLBACK_TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "execute_command",
            "description": "Execute a shell command in the workspace",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to execute",
                    },
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file (relative to workspace)",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file (relative to workspace)",
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write to the file",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "List files and directories in a path",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory path (relative to workspace)",
                        "default": ".",
                    },
                },
            },
        },
    },
]

# Use Kiro tools if available, otherwise fallback
TOOL_DEFINITIONS = get_kiro_tool_definitions() if KIRO_TOOLS_AVAILABLE else _FALLBACK_TOOL_DEFINITIONS

# Mapping from simple tool names to Kiro tool names (for backward compatibility)
_TOOL_NAME_MAP = {
    "execute_command": "executeBash",
    "read_file": "readFile",
    "write_file": "fsWrite",
    "list_directory": "listDirectory",
}

# Mapping for argument names that differ between simple and Kiro tools
_TOOL_ARG_MAP = {
    "fsWrite": {"content": "text"},  # write_file uses "content", fsWrite uses "text"
}


async def execute_tool(workspace: Workspace, tool_name: str, arguments: dict) -> str:
    """
    Execute a tool in the Docker workspace.
    
    Uses Kiro-on-Strands tools if available, otherwise falls back to simple implementations.
    Supports both Kiro tool names (executeBash, readFile, etc.) and simple names
    (execute_command, read_file, etc.) for backward compatibility.
    """
    if not workspace.is_active or not workspace.container_id:
        return "Error: Workspace not available"
    
    # Use Kiro tools if available
    if KIRO_TOOLS_AVAILABLE:
        # Map simple tool names to Kiro tool names if needed
        kiro_tool_name = _TOOL_NAME_MAP.get(tool_name, tool_name)
        
        # Map arguments if needed
        mapped_args = arguments.copy()
        if kiro_tool_name in _TOOL_ARG_MAP:
            for old_key, new_key in _TOOL_ARG_MAP[kiro_tool_name].items():
                if old_key in mapped_args:
                    mapped_args[new_key] = mapped_args.pop(old_key)
        
        # Execute using Kiro tool
        result = execute_kiro_tool(
            tool_name=kiro_tool_name,
            arguments=mapped_args,
            container_id=workspace.container_id,
            container_workspace=workspace.workdir,
        )
        
        return format_kiro_tool_result(result)
    
    # Fallback to simple implementations
    if tool_name == "execute_command" or tool_name == "executeBash":
        command = arguments.get("command", "")
        if not command:
            return "Error: No command provided"
        return await docker_exec(workspace, command)
    
    elif tool_name == "read_file" or tool_name == "readFile":
        path = arguments.get("path", "")
        if not path:
            return "Error: No path provided"
        return await docker_exec(workspace, f"cat '{path}'")
    
    elif tool_name == "write_file" or tool_name == "fsWrite":
        path = arguments.get("path", "")
        content = arguments.get("content", arguments.get("text", ""))
        if not path:
            return "Error: No path provided"
        # Use heredoc for safe content writing
        return await docker_exec(
            workspace,
            f"cat > '{path}' << 'KIRO_EOF'\n{content}\nKIRO_EOF"
        )
    
    elif tool_name == "list_directory" or tool_name == "listDirectory":
        path = arguments.get("path", ".")
        explanation = arguments.get("explanation", "Listing directory")
        return await docker_exec(workspace, f"ls -la '{path}'")
    
    elif tool_name == "grepSearch":
        query = arguments.get("query", "")
        if not query:
            return "Error: No query provided"
        include_pattern = arguments.get("includePattern", "")
        cmd = f"rg '{query}' . --max-count 50 -C 2"
        if include_pattern:
            cmd += f" --glob '{include_pattern}'"
        return await docker_exec(workspace, cmd)
    
    elif tool_name == "fileSearch":
        query = arguments.get("query", "")
        if not query:
            return "Error: No query provided"
        return await docker_exec(workspace, f"find . -name '*{query}*' | head -20")
    
    elif tool_name == "strReplace":
        path = arguments.get("path", "")
        old_str = arguments.get("oldStr", "")
        new_str = arguments.get("newStr", "")
        if not path or not old_str:
            return "Error: path and oldStr are required"
        # Read file, replace, write back
        content = await docker_exec(workspace, f"cat '{path}'")
        if "Error:" in content:
            return content
        new_content = content.replace(old_str, new_str, 1)
        return await docker_exec(
            workspace,
            f"cat > '{path}' << 'KIRO_EOF'\n{new_content}\nKIRO_EOF"
        )
    
    elif tool_name == "fsAppend":
        path = arguments.get("path", "")
        text = arguments.get("text", "")
        if not path:
            return "Error: No path provided"
        return await docker_exec(workspace, f"echo '{text}' >> '{path}'")
    
    elif tool_name == "deleteFile":
        path = arguments.get("targetFile", arguments.get("path", ""))
        if not path:
            return "Error: No path provided"
        return await docker_exec(workspace, f"rm -f '{path}'")
    
    elif tool_name == "readMultipleFiles":
        paths = arguments.get("paths", [])
        if not paths:
            return "Error: No paths provided"
        results = []
        for p in paths:
            content = await docker_exec(workspace, f"cat '{p}'")
            results.append(f"=== {p} ===\n{content}")
        return "\n\n".join(results)
    
    else:
        return f"Error: Unknown tool '{tool_name}'"


# ============================================================================
# Tool Call Parsing
# ============================================================================

def parse_tool_calls(response: str) -> list[dict] | None:
    """
    Parse tool calls from model response.
    
    Supports multiple formats:
    1. Qwen3 format: <tool_call>{"name": "...", "arguments": {...}}</tool_call>
    2. Function format: <function=name>...</function>
    3. JSON code blocks
    """
    import re
    
    tool_calls = []
    
    # Try Qwen3/OpenAI style: <tool_call>{"name": "...", "arguments": {...}}</tool_call>
    qwen_pattern = r"<tool_call>\s*(\{.*?\})\s*</tool_call>"
    matches = re.findall(qwen_pattern, response, re.DOTALL)
    for match in matches:
        try:
            call = json.loads(match)
            if "name" in call:
                tool_calls.append({
                    "name": call["name"],
                    "arguments": call.get("arguments", {}),
                })
        except json.JSONDecodeError:
            continue
    
    # Try function format: <function=name><parameter=key>value</parameter></function>
    if not tool_calls:
        func_pattern = r"<function=(\w+)>(.*?)</function>"
        matches = re.findall(func_pattern, response, re.DOTALL)
        for name, params_str in matches:
            arguments = {}
            param_pattern = r"<parameter=(\w+)>\s*(.*?)\s*</parameter>"
            param_matches = re.findall(param_pattern, params_str, re.DOTALL)
            for param_name, param_value in param_matches:
                arguments[param_name] = param_value.strip()
            if name:
                tool_calls.append({"name": name, "arguments": arguments})
    
    # Try JSON code block
    if not tool_calls:
        json_pattern = r"```(?:json)?\s*(\{[^`]*\"name\"[^`]*\})\s*```"
        matches = re.findall(json_pattern, response, re.DOTALL)
        for match in matches:
            try:
                call = json.loads(match)
                if "name" in call:
                    tool_calls.append({
                        "name": call["name"],
                        "arguments": call.get("arguments", {}),
                    })
            except json.JSONDecodeError:
                continue
    
    return tool_calls if tool_calls else None


def format_tool_result(tool_name: str, tool_id: str, result: str) -> str:
    """Format tool execution result for the model."""
    return f"\n<tool_result tool_call_id=\"{tool_id}\" name=\"{tool_name}\">\n{result}\n</tool_result>\n"


def is_task_complete(response: str) -> bool:
    """Check if the agent indicates task completion."""
    completion_markers = [
        "<task_complete>",
        "TASK COMPLETE",
        "I have completed the task",
        "The task is complete",
        "changes have been made",
        "fix has been applied",
    ]
    response_lower = response.lower()
    return any(marker.lower() in response_lower for marker in completion_markers)


# ============================================================================
# Main Generate Function
# ============================================================================

async def generate(args, sample: Sample, sampling_params: dict) -> Sample:
    """
    Multi-turn generation with tool use for SWE-bench tasks.
    
    This function:
    1. Preloads Docker image if available
    2. Sets up a Docker container with the repo mounted
    3. Runs a multi-turn loop where the model can call tools
    4. Tools execute via `docker exec` inside the container
    5. Extracts the git diff patch at the end
    6. Logs trajectory and metrics
    7. Cleans up the container
    
    Args:
        args: SLIME rollout arguments
        sample: Sample with prompt and metadata (instance_id, repo, base_commit, etc.)
        sampling_params: LLM sampling parameters
        
    Returns:
        Sample with tokens, response, loss_mask, and rollout_log_probs
    """
    assert not args.partial_rollout, "Partial rollout not supported for SWE-bench tasks"
    
    state = GenerateState(args)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    
    instance_id = sample.metadata.get("instance_id", f"instance_{sample.index}")
    rollout_idx = sample.metadata.get("rollout_idx", sample.index % args.n_samples_per_prompt)
    
    # Initialize trajectory logger
    trajectory_logger = TrajectoryLogger(
        instance_id=instance_id,
        rollout_idx=rollout_idx,
        trajectory_folder=CONFIG.trajectory_folder,
    )
    trajectory_logger.start()
    
    # Setup workspace (includes Docker image preloading and container start)
    workspace = await setup_workspace(sample, rollout_idx)
    
    if not workspace.is_active:
        logger.error(f"Workspace setup failed for {instance_id}")
        sample.status = Sample.Status.FAILED
        sample.response = "Error: Workspace setup failed"
        sample.tokens = []
        sample.response_length = 0
        sample.loss_mask = []
        sample.rollout_log_probs = []
        
        # Log failed trajectory
        trajectory_logger.finalize(
            status="FAILED",
            error_message="Workspace setup failed",
        )
        trajectory_logger.save()
        
        return sample
    
    start_time = time.time()
    error_message = ""
    
    try:
        # Get file tree for context
        file_tree = get_file_tree_from_container(workspace)
        
        # Build initial prompt
        problem_statement = sample.prompt if isinstance(sample.prompt, str) else sample.prompt[0]["content"]
        
        # Build system prompt with tools
        system_prompt = CONFIG.system_prompt
        
        # Build user message using instruction template from Kiro-on-Strands
        # Format: file tree context + instruction with problem statement
        user_message = f"""You are operating in a workspace with files and folders. Below is the known structure of the workspace.

{file_tree}

{CONFIG.instruction_template.format(pr_description=problem_statement, location=workspace.workdir)}"""
        
        # Log initial messages
        trajectory_logger.add_system_message(system_prompt)
        trajectory_logger.add_user_message(user_message)
        
        # Format as chat
        if hasattr(state.tokenizer, "apply_chat_template"):
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ]
            full_prompt = state.tokenizer.apply_chat_template(
                messages,
                tools=TOOL_DEFINITIONS,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            full_prompt = f"{system_prompt}\n\nUser: {user_message}\n\nAssistant:"
        
        # Tokenize initial prompt
        prompt_tokens = state.tokenizer.encode(full_prompt, add_special_tokens=False)
        trajectory_logger.set_prompt_tokens(len(prompt_tokens))
        
        logger.info(
            f"[GENERATE] Starting {instance_id}: prompt_tokens={len(prompt_tokens)}, "
            f"max_turns={CONFIG.max_turns}, timeout={CONFIG.timeout_seconds}s"
        )
        
        # Initialize tracking
        response = ""
        response_tokens = []
        loss_mask = []
        rollout_log_probs = []
        tool_iterations = 0
        tool_call_count = 0
        
        # Multi-turn loop
        for turn in range(CONFIG.max_turns):
            # Check timeout
            if time.time() - start_time > CONFIG.timeout_seconds:
                logger.warning(f"Timeout reached for {instance_id}")
                sample.status = Sample.Status.TRUNCATED
                error_message = f"Timeout after {CONFIG.timeout_seconds}s"
                break
            
            # Start turn timing
            trajectory_logger.start_turn()
            turn_start = time.time()
            
            # Build payload
            payload = {
                "text": full_prompt + response,
                "sampling_params": sampling_params,
                "return_logprob": True,
            }
            
            # Call SGLang server
            output = await post(url, payload)
            
            # Handle abort
            if output["meta_info"]["finish_reason"]["type"] == "abort":
                sample.status = Sample.Status.ABORTED
                error_message = "Generation aborted by server"
                break
            
            finish_reason = output["meta_info"]["finish_reason"]["type"]
            
            # Extract tokens and log probs
            if "output_token_logprobs" in output["meta_info"]:
                cur_tokens = [item[1] for item in output["meta_info"]["output_token_logprobs"]]
                cur_log_probs = [item[0] for item in output["meta_info"]["output_token_logprobs"]]
            else:
                cur_tokens = state.tokenizer.encode(output["text"], add_special_tokens=False)
                cur_log_probs = [0.0] * len(cur_tokens)
            
            cur_response = output["text"]
            
            # Add model output to tracking (loss_mask = 1 for model output)
            response += cur_response
            response_tokens += cur_tokens
            loss_mask += [1] * len(cur_tokens)
            rollout_log_probs += cur_log_probs
            
            # Parse tool calls for logging
            tool_calls = parse_tool_calls(cur_response)
            
            # Log assistant message
            trajectory_logger.add_assistant_message(
                content=cur_response,
                tokens=cur_tokens,
                log_probs=cur_log_probs,
                finish_reason=finish_reason,
                tool_calls=tool_calls,
            )
            
            logger.debug(
                f"[GENERATE] Turn {turn}: tokens={len(cur_tokens)}, "
                f"tool_calls={len(tool_calls) if tool_calls else 0}, "
                f"finish_reason={finish_reason}"
            )
            
            # Check for completion
            if finish_reason == "length":
                sample.status = Sample.Status.TRUNCATED
                error_message = "Max tokens reached"
                break
            
            if is_task_complete(cur_response):
                sample.status = Sample.Status.COMPLETED
                break
            
            # Handle tool calls
            if not tool_calls:
                # No tool calls - check if model is done
                if turn > 0:
                    sample.status = Sample.Status.COMPLETED
                break
            
            if tool_iterations >= CONFIG.max_iterations:
                logger.warning(f"Max iterations ({CONFIG.max_iterations}) reached for {instance_id}")
                sample.status = Sample.Status.TRUNCATED
                error_message = f"Max iterations ({CONFIG.max_iterations}) reached"
                break
            
            # Execute tools and add results
            tool_results = []
            for i, tool_call in enumerate(tool_calls):
                tool_name = tool_call["name"]
                tool_args = tool_call["arguments"]
                tool_id = f"tool_{turn}_{i}"
                
                logger.debug(f"[GENERATE] Executing tool: {tool_name}({json.dumps(tool_args)[:100]}...)")
                
                tool_start = time.time()
                result = await execute_tool(workspace, tool_name, tool_args)
                tool_duration = time.time() - tool_start
                
                tool_results.append(format_tool_result(tool_name, tool_id, result))
                tool_call_count += 1
                
                # Log tool result
                trajectory_logger.add_tool_result(
                    tool_name=tool_name,
                    tool_id=tool_id,
                    result=result,
                    duration=tool_duration,
                )
            
            tool_iterations += 1
            
            # Add tool results to response (loss_mask = 0 for tool output)
            tool_output = "".join(tool_results)
            tool_tokens = state.tokenizer.encode(tool_output, add_special_tokens=False)
            
            response += tool_output
            response_tokens += tool_tokens
            loss_mask += [0] * len(tool_tokens)  # Don't train on tool output
            rollout_log_probs += [0.0] * len(tool_tokens)  # Dummy values
        
        # Set final status if not already set
        if sample.status == Sample.Status.PENDING:
            sample.status = Sample.Status.COMPLETED
        
        # Extract patch from container
        patch = get_patch_from_container(workspace)
        
        # Populate sample
        sample.tokens = prompt_tokens + response_tokens
        sample.response = response
        sample.response_length = len(response_tokens)
        sample.loss_mask = loss_mask
        sample.rollout_log_probs = rollout_log_probs
        
        # Store metadata
        sample.metadata["patch"] = patch
        sample.metadata["tool_iterations"] = tool_iterations
        sample.metadata["tool_call_count"] = tool_call_count
        sample.metadata["num_turns"] = turn + 1
        sample.metadata["agent_duration"] = time.time() - start_time
        
        # Finalize and save trajectory
        trajectory_logger.finalize(
            status=sample.status.name,
            patch=patch,
            error_message=error_message,
        )
        trajectory_file = trajectory_logger.save()
        
        if trajectory_file:
            sample.metadata["trajectory_file"] = trajectory_file
        
        # Log summary
        summary = trajectory_logger.get_summary()
        logger.info(
            f"[GENERATE] Completed {instance_id}: status={summary['status']}, "
            f"turns={summary['total_turns']}, tokens={summary['total_tokens']}, "
            f"tools={summary['total_tool_calls']}, patch_lines={summary['patch_lines']}, "
            f"duration={summary['duration_seconds']}s"
        )
    
    except Exception as e:
        error_message = f"{type(e).__name__}: {str(e)}"
        logger.error(f"[GENERATE] Error for {instance_id}: {error_message}")
        
        sample.status = Sample.Status.FAILED
        sample.response = f"Error: {error_message}"
        sample.tokens = []
        sample.response_length = 0
        sample.loss_mask = []
        sample.rollout_log_probs = []
        
        # Log failed trajectory
        trajectory_logger.finalize(
            status="FAILED",
            error_message=error_message,
        )
        trajectory_logger.save()
    
    finally:
        # Always cleanup workspace
        await cleanup_workspace(workspace)
    
    return sample


# ============================================================================
# Reward Function
# ============================================================================

async def reward_func(args, sample: Sample, **kwargs) -> float:
    # TODO: Fake reward for now
    return 1