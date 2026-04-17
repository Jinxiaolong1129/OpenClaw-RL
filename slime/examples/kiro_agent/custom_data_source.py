"""
Custom Data Sources for Agent-based Training

This module provides two custom data source implementations for loading
parquet datasets with custom fields (instance_id, env_id, workspace, etc.)
for multi-turn agent training.

Usage:
    # Without buffer (simple, no partial rollout support):
    --data-source-path examples.kiro_on_strands.custom_data_source:AgentDataSource

    # With buffer (supports partial rollout):
    --data-source-path examples.kiro_on_strands.custom_data_source:AgentDataSourceWithBuffer
"""

import copy
import logging
import os
from pathlib import Path

import pandas as pd
import torch

from slime.rollout.data_source import RolloutDataSource, RolloutDataSourceWithBuffer
from slime.utils.misc import load_function
from slime.utils.types import Sample

logger = logging.getLogger(__name__)


class AgentDataSource(RolloutDataSource):
    """
    Custom data source for agent-based training WITHOUT buffer support.

    Loads parquet/jsonl/csv files with custom fields and passes them
    through Sample.metadata for use in custom generate functions.

    Expected data format (parquet/jsonl/csv):
        - problem_statement: str - The prompt/task description
        - instance_id: str - Unique identifier for the task
        - metadata.workdir: str (optional) - Workspace path or configuration
        - patch: str (optional) - Ground truth
        - ... any other custom fields

    Example parquet schema:
        | problem_statement | instance_id | patch | metadata.workdir | .... |
        |-------------------|-------------|--------|-----------|-----------------|
        | "Fix the bug..." | "bug-001"   | "gt"     | "/app"   | "..." |
    """

    def __init__(self, args):
        # Don't call super().__init__ to avoid loading dataset twice
        # We'll handle our own data loading
        self.args = args
        self.epoch_id = 0
        self.sample_offset = 0
        self.sample_group_index = 0
        self.sample_index = 0
        self.metadata = {}

        # Load custom dataset
        self.dataset = self._load_dataset()
        logger.info(f"Loaded {len(self.dataset)} samples from {args.prompt_data}")

        # Log dataset columns for debugging metadata propagation
        if self.dataset:
            columns = list(self.dataset[0].keys())
            logger.info(f"Dataset columns: {columns}")
            if "instance_id" not in columns:
                logger.warning(
                    f"'instance_id' column not found in dataset. "
                    f"Available columns: {columns}. "
                    f"Samples will use fallback instance_id='instance_{{index}}'."
                )

    def _load_dataset(self) -> list[dict]:
        """Load dataset from args.prompt_data with support for multiple formats."""
        data_path = self.args.prompt_data

        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Dataset path '{data_path}' does not exist.")

        if data_path.endswith(".parquet"):
            df = pd.read_parquet(data_path)
        elif data_path.endswith(".csv"):
            df = pd.read_csv(data_path)
        elif data_path.endswith(".jsonl"):
            df = pd.read_json(data_path, lines=True)
        elif data_path.endswith(".json"):
            df = pd.read_json(data_path)
        else:
            raise ValueError(
                f"Unsupported file format: {data_path}. "
                "Supported formats: .parquet, .csv, .jsonl, .json"
            )

        return df.to_dict("records")

    def _build_sample_from_item(self, item: dict) -> Sample:
        """Convert a data item to a Sample object."""
        # Get prompt from configurable key or fallback
        prompt_key = getattr(self.args, "input_key", "problem_statement")
        prompt = item.get(prompt_key, item.get("problem_statement", item.get("prompt", "")))

        # Get label/ground truth from configurable key or fallback
        label_key = getattr(self.args, "label_key", "patch")
        label = item.get(label_key) if label_key else item.get("patch", item.get("label"))

        # Build metadata with all custom fields
        # Exclude prompt and label fields from metadata to avoid duplication
        exclude_keys = {prompt_key, "problem_statement", "prompt", "label", "patch"}
        if label_key:
            exclude_keys.add(label_key)

        metadata = {k: v for k, v in item.items() if k not in exclude_keys}

        if "instance_id" not in metadata:
            logger.debug(
                f"No 'instance_id' in metadata for item. "
                f"Item keys: {list(item.keys())}, exclude_keys: {exclude_keys}, "
                f"metadata keys: {list(metadata.keys())}"
            )

        metadata = {k: v for k, v in item.items() if k not in exclude_keys}

        return Sample(
            prompt=prompt,
            label=label,
            metadata=metadata,
        )

    def get_samples(self, num_samples: int) -> list[list[Sample]]:
        """
        Return num_samples groups of samples.

        Each group contains n_samples_per_prompt copies of the same prompt
        (for techniques like GRPO that need multiple responses per prompt).
        """
        samples = []

        for _ in range(num_samples):
            # Handle epoch wraparound
            if self.sample_offset >= len(self.dataset):
                self.epoch_id += 1
                self.sample_offset = 0
                if self.args.rollout_shuffle:
                    self._shuffle_dataset()
                logger.info(f"Starting epoch {self.epoch_id}")

            item = self.dataset[self.sample_offset]
            self.sample_offset += 1

            # Create a group of samples (n_samples_per_prompt copies)
            group = []
            for _ in range(self.args.n_samples_per_prompt):
                sample = self._build_sample_from_item(item)
                sample.group_index = self.sample_group_index
                sample.index = self.sample_index
                self.sample_index += 1
                group.append(copy.deepcopy(sample))

            self.sample_group_index += 1
            samples.append(group)

        return samples

    def _shuffle_dataset(self):
        """Shuffle dataset with deterministic seed."""
        import random
        seed = getattr(self.args, "rollout_seed", 42) + self.epoch_id
        random.seed(seed)
        random.shuffle(self.dataset)

    def add_samples(self, samples: list[list[Sample]]):
        """Discard aborted samples. Use AgentDataSourceWithBuffer for buffer support."""
        if samples:
            logger.info(
                f"Discarding {len(samples)} aborted sample groups "
                f"(use AgentDataSourceWithBuffer for partial rollout support)."
            )

    def save(self, rollout_id):
        """Save state for checkpointing."""
        if self.args.save is None:
            return

        state_dict = {
            "sample_offset": self.sample_offset,
            "epoch_id": self.epoch_id,
            "sample_group_index": self.sample_group_index,
            "sample_index": self.sample_index,
            "metadata": self.metadata,
        }
        path = os.path.join(self.args.save, f"rollout/agent_data_source_state_{rollout_id}.pt")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(state_dict, path)
        logger.info(f"Saved data source state to {path}")

    def load(self, rollout_id=None):
        """Load state from checkpoint."""
        if self.args.load is None:
            return

        path = os.path.join(self.args.load, f"rollout/agent_data_source_state_{rollout_id}.pt")
        if not os.path.exists(path):
            logger.info(f"Checkpoint {path} does not exist, starting fresh.")
            return

        state_dict = torch.load(path, weights_only=True)
        self.sample_offset = state_dict.get("sample_offset", 0)
        self.epoch_id = state_dict.get("epoch_id", 0)
        self.sample_group_index = state_dict.get("sample_group_index", 0)
        self.sample_index = state_dict.get("sample_index", 0)
        self.metadata = state_dict.get("metadata", {})

        # Re-shuffle to match the epoch state
        if self.args.rollout_shuffle and self.epoch_id > 0:
            self._shuffle_dataset()

        logger.info(f"Loaded data source state from {path}, epoch={self.epoch_id}, offset={self.sample_offset}")


class AgentDataSourceWithBuffer(AgentDataSource):
    """
    Custom data source for agent-based training WITH buffer support.

    Extends AgentDataSource with a buffer for storing partial rollout samples.
    When using --partial-rollout, aborted samples are saved to the buffer
    and reused in subsequent rollouts.

    Usage:
        --data-source-path examples.kiro_on_strands.custom_data_source:AgentDataSourceWithBuffer
        --partial-rollout  # Enable partial rollout to use buffer
    """

    def __init__(self, args):
        super().__init__(args)
        self.buffer = []

        # Load custom buffer filter if specified
        if getattr(args, "buffer_filter_path", None) is not None:
            self.buffer_filter = load_function(args.buffer_filter_path)
        else:
            self.buffer_filter = self._default_buffer_filter

    def _default_buffer_filter(
        self, args, rollout_id, buffer: list[list[Sample]], num_samples: int
    ) -> list[list[Sample]]:
        """Default buffer filter: FIFO (first in, first out)."""
        num_to_pop = min(len(buffer), num_samples)
        samples = buffer[:num_to_pop]
        del buffer[:num_to_pop]
        return samples

    def get_samples(self, num_samples: int) -> list[list[Sample]]:
        """
        Return num_samples groups of samples.

        First draws from the buffer (partial rollout samples),
        then from the dataset if more samples are needed.
        """
        # First, try to get samples from buffer
        samples = self._get_samples_from_buffer(num_samples)
        remaining = num_samples - len(samples)

        if remaining > 0:
            # Get remaining samples from dataset
            samples += super().get_samples(remaining)

        return samples

    def _get_samples_from_buffer(self, num_samples: int) -> list[list[Sample]]:
        """Get samples from the buffer."""
        if len(self.buffer) == 0 or num_samples == 0:
            return []

        samples = self.buffer_filter(self.args, None, self.buffer, num_samples)
        logger.debug(f"Retrieved {len(samples)} samples from buffer, {len(self.buffer)} remaining")
        return samples

    def add_samples(self, samples: list[list[Sample]]):
        """
        Add sample groups to the buffer.

        Called when partial rollout samples are aborted and need to be
        saved for continuation in the next rollout.
        """
        if not samples:
            return

        assert isinstance(samples, list), f"samples must be a list, got {type(samples)}"
        assert isinstance(samples[0], list), f"elements must be lists, got {type(samples[0])}"

        for group in samples:
            assert len(group) == self.args.n_samples_per_prompt, (
                f"Group size {len(group)} != n_samples_per_prompt {self.args.n_samples_per_prompt}"
            )
            self.buffer.append(group)

        logger.info(f"Added {len(samples)} sample groups to buffer, total buffer size: {len(self.buffer)}")

    def get_buffer_length(self) -> int:
        """Return the current buffer size."""
        return len(self.buffer)

    def save(self, rollout_id):
        """Save state including buffer for checkpointing."""
        super().save(rollout_id)

        if self.args.save is None or len(self.buffer) == 0:
            return

        # Save buffer separately (can be large)
        buffer_path = os.path.join(self.args.save, f"rollout/agent_data_source_buffer_{rollout_id}.pt")
        buffer_data = [[sample.to_dict() for sample in group] for group in self.buffer]
        torch.save(buffer_data, buffer_path)
        logger.info(f"Saved buffer with {len(self.buffer)} groups to {buffer_path}")

    def load(self, rollout_id=None):
        """Load state including buffer from checkpoint."""
        super().load(rollout_id)

        if self.args.load is None:
            return

        buffer_path = os.path.join(self.args.load, f"rollout/agent_data_source_buffer_{rollout_id}.pt")
        if not os.path.exists(buffer_path):
            return

        buffer_data = torch.load(buffer_path, weights_only=False)
        self.buffer = [[Sample.from_dict(s) for s in group] for group in buffer_data]
        logger.info(f"Loaded buffer with {len(self.buffer)} groups from {buffer_path}")
