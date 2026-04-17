"""mini-swe-agent scaffold + main agent loop.

Ported from ``aws_kiro_infra/generate_with_swe.py`` (trainer-side), now runs
inside the CPU agent pod. Produces per-message ``token_ids`` / ``token_mask``
/ ``token_logprobs`` so the trainer can reconstruct a slime ``Sample`` with
byte-perfect fidelity.

Dependencies on CPU pod:
  * transformers (tokenizer)
  * jinja2 (observation template)
  * PyYAML (swebench.yaml config)
  * swebench / swegym (eval harness) — delegated to ``patch_utils``
"""

from __future__ import annotations

import asyncio
import copy
import logging
import os
import threading
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from . import patch_utils
from .docker_ops import DockerManager
from .sglang_client import SGLangClient

logger = logging.getLogger("swe_agent_server.mini_swe_agent")

_THIS_DIR = Path(__file__).resolve().parent
# Default config: aws_kiro_infra/swebench.yaml (one level up from server/)
_DEFAULT_CONFIG_PATH = str(_THIS_DIR.parent / "swebench.yaml")

_DEFAULT_REGISTRY = "docker.io"


def get_docker_image_name(instance: dict, data_source: str) -> str:
    """Resolve the docker image for an instance.

    Priority:
      1. ``instance["docker_image"]`` — explicit (R2E-Gym native; also any
         custom dataset that pre-computes image names)
      2. ``instance["image_name"]`` — explicit (legacy alias)
      3. Derive from ``data_source`` + ``instance_id``:
           - ``swe-gym*``   → ``docker.io/xingyaoww/sweb.eval.x86_64.<iid_s_>:latest``
           - ``swe-bench*`` → ``docker.io/swebench/sweb.eval.x86_64.<iid_1776_>:latest``
           - ``r2e-gym*`` without explicit ``docker_image`` is an error
             (R2E-Gym images aren't derivable from instance_id — they're in
             the HF dataset's ``docker_image`` column).
    """
    # Priority 1: R2E-Gym style
    explicit = instance.get("docker_image") or instance.get("image_name")
    if explicit:
        return explicit

    registry = os.getenv("SWE_DOCKER_REGISTRY", _DEFAULT_REGISTRY)
    iid = instance["instance_id"]
    ds = data_source.lower()
    if "swe-gym" in ds:
        id_docker_compatible = iid.replace("__", "_s_")
        return f"{registry}/xingyaoww/sweb.eval.x86_64.{id_docker_compatible}:latest".lower()
    if "swe-bench" in ds:
        id_docker_compatible = iid.replace("__", "_1776_")
        return f"{registry}/swebench/sweb.eval.x86_64.{id_docker_compatible}:latest".lower()
    if "r2e" in ds:
        raise ValueError(
            f"R2E-Gym instance {iid!r} is missing `instance.docker_image`; "
            "ensure your preprocess script copies the HF `docker_image` column "
            "into instance. See aws_kiro_infra/data/preprocess_r2egym.py."
        )
    raise NotImplementedError(f"Data source: {data_source} is not supported")


# =============================================================================
# Config loading (mini-swe-agent swebench.yaml)
# =============================================================================

@lru_cache(maxsize=1)
def get_sweagent_config(config_path: str | None = None) -> dict:
    path = Path(config_path or os.getenv("SWE_CONFIG_PATH", _DEFAULT_CONFIG_PATH))
    if not path.exists():
        parent_yaml = _THIS_DIR.parent.parent / "swebench.yaml"
        if parent_yaml.exists():
            path = parent_yaml
        else:
            raise FileNotFoundError(f"SWE config not found: {config_path or _DEFAULT_CONFIG_PATH}")
    return yaml.safe_load(path.read_text())


# =============================================================================
# Observation rendering
# =============================================================================

def render_observation(config: dict, returncode: int, output: str) -> str:
    from jinja2 import Template
    template_str = config.get("agent", {}).get("action_observation_template", "")
    if not template_str:
        return f"<returncode>{returncode}</returncode>\n<output>\n{output}\n</output>"
    template = Template(template_str)
    return template.render(output={"returncode": returncode, "output": output})


# =============================================================================
# Chat-template wrapper tokens (byte-perfect per-message tracking)
# =============================================================================

_wrapper_token_cache: dict[int, dict[str, list[int]]] = {}
_wrapper_token_cache_lock = threading.Lock()


def _compute_wrapper_tokens(tokenizer) -> dict[str, list[int]]:
    probe_user = [{"role": "user", "content": "_"}]
    with_gen = tokenizer.apply_chat_template(probe_user, tokenize=True, add_generation_prompt=True)
    without_gen = tokenizer.apply_chat_template(probe_user, tokenize=True, add_generation_prompt=False)
    if not isinstance(with_gen, list):
        with_gen = list(with_gen)
    if not isinstance(without_gen, list):
        without_gen = list(without_gen)
    assert with_gen[: len(without_gen)] == without_gen, (
        "Chat template generation prompt is not a pure suffix; trajectory mode relies on it."
    )
    gen_prompt = with_gen[len(without_gen):]
    trailing_nl = tokenizer.encode("\n", add_special_tokens=False)
    im_end_id = None
    try:
        im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    except Exception:
        im_end_id = None
    if im_end_id is None or im_end_id < 0:
        im_end_id = tokenizer.eos_token_id
    return {
        "gen_prompt": list(gen_prompt),
        "trailing_nl": list(trailing_nl),
        "im_end": [im_end_id] if im_end_id is not None else [],
    }


def get_wrapper_tokens(tokenizer) -> dict[str, list[int]]:
    key = id(tokenizer)
    with _wrapper_token_cache_lock:
        cached = _wrapper_token_cache.get(key)
        if cached is not None:
            return cached
    computed = _compute_wrapper_tokens(tokenizer)
    with _wrapper_token_cache_lock:
        _wrapper_token_cache[key] = computed
    return computed


def encode_single_message(tokenizer, msg: dict) -> list[int]:
    rendered = tokenizer.apply_chat_template(
        [msg], tokenize=True, add_generation_prompt=False,
    )
    if not isinstance(rendered, list):
        rendered = list(rendered)
    return rendered


def build_env_side_msg(tokenizer, role: str, content: str) -> dict:
    """Build a non-assistant message (system/user) with token ids + zero mask."""
    msg = {"role": role, "content": content}
    toks = encode_single_message(tokenizer, msg)
    msg["token_ids"] = toks
    msg["token_mask"] = [0] * len(toks)
    msg["token_logprobs"] = [0.0] * len(toks)
    return msg


# =============================================================================
# Main agent loop
# =============================================================================

async def run_agent_loop(
    *,
    docker: DockerManager,
    container_id: str,
    sglang: SGLangClient,
    tokenizer,
    instance: dict,
    sweagent_config: dict,
    sampling_params: dict,
    step_limit: int = 30,
    max_context_len: int = 0,
    max_new_tokens: int = 4096,
    cwd: str = "/testbed",
    exec_timeout: int = 180,
) -> dict[str, Any]:
    """Run the mini-swe-agent multi-turn loop.

    Returns a dict of:
      * ``messages``      — per-message {role, content, token_ids, token_mask, token_logprobs}
      * ``step_debug``    — per-step action / exec output summary
      * ``git_patch``     — final unified diff (None if never produced)
      * ``patch_source``  — "submission" | "git_diff_fallback" | None
      * ``exit_status``   — "submitted" | "max_steps" | "context_overflow" | "error"
      * ``n_steps``
      * ``error``
    """
    iid = instance.get("instance_id", "unknown")
    agent_config = sweagent_config.get("agent", {})

    system_template = agent_config.get("system_template", "You are a helpful assistant.")
    instance_template = agent_config.get("instance_template", "{{task}}")
    from jinja2 import Template
    instance_message = Template(instance_template).render(task=instance["problem_statement"])

    wrapper = get_wrapper_tokens(tokenizer)
    gen_prompt_tokens = wrapper["gen_prompt"]
    trailing_nl_tokens = wrapper["trailing_nl"]
    im_end_tokens = wrapper["im_end"]

    messages: list[dict] = [
        build_env_side_msg(tokenizer, "system", system_template),
        build_env_side_msg(tokenizer, "user", instance_message),
    ]

    step_debug: list[dict] = []
    git_patch: str | None = None
    patch_source: str | None = None
    exit_status: str | None = None
    error: str | None = None
    n_steps = 0

    sglang_sampling_params = {
        "temperature": sampling_params.get("temperature", 1.0),
        "max_new_tokens": max_new_tokens,
    }
    if "top_p" in sampling_params:
        sglang_sampling_params["top_p"] = sampling_params["top_p"]
    stop_token_ids: set[int] = set()
    if tokenizer.eos_token_id is not None:
        stop_token_ids.add(tokenizer.eos_token_id)
    if im_end_tokens:
        stop_token_ids.add(im_end_tokens[0])
    if stop_token_ids:
        sglang_sampling_params["stop_token_ids"] = list(stop_token_ids)

    t0 = time.time()
    for step_idx in range(step_limit):
        n_steps = step_idx + 1

        full_input_ids: list[int] = []
        for m in messages:
            full_input_ids.extend(m["token_ids"])
        full_input_ids.extend(gen_prompt_tokens)

        if max_context_len > 0 and len(full_input_ids) + max_new_tokens > max_context_len:
            exit_status = "context_overflow"
            error = (
                f"context overflow at step {step_idx}: "
                f"input={len(full_input_ids)}, max_new={max_new_tokens}, budget={max_context_len}"
            )
            logger.warning("[%s] %s", iid, error)
            break

        try:
            gen = await sglang.generate(
                input_ids=full_input_ids,
                sampling_params=sglang_sampling_params,
                return_logprob=True,
            )
        except Exception as e:
            error = f"SGLang call failed at step {step_idx}: {e}"
            logger.error("[%s] %s", iid, error)
            exit_status = "error"
            break

        assistant_text = gen["text"]
        output_token_ids = gen["output_token_ids"]
        output_logprobs = gen["output_logprobs"]
        if not output_token_ids:
            # Router didn't return token ids; fall back to re-tokenization
            output_token_ids = tokenizer.encode(assistant_text, add_special_tokens=False)
            output_logprobs = [0.0] * len(output_token_ids)

        if im_end_tokens and (not output_token_ids or output_token_ids[-1] != im_end_tokens[0]):
            output_token_ids = list(output_token_ids) + im_end_tokens
            output_logprobs = list(output_logprobs) + [0.0]

        asst_tok_ids = list(gen_prompt_tokens) + list(output_token_ids) + list(trailing_nl_tokens)
        asst_tok_mask = (
            [0] * len(gen_prompt_tokens)
            + [1] * len(output_token_ids)
            + [0] * len(trailing_nl_tokens)
        )
        asst_tok_logprobs = (
            [0.0] * len(gen_prompt_tokens)
            + list(output_logprobs)
            + [0.0] * len(trailing_nl_tokens)
        )
        messages.append({
            "role": "assistant",
            "content": assistant_text,
            "token_ids": asst_tok_ids,
            "token_mask": asst_tok_mask,
            "token_logprobs": asst_tok_logprobs,
        })

        bash_cmd = patch_utils.parse_bash_action(assistant_text)
        if bash_cmd is None:
            obs = render_observation(
                sweagent_config, -1, "No valid bash command found in response."
            )
            messages.append(build_env_side_msg(tokenizer, "user", obs))
            continue

        is_submit = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in bash_cmd

        step_t0 = time.time()
        step_timeout = exec_timeout + 30
        try:
            exec_result = await asyncio.wait_for(
                docker.exec(
                    container_id=container_id,
                    command=bash_cmd,
                    cwd=cwd,
                    timeout=exec_timeout,
                ),
                timeout=step_timeout,
            )
            returncode = exec_result.get("returncode", -1)
            output_text = exec_result.get("output", "")
        except (asyncio.TimeoutError, TimeoutError):
            returncode = -1
            output_text = f"Command timed out after {exec_timeout}s."
            logger.warning("[%s] step %d timed out", iid, step_idx)
        except Exception as e:
            returncode = -1
            output_text = f"Execution error: {e}"
            logger.error("[%s] step %d exec error: %s", iid, step_idx, e)

        step_debug.append({
            "step_idx": step_idx,
            "action": bash_cmd,
            "returncode": returncode,
            "output_len": len(output_text),
            "output_head": output_text[:2000],
            "output_tail": output_text[-2000:] if len(output_text) > 2000 else output_text,
            "start_ts": step_t0,
            "end_ts": time.time(),
            "ok": returncode != -1,
        })

        if is_submit:
            exit_status = "submitted"
            candidate_patch = patch_utils.extract_patch_from_submission(output_text)
            if patch_utils.is_valid_git_patch(candidate_patch):
                git_patch = candidate_patch
                patch_source = "submission"
            break

        observation = render_observation(sweagent_config, returncode, output_text)
        remaining = step_limit - (step_idx + 1)
        if remaining == 1:
            observation += "\nREMINDER: You only have 1 turn left. Please provide the final answer"
        elif remaining > 1:
            observation += f"\nREMINDER: You have {remaining} turns left to arrive at the solution."
        messages.append(build_env_side_msg(tokenizer, "user", observation))

    # Git diff fallback if agent ended without explicit submission
    if git_patch is None and exit_status not in ("error", "context_overflow"):
        try:
            fallback_patch = await docker.diff(container_id=container_id, cwd=cwd)
            if patch_utils.is_valid_git_patch(fallback_patch):
                git_patch = fallback_patch
                patch_source = "git_diff_fallback"
            if exit_status is None:
                exit_status = "max_steps"
        except Exception as e:
            if error is None:
                error = f"diff failed: {e}"

    logger.info(
        "[%s] Agent done: steps=%d, exit=%s, patch=%s, elapsed=%.1fs",
        iid, n_steps, exit_status, "yes" if git_patch else "no", time.time() - t0,
    )

    return {
        "messages": messages,
        "step_debug": step_debug,
        "git_patch": git_patch,
        "patch_source": patch_source,
        "exit_status": exit_status,
        "n_steps": n_steps,
        "error": error,
    }
