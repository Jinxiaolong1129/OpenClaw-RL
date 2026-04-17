"""Trajectory-level LLM-as-judge auxiliary reward.

Not a PRM (no per-step scoring). Called ONCE per trajectory after rollout
completes, with the full context (problem, agent steps, final patch, test
outcome). Returns a single scalar in [-1, +1] that's combined with the
outcome reward as an **auxiliary** signal:

    final_score = outcome_reward + aux_coef * judge_score

Rationale:
  * Much cheaper than PRM (N turns × LLM calls → 1 LLM call per trajectory).
  * Simpler training-time plumbing (one scalar reward, no step_wise metadata).
  * Useful when outcome reward is sparse (many trajectories fail all tests,
    the judge can still distinguish "honest attempt with right direction"
    from "random flailing") — smoothens the binary outcome signal without
    replacing it.

Configuration (env vars):
  * ``AUX_JUDGE_MODEL``         — litellm model (e.g. "openai/gpt-4o-mini")
  * ``AUX_JUDGE_NUM_VOTES``     — majority-vote votes, default 1
  * ``AUX_JUDGE_TEMPERATURE``   — default 0.0
  * ``AUX_JUDGE_MAX_TOKENS``    — default 512
  * ``AUX_JUDGE_MAX_CONCURRENCY`` — judge calls in flight at once, default 16
  * ``AUX_JUDGE_TIMEOUT``       — per-call timeout seconds, default 120
  * API keys via the usual litellm env vars (``OPENAI_API_KEY`` etc.)
"""

from __future__ import annotations

import asyncio
import os
import re
from typing import Any, Dict, List

from loguru import logger


# =============================================================================
# Prompt templates — trajectory-level (not per-step)
# =============================================================================

_TRAJ_JUDGE_SYSTEM_PROMPT = """You are a strict evaluator for a software engineering agent that fixes GitHub issues.
You are given:
1) The issue description (problem statement).
2) The full sequence of the agent's steps (thoughts + bash commands + outputs).
3) The agent's final git patch (if any).
4) Whether the official test suite passed.

Your job is to judge the OVERALL quality of the trajectory as an auxiliary
reward signal, independent of the pass/fail outcome.""".strip()


_TRAJ_JUDGE_USER_TEMPLATE = """\
## Issue Description
{problem_statement}

## Agent Trajectory ({n_steps} steps)
{steps_block}

## Final Patch
{patch_block}

## Official Test Result
{outcome_text}
"""


_TRAJ_JUDGE_SCORING_TEMPLATE = """\
Evaluate the OVERALL quality of this trajectory on a [-1, +1] scale.

Assign a HIGH positive score (+0.5 to +1.0) if:
- The agent correctly understood the issue and explored the codebase systematically;
- The fix is targeted at the real root cause (not a surface patch);
- The agent verified its fix with tests or reproduction scripts;
- Few wasted / circular / redundant steps.

Assign a LOW / NEGATIVE score (-1.0 to -0.3) if:
- The agent misunderstood the issue or fixed the wrong thing;
- The trajectory is full of wasted steps (repeated failed commands, unnecessary exploration);
- The patch is random, doesn't address the issue, or introduces obvious bugs;
- No verification attempt.

Assign NEUTRAL (-0.3 to +0.3) if the trajectory is mixed — e.g., correct direction
but noisy execution, or careful exploration but incorrect fix.

IMPORTANT: Judge the **process and engineering quality**, not just whether the
final tests passed. A trajectory that failed tests but demonstrated sound
engineering still deserves a positive score. A trajectory that happened to
pass tests but flailed around should not get a top score just for that.

Think step by step, then put your final numeric score (a single decimal number
between -1 and +1, inclusive) in \\boxed{}.
"""


# =============================================================================
# Score extraction
# =============================================================================

_BOXED_PATTERN = re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}", re.DOTALL)
_NUMBER_PATTERN = re.compile(r"[-+]?\d+(?:\.\d+)?")


def _extract_score_from_text(text: str) -> float:
    """Extract a [-1, +1] score from the judge's ``\\boxed{...}`` output.
    Returns 0.0 if unparseable (treated as neutral).
    """
    if not text:
        return 0.0
    match = _BOXED_PATTERN.search(text)
    if not match:
        return 0.0
    boxed = match.group(1).strip()
    number_match = _NUMBER_PATTERN.search(boxed)
    if not number_match:
        return 0.0
    try:
        value = float(number_match.group(0))
    except ValueError:
        return 0.0
    # Clip to [-1, +1] in case the judge overshoots.
    return max(-1.0, min(1.0, value))


# =============================================================================
# Formatting helpers
# =============================================================================

def _summarize_step(step: dict, max_output_chars: int = 1200) -> str:
    """Render one entry from step_debug into a readable block."""
    action = step.get("action", "?")
    rc = step.get("returncode", "?")
    output_len = int(step.get("output_len", 0))
    head = step.get("output_head", "") or ""
    tail = step.get("output_tail", "") or ""

    # Prefer head for short outputs, sandwich for long.
    if output_len <= 2000:
        out = head
    elif output_len <= 4000:
        overlap = 4000 - output_len
        out = head + tail[overlap:]
    else:
        gap = output_len - 4000
        out = f"{head}\n... ({gap} chars omitted) ...\n{tail}"

    if len(out) > max_output_chars:
        out = out[:max_output_chars] + f"\n... (truncated to {max_output_chars} chars)"

    return f"$ {action}\n  returncode={rc}\n  output:\n{out}"


def _summarize_trajectory(
    step_debug: List[dict],
    max_steps_shown: int = 30,
    max_output_per_step: int = 1200,
) -> str:
    """Compact summary of an entire trajectory for judge consumption."""
    n = len(step_debug)
    if n == 0:
        return "(no steps)"

    if n <= max_steps_shown:
        blocks = []
        for i, s in enumerate(step_debug):
            blocks.append(
                f"Step {i + 1}:\n{_summarize_step(s, max_output_chars=max_output_per_step)}"
            )
        return "\n\n".join(blocks)

    # Trajectory too long — keep first half and last half, elide middle.
    keep = max_steps_shown // 2
    head_steps = step_debug[:keep]
    tail_steps = step_debug[-keep:]
    blocks = []
    for i, s in enumerate(head_steps):
        blocks.append(
            f"Step {i + 1}:\n{_summarize_step(s, max_output_chars=max_output_per_step)}"
        )
    blocks.append(f"... (omitting {n - 2 * keep} middle steps) ...")
    for j, s in enumerate(tail_steps):
        blocks.append(
            f"Step {n - keep + j + 1}:\n{_summarize_step(s, max_output_chars=max_output_per_step)}"
        )
    return "\n\n".join(blocks)


def _summarize_patch(patch: str | None, max_chars: int = 4000) -> str:
    if not patch:
        return "(none)"
    if len(patch) <= max_chars:
        return patch
    return patch[:max_chars] + f"\n... (truncated to {max_chars} of {len(patch)} chars)"


# =============================================================================
# Judge agent
# =============================================================================

class TrajectoryJudgeAgent:
    """LLM-as-judge that scores an entire trajectory with one call.

    Usage:
        judge = TrajectoryJudgeAgent()
        result = await judge.judge_trajectory(
            problem_statement=...,
            step_debug=[...],
            final_patch=...,
            resolved=False,
        )
        # result = {"score": float, "votes": [...], "raw": "..."}
    """

    def __init__(
        self,
        judge_model: str | None = None,
        num_votes: int | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: float | None = None,
        max_concurrency: int | None = None,
        max_problem_len: int = 8000,
        max_steps_shown: int = 30,
        max_output_per_step: int = 1200,
        max_patch_chars: int = 4000,
    ):
        self.judge_model = (
            judge_model or os.getenv("AUX_JUDGE_MODEL", "openai/gpt-4o-mini")
        ).strip()
        self.num_votes = max(1, int(num_votes or os.getenv("AUX_JUDGE_NUM_VOTES", "1")))
        self.temperature = float(
            temperature if temperature is not None else os.getenv("AUX_JUDGE_TEMPERATURE", "0.0")
        )
        self.max_tokens = int(max_tokens or os.getenv("AUX_JUDGE_MAX_TOKENS", "512"))
        self.timeout = float(timeout or os.getenv("AUX_JUDGE_TIMEOUT", "120"))
        cap = int(max_concurrency or os.getenv("AUX_JUDGE_MAX_CONCURRENCY", "16"))
        self._sem_cap = cap
        self._sem: asyncio.Semaphore | None = None

        self.max_problem_len = int(max_problem_len)
        self.max_steps_shown = int(max_steps_shown)
        self.max_output_per_step = int(max_output_per_step)
        self.max_patch_chars = int(max_patch_chars)

    def _get_semaphore(self) -> asyncio.Semaphore:
        if self._sem is None:
            self._sem = asyncio.Semaphore(self._sem_cap)
        return self._sem

    def _build_messages(
        self,
        *,
        problem_statement: str,
        step_debug: List[dict],
        final_patch: str | None,
        resolved: bool | None,
    ) -> List[Dict[str, str]]:
        problem = problem_statement or ""
        if len(problem) > self.max_problem_len:
            problem = problem[: self.max_problem_len] + "\n... (truncated)"

        steps_block = _summarize_trajectory(
            step_debug,
            max_steps_shown=self.max_steps_shown,
            max_output_per_step=self.max_output_per_step,
        )
        patch_block = _summarize_patch(final_patch, max_chars=self.max_patch_chars)

        if resolved is None:
            outcome_text = "(not evaluated)"
        elif resolved:
            outcome_text = "PASSED (the official tests all passed with this patch)"
        else:
            outcome_text = "FAILED (one or more official tests did not pass, or no patch was produced)"

        user_content = _TRAJ_JUDGE_USER_TEMPLATE.format(
            problem_statement=problem,
            n_steps=len(step_debug),
            steps_block=steps_block,
            patch_block=patch_block,
            outcome_text=outcome_text,
        ) + "\n" + _TRAJ_JUDGE_SCORING_TEMPLATE

        return [
            {"role": "system", "content": _TRAJ_JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

    async def _single_vote(
        self, messages: List[Dict[str, str]], vote_id: int,
    ) -> Dict[str, Any]:
        try:
            from litellm import acompletion
        except ImportError as e:
            raise RuntimeError("litellm is required for TrajectoryJudgeAgent") from e

        try:
            resp = await asyncio.wait_for(
                acompletion(
                    model=self.judge_model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    seed=vote_id,
                ),
                timeout=self.timeout,
            )
            text = (resp.choices[0].message.content or "") if resp and resp.choices else ""
        except Exception as e:
            logger.warning(f"[SWE-TRAJ-JUDGE] vote {vote_id} error: {e}")
            return {"score": 0.0, "raw": "", "error": str(e), "vote_id": vote_id}

        score = _extract_score_from_text(text)
        return {"score": score, "raw": text, "vote_id": vote_id}

    async def judge_trajectory(
        self,
        *,
        problem_statement: str,
        step_debug: List[dict],
        final_patch: str | None = None,
        resolved: bool | None = None,
    ) -> Dict[str, Any]:
        """Judge a full trajectory with ``num_votes`` judge calls and return
        the aggregated score.

        Returns:
            ``{"score": float in [-1, 1], "votes": [...], "model": str, ...}``
        """
        messages = self._build_messages(
            problem_statement=problem_statement,
            step_debug=step_debug,
            final_patch=final_patch,
            resolved=resolved,
        )
        sem = self._get_semaphore()

        async def _bounded_vote(vid: int):
            async with sem:
                return await self._single_vote(messages, vid)

        votes = await asyncio.gather(
            *[_bounded_vote(i) for i in range(self.num_votes)],
            return_exceptions=False,
        )
        scores = [float(v.get("score", 0.0)) for v in votes]
        mean_score = sum(scores) / max(1, len(scores))
        return {
            "status": "ok",
            "score": float(mean_score),
            "votes": votes,
            "model": self.judge_model,
            "num_votes": self.num_votes,
        }
