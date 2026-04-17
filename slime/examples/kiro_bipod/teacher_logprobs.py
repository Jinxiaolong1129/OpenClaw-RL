"""Query an external teacher SGLang server for top-K logprobs.

The teacher server is a standard SGLang instance hosting a (typically larger)
teacher model.  It is queried with the student's full token sequence
(prompt + multi-turn response) and returns per-position top-K log-probs
over the vocabulary, starting from the first response token.

To avoid OOMing the teacher on long multi-turn trajectories, we only
request logprobs for *trainable* positions (loss_mask == 1).  Each
contiguous trainable span is queried as a separate request with
``input_ids`` truncated to the span's end and ``logprob_start_len``
set to just before the span.  SGLang's radix/prefix cache reuses KV
state across overlapping prefixes, so the redundant prefill cost is low.

Environment variables:
    TEACHER_URL:  SGLang /generate endpoint, e.g. "http://teacher-host:30000/generate"
    TEACHER_TOPK: Number of top-K logprobs to request (default: 50, overridden by --distill-topk)
    TEACHER_MAX_CONCURRENCY: Max concurrent teacher requests (default: 8)
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

import httpx
import torch

logger = logging.getLogger(__name__)

TEACHER_URL = os.environ.get("TEACHER_URL", "")
TEACHER_MAX_CONCURRENCY = int(os.environ.get("TEACHER_MAX_CONCURRENCY", "8"))

_semaphore: asyncio.Semaphore | None = None
_client: httpx.AsyncClient | None = None

# Padding value for logprobs: must be very negative (prob ≈ 0), NOT 0.0
# (logprob=0 means prob=1, which would corrupt the tail trick).
_PAD_LP = -1e10


def _get_semaphore() -> asyncio.Semaphore:
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(TEACHER_MAX_CONCURRENCY)
    return _semaphore


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(
            timeout=httpx.Timeout(1800.0, connect=30.0),
            limits=httpx.Limits(
                max_connections=TEACHER_MAX_CONCURRENCY * 2,
                max_keepalive_connections=TEACHER_MAX_CONCURRENCY,
            ),
        )
    return _client


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_trainable_spans(loss_mask: list[int] | None) -> list[tuple[int, int]]:
    """Return ``[(start, end), ...]`` for contiguous runs of ``loss_mask == 1``."""
    if not loss_mask:
        return []
    spans: list[tuple[int, int]] = []
    i = 0
    while i < len(loss_mask):
        if loss_mask[i] == 1:
            start = i
            while i < len(loss_mask) and loss_mask[i] == 1:
                i += 1
            spans.append((start, i))
        else:
            i += 1
    return spans


def _parse_sglang_logprobs(
    inp_top: list, K: int,
) -> tuple[list[list[float]], list[list[int]]]:
    """Parse SGLang ``input_top_logprobs`` into (logprobs, indices) lists."""
    all_logprobs: list[list[float]] = []
    all_indices: list[list[int]] = []
    for pos_data in inp_top:
        if isinstance(pos_data, (list, tuple)):
            row_lp: list[float] = []
            row_idx: list[int] = []
            for entry in pos_data:
                if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                    row_lp.append(float(entry[0]) if entry[0] is not None else _PAD_LP)
                    row_idx.append(int(entry[1]))
                elif isinstance(entry, dict):
                    row_lp.append(float(entry.get("logprob", _PAD_LP)))
                    row_idx.append(int(entry.get("token_id", 0)))
                else:
                    row_lp.append(_PAD_LP)
                    row_idx.append(0)
            while len(row_lp) < K:
                row_lp.append(_PAD_LP)
                row_idx.append(0)
            all_logprobs.append(row_lp[:K])
            all_indices.append(row_idx[:K])
        else:
            all_logprobs.append([_PAD_LP] * K)
            all_indices.append(list(range(K)))
    return all_logprobs, all_indices


async def _query_span(
    url: str,
    input_ids: list[int],
    prompt_length: int,
    span_start: int,
    span_end: int,
    K: int,
) -> tuple[int, int, list[list[float]], list[list[int]]] | None:
    """Query teacher logprobs for a single trainable span.

    Args:
        url: Teacher SGLang ``/generate`` endpoint.
        input_ids: Full token sequence (prompt + response).
        prompt_length: Number of prompt tokens.
        span_start: Start of the trainable span (response-relative, inclusive).
        span_end: End of the trainable span (response-relative, exclusive).
        K: Number of top logprobs per position.

    Returns:
        ``(span_start, span_end, logprobs, indices)`` or *None* on failure.
    """
    span_len = span_end - span_start

    # Truncate input_ids to the end of this span — tokens after the span
    # are not needed for computing logprobs within the span.
    truncated_ids = input_ids[:prompt_length + span_end]

    # Start one position before the span so that after SGLang's mandatory
    # None at position 0, we still get logprobs for the full span.
    start_len = max(0, prompt_length + span_start - 1)

    payload = {
        "input_ids": truncated_ids,
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": 0,
            "skip_special_tokens": False,
        },
        "return_logprob": True,
        "logprob_start_len": start_len,
        "top_logprobs_num": K,
    }

    sem = _get_semaphore()
    client = _get_client()

    try:
        async with sem:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            result = resp.json()
    except Exception as e:
        logger.warning(
            "[BIPOD] Teacher span query failed (span %d-%d): %s",
            span_start, span_end, e,
        )
        return None

    meta = result.get("meta_info", {}) if isinstance(result, dict) else {}
    inp_top = meta.get("input_top_logprobs")
    if not isinstance(inp_top, list):
        logger.warning(
            "[BIPOD] No input_top_logprobs for span %d-%d", span_start, span_end,
        )
        return None

    all_logprobs, all_indices = _parse_sglang_logprobs(inp_top, K)

    # Skip first entry (SGLang's mandatory None at position 0).
    if len(all_logprobs) > 1:
        all_logprobs = all_logprobs[1:]
        all_indices = all_indices[1:]

    # Align to span length (right-align, pad front if needed).
    if len(all_logprobs) >= span_len:
        all_logprobs = all_logprobs[-span_len:]
        all_indices = all_indices[-span_len:]
    else:
        pad_len = span_len - len(all_logprobs)
        all_logprobs = [[_PAD_LP] * K] * pad_len + all_logprobs
        all_indices = [list(range(K))] * pad_len + all_indices

    return (span_start, span_end, all_logprobs, all_indices)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def query_teacher_topk(
    input_ids: list[int],
    response_length: int,
    loss_mask: list[int] | None,
    K: int = 50,
    teacher_url: str | None = None,
) -> dict[str, Any] | None:
    """Query teacher for top-K logprobs at trainable positions only.

    Instead of requesting logprobs for every response position (which can
    OOM the teacher on long multi-turn trajectories), this function
    identifies contiguous trainable spans from *loss_mask* and queries
    each span separately.  Non-trainable positions receive pad values
    that are masked out by the distillation loss.

    Args:
        input_ids: Full token sequence (prompt + response), length N.
        response_length: Number of response tokens (last ``response_length``
            tokens of ``input_ids``).
        loss_mask: Binary mask of length ``response_length``.  1 = trainable
            (model-generated), 0 = non-trainable (tool output / injected).
            If *None*, the entire response is treated as one trainable span.
        K: Number of top logprobs per position.
        teacher_url: Override for TEACHER_URL env var.

    Returns:
        Dict with ``log_probs`` (float32 tensor ``[response_length, K]``)
        and ``indices`` (long tensor ``[response_length, K]``),
        or *None* on failure.
    """
    url = teacher_url or TEACHER_URL
    if not url:
        logger.warning("[BIPOD] TEACHER_URL not set, skipping teacher query")
        return None

    prompt_length = len(input_ids) - response_length

    # Identify trainable spans.  If loss_mask is not provided, treat the
    # entire response as a single trainable span (backward compat).
    spans = _get_trainable_spans(loss_mask)
    if not spans and loss_mask is not None:
        logger.warning("[BIPOD] No trainable positions in loss_mask, skipping")
        return None
    if not spans:
        # loss_mask is None → treat entire response as one span.
        spans = [(0, response_length)]

    total_trainable = sum(e - s for s, e in spans)
    logger.info(
        "[BIPOD] Querying teacher for %d trainable positions across %d span(s) "
        "(response_length=%d, %.1f%% reduction)",
        total_trainable, len(spans), response_length,
        (1 - total_trainable / max(response_length, 1)) * 100,
    )

    # Query spans sequentially in position order so that SGLang's prefix
    # cache can reuse KV state from earlier spans.  Each request only
    # prefills the NEW tokens since the previous span's end, so total
    # prefill work equals a single full-sequence request.
    results = []
    for s, e in spans:
        r = await _query_span(url, input_ids, prompt_length, s, e, K)
        if r is None:
            logger.warning("[BIPOD] Span query failed (span %d-%d)", s, e)
            return None
        results.append(r)

    # Assemble [response_length, K] tensors.  Non-trainable positions
    # keep pad values; loss_mask ensures they don't affect training.
    full_logprobs = [[_PAD_LP] * K for _ in range(response_length)]
    full_indices = [[0] * K for _ in range(response_length)]

    for span_start, span_end, span_lps, span_ids in results:
        for i in range(span_end - span_start):
            full_logprobs[span_start + i] = span_lps[i]
            full_indices[span_start + i] = span_ids[i]

    return {
        "log_probs": torch.tensor(full_logprobs, dtype=torch.float32),
        "indices": torch.tensor(full_indices, dtype=torch.long),
    }
