"""Thin async client for SGLang's native ``/generate`` endpoint.

We POST ``input_ids`` (pre-tokenized) rather than chat messages, because
trajectory-mode RL needs byte-perfect per-turn token_ids. The native
endpoint returns ``output_token_logprobs`` from which we derive both the
token ids and logprobs — slime's convention.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

logger = logging.getLogger("swe_agent_server.sglang_client")


class SGLangClient:
    def __init__(
        self,
        router_url: str,
        timeout: float = 600.0,
        max_connections: int = 256,
    ):
        """``router_url`` is the SGLang router, e.g. ``http://<ip>:30000``.
        We append ``/generate`` at request time."""
        self.router_url = router_url.rstrip("/")
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout, connect=30.0),
            limits=httpx.Limits(
                max_connections=max_connections,
                max_keepalive_connections=max(16, max_connections // 4),
            ),
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def generate(
        self,
        input_ids: list[int],
        sampling_params: dict[str, Any],
        return_logprob: bool = True,
    ) -> dict[str, Any]:
        """POST to ``<router>/generate``.

        Returns a dict with:
          * ``text``: decoded assistant text
          * ``output_token_ids``: list[int] generated tokens (from output_token_logprobs)
          * ``output_logprobs``: list[float]
          * ``meta_info``: raw passthrough
        """
        payload = {
            "input_ids": input_ids,
            "sampling_params": sampling_params,
            "return_logprob": return_logprob,
        }
        r = await self._client.post(f"{self.router_url}/generate", json=payload)
        r.raise_for_status()
        output = r.json()

        assistant_text: str = output.get("text", "")
        meta_info = output.get("meta_info", {}) or {}
        raw_logprobs = meta_info.get("output_token_logprobs") or []
        if raw_logprobs:
            output_token_ids = [x[1] for x in raw_logprobs]
            output_logprobs = [x[0] for x in raw_logprobs]
        else:
            # Fallback: router didn't return logprobs. Caller can re-tokenize if needed.
            output_token_ids = []
            output_logprobs = []

        return {
            "text": assistant_text,
            "output_token_ids": output_token_ids,
            "output_logprobs": output_logprobs,
            "meta_info": meta_info,
        }
