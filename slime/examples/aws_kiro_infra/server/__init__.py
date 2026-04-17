"""CPU-side agent server (runs on ``swe_agent_workers`` pod pool).

Mirrors Kiro's ``remote_rollout_server/sglang_remote_rollout.py`` role, but
uses **mini-swe-agent scaffold** (bash backtick + submit sentinel) instead of
the Kiro-native Strands + Qwen tool_call.

Architecture:
  - Trainer (GPU pod) POSTs ``/generate_trajectory`` to this server
  - Server runs the full mini-swe-agent loop IN-PROCESS:
      * tokenizer (apply_chat_template → input_ids)
      * SGLang /generate (via sglang_client)
      * bash parsing + local docker exec (via docker_ops)
      * submit sentinel + git patch extraction
      * eval harness + policy gate
  - Returns per-turn token_ids / loss_mask / logprobs so trainer can build a
    slime Sample with byte-perfect token fidelity

Entry point: ``swe_agent_server:app`` (FastAPI).
"""
