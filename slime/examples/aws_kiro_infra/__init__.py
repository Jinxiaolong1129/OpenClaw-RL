"""SWE-RL AWS Kiro infra variant — Kiro-on-Strands (KoS) 3-pool topology.

Deployment is fully aligned with Kiro team's **latest production** (SageMaker
HyperPod / EKS), with ONE difference: the agent scaffold is
**mini-swe-agent** (bash backtick + submit sentinel) instead of Strands +
Qwen tool_call.

Three PyTorchJobs coordinate via shared-PVC env files (mirrors
``slime/examples/kiro_agent/sglang_rollout_server/job_yamsl/``):

    Job 1 — SGLang pool (GPU)      k8s/launch_sglang_{4b,30b}.yaml
    ────────────────────────────────────────────────────────────────────
    Per pod: 1 SGLang engine (Ray actor, TP=8)
    Head pod: + SGLang router
    Produces: <OUT>/<RUN>/sglang_external_rollout.env

    Job 2 — swe_agent pool (CPU)   k8s/launch_swe_agent_workers.yaml
    ────────────────────────────────────────────────────────────────────
    Per pod: dockerd + FastAPI (server.swe_agent_server:app)
    FastAPI runs the mini-swe-agent loop IN-PROCESS:
      * tokenizer (apply_chat_template → input_ids)
      * SGLang router call (via server/sglang_client.py)
      * docker exec (via server/docker_ops.py)
      * eval harness + policy gate (via server/patch_utils.py)
    Returns per-turn token_ids / loss_mask / logprobs to trainer
    Produces: <OUT>/<RUN>/swe_agents.env (each pod flock-appends its URL)

    Job 3 — trainer pool (GPU)     k8s/launch_trainer_{4b,30b}.yaml
    ────────────────────────────────────────────────────────────────────
    Waits for BOTH env files, starts its OWN Ray cluster, runs
    ``slime/train.py`` (4B sync) or ``slime/train_async.py`` (30B async)
    with --rollout-external.

Mapping to Kiro reference (``slime/examples/kiro_agent/``):
  - ``sglang_rollout_server/launch_sglang_only.sh``  → ``scripts/launch_sglang.sh``
  - ``sglang_rollout_server/launch_kos_only.sh``     → ``scripts/launch_swe_agent_workers_cpu.sh``
                                                         (KoS FastAPI → swe_agent_server)
  - ``sglang_rollout_server/launch_external_sglang.py`` → reused verbatim
  - ``job_yamsl/launch_sglang_only_async.yaml``      → ``k8s/launch_sglang_{4b,30b}.yaml``
  - ``job_yamsl/launch_kos_only_async.yaml``         → ``k8s/launch_swe_agent_workers.yaml``
  - ``jobs/slime_train_async_debug_run.yaml``        → ``k8s/launch_trainer_{4b,30b}.yaml``
  - ``kiro_generate_with_kos.py`` (Strands scaffold) → ``generate_kiro.py`` +
                                                         ``generate_with_swe_remote.py`` +
                                                         ``server/mini_swe_agent.py``

Algorithm recipes:
  * 4B  (``run_swe_rl_4b_kiro_trainer.sh``):   DAPO 0.2/0.28 clip, sync, aux LLM judge on
  * 30B (``run_swe_rl_30b_kiro_trainer.sh``):  symmetric 0.2/0.2, async, TIS on,
                                                calculate-per-token-loss, NO aux judge
                                                (1:1 with Kiro's ``run-qwen3-30b-kiro-kos-async-debug-run.sh``)
"""
