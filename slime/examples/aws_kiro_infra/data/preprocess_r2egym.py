#!/usr/bin/env python3
"""Preprocess R2E-Gym HF datasets into the parquet format expected by
``aws_kiro_infra`` trainer, using the **r2e** eval strategy.

R2E-Gym (https://r2e-gym.github.io/) has a different reward mechanism than
SWE-Gym:
  * **No** ``FAIL_TO_PASS`` / ``PASS_TO_PASS``
  * **No** ``eval_script`` (images ship with built-in ``/run_tests.sh``)
  * Uses ``expected_output_json`` — a pre-computed dict of test-name → status
  * Docker image name is **explicit** in ``docker_image`` column (not derived
    from instance_id)

Output parquet columns (same shape as SWE-Gym version, different instance fields)::

    prompt       list[dict]  — [{"role":"user","content":"<problem_statement>"}]
    data_source  str         — "r2e-gym"  (★ used by server to pick eval strategy)
    ability      str         — "coding"
    instance     dict        — {instance_id, repo, docker_image, base_commit,
                                problem_statement, expected_output_json, ...}

See ``aws_kiro_infra/data/README.md`` for the full schema spec and how the
trainer + agent server dispatch on ``data_source``.

Dependencies (preprocessing env only)::

    pip install datasets pandas pyarrow

Usage
-----
Default: ``R2E-Gym/R2E-Gym-Subset``::

    python3 preprocess_r2egym.py \\
        --output-path /data/r2e_gym_lite/train.parquet

Full R2E-Gym training set (4578)::

    python3 preprocess_r2egym.py \\
        --data-source R2E-Gym/R2E-Gym-V1 \\
        --split train \\
        --output-path /data/r2e_gym_full/train.parquet

Smoke test (10 instances)::

    python3 preprocess_r2egym.py --max-samples 10 \\
        --output-path /tmp/r2e_smoke.parquet
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from itertools import islice
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("preprocess_r2egym")


# ---------------------------------------------------------------------------
# R2E-Gym HF schema → our instance dict
# ---------------------------------------------------------------------------

# Reference HF schema (from R2E-Gym family datasets):
#   repo_name (str)               → instance.repo
#   docker_image (str)            → instance.docker_image  (★ explicit image)
#   commit_hash (str)             → instance.base_commit
#   parsed_commit_content (str)   → kept as-is (gold diff reference)
#   execution_result_content (str)→ kept as-is
#   modified_files (list[str])    → instance.modified_files
#   relevant_files (list[str])    → instance.relevant_files
#   num_non_test_files (int)      → kept
#   num_non_test_func_methods(int)→ kept
#   num_non_test_lines (int)      → kept
#   prompt (str)                  → unused (we build from problem_statement)
#   problem_statement (str)       → instance.problem_statement
#   expected_output_json (str)    → instance.expected_output_json  (★ gold dict)


def _coerce_str(v: Any) -> str:
    if v is None:
        return ""
    return str(v)


def _coerce_list(v: Any) -> list[str]:
    if v is None:
        return []
    if isinstance(v, (list, tuple)):
        return [str(x) for x in v]
    return [str(v)]


def build_instance_dict(example: dict) -> dict:
    """Map R2E-Gym HF row → our instance dict."""
    # instance_id: R2E doesn't have a direct field; derive from repo + commit prefix
    # (we need something unique & filename-safe for tarball naming)
    repo = _coerce_str(example.get("repo_name"))
    commit = _coerce_str(example.get("commit_hash"))
    docker_image = _coerce_str(example.get("docker_image"))
    # Prefer deriving instance_id from docker_image so it remains stable and unique.
    # Include tag when present to avoid collisions across rows that share image name
    # but differ by tag (common in R2E subsets).
    iid = ""
    if docker_image:
        tail = docker_image.rsplit("/", 1)[-1]  # "<name>:<tag>" or "<name>"
        if ":" in tail:
            name, tag = tail.split(":", 1)
            tag = tag.strip()
            # Keep backward-friendly IDs for ":latest" images while preserving
            # uniqueness for commit-like tags.
            iid = name if not tag or tag.lower() == "latest" else f"{name}__{tag}"
        else:
            iid = tail
    if not iid:
        short = commit[:12] if commit else "unknown"
        iid = f"{repo.replace('/', '__')}-{short}" if repo else f"r2e-{short}"

    expected_output_json = _coerce_str(example.get("expected_output_json"))
    # Validate it's parseable JSON (fail-fast instead of at training time)
    if expected_output_json:
        try:
            json.loads(expected_output_json)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"instance {iid}: expected_output_json is not valid JSON: {e}"
            )
    else:
        raise ValueError(
            f"instance {iid}: expected_output_json is empty — cannot grade"
        )

    problem_statement = _coerce_str(example.get("problem_statement"))
    if not problem_statement:
        raise ValueError(f"instance {iid}: problem_statement is empty")

    inst = {
        "instance_id":           iid,
        "repo":                  repo,
        "docker_image":          docker_image,           # ★ explicit,绕过 name derivation
        "base_commit":           commit,
        "problem_statement":     problem_statement,
        "expected_output_json":  expected_output_json,   # ★ 核心: R2E reward 依据
        # Optional / informational fields (kept for debug):
        "modified_files":        _coerce_list(example.get("modified_files")),
        "relevant_files":        _coerce_list(example.get("relevant_files")),
        "parsed_commit_content": _coerce_str(example.get("parsed_commit_content")),
        "num_non_test_files":    int(example.get("num_non_test_files", 0) or 0),
        "num_non_test_lines":    int(example.get("num_non_test_lines", 0) or 0),
    }
    # Empty stubs for fields the SWE-Gym-style policy gate may look at.
    # R2E path doesn't use them, but keeping consistent dict shape simplifies logging.
    inst.setdefault("FAIL_TO_PASS", [])
    inst.setdefault("PASS_TO_PASS", [])
    inst.setdefault("patch", "")       # R2E gold patch lives in parsed_commit_content; not used at runtime
    inst.setdefault("test_patch", "")
    inst.setdefault("version", commit or "v1")
    inst.setdefault("eval_script", "")  # R2E uses image's built-in /run_tests.sh
    return inst


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-source", default="R2E-Gym/R2E-Gym-Subset",
                    help="HuggingFace dataset ID (default: %(default)s). "
                         "Common options: R2E-Gym/R2E-Gym-Subset, "
                         "R2E-Gym/R2E-Gym-Lite (230), "
                         "R2E-Gym/R2E-Gym-V1 (4578 train)")
    ap.add_argument("--config", default="default", help="HF dataset config name")
    ap.add_argument("--split", default="train", help="HF split (default: train)")
    ap.add_argument("--data-source-tag", default="r2e-gym",
                    help="Value written into the 'data_source' column. "
                         "Must contain 'r2e-gym' so swe_agent_server picks the "
                         "r2e eval strategy.")
    ap.add_argument("--output-path", required=True,
                    help="Output .parquet path (directories will be created).")
    ap.add_argument("--max-samples", type=int, default=0,
                    help="Limit to first N. 0 = full split.")
    ap.add_argument(
        "--streaming",
        action="store_true",
        help=(
            "Use HF streaming mode to avoid materializing full split before processing. "
            "Recommended for smoke tests with --max-samples."
        ),
    )
    args = ap.parse_args()

    # Sanity: data_source_tag must include 'r2e' so server dispatches correctly
    if "r2e" not in args.data_source_tag.lower():
        logger.warning(
            "--data-source-tag=%r does not contain 'r2e'. server.patch_utils."
            "detect_eval_strategy may route this to the SWE-Bench harness path "
            "by mistake, which will fail for R2E-Gym instances (no FAIL_TO_PASS).",
            args.data_source_tag,
        )

    import pandas as pd
    import datasets

    logger.info("Loading %s [%s/%s] ...", args.data_source, args.config, args.split)
    if args.streaming:
        ds = datasets.load_dataset(
            args.data_source,
            args.config,
            split=args.split,
            streaming=True,
        )
        if args.max_samples > 0:
            ds_iter = islice(ds, args.max_samples)
            logger.info("Streaming mode enabled; reading first %d instances", args.max_samples)
        else:
            ds_iter = ds
            logger.info("Streaming mode enabled; reading full split (unknown total size)")
        total = None
    else:
        ds = datasets.load_dataset(args.data_source, args.config)[args.split]
        total = len(ds)
        if args.max_samples > 0 and args.max_samples < total:
            ds = ds.select(range(args.max_samples))
            logger.info("Sliced to first %d of %d", args.max_samples, total)
        else:
            logger.info("Using all %d instances", total)
        ds_iter = ds

    rows = []
    n_errors = 0
    for i, example in enumerate(ds_iter):
        try:
            inst = build_instance_dict(example)
            rows.append({
                "prompt":      [{"role": "user", "content": inst["problem_statement"]}],
                "data_source": args.data_source_tag,
                "ability":     "coding",
                "instance":    inst,
            })
            if args.streaming:
                if (i + 1) % 10 == 0:
                    logger.info("  processed %d%s  (last: %s)", i + 1, "" if total is None else f" / {total}", inst["instance_id"])
            elif (i + 1) % 100 == 0:
                logger.info("  processed %d / %d  (last: %s)", i + 1, len(ds), inst["instance_id"])
        except Exception as e:
            n_errors += 1
            iid_hint = example.get("docker_image") or example.get("repo_name") or "?"
            logger.warning("  skipping (%s): %s", iid_hint, e)

    logger.info("Converted %d rows (%d skipped)", len(rows), n_errors)
    if not rows:
        logger.error("No rows produced; aborting.")
        return 3

    out_path = Path(args.output_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_parquet(out_path, index=False)
    logger.info("Wrote %d rows → %s (%.1f MB)",
                len(df), out_path, out_path.stat().st_size / 1e6)

    # Sanity check
    df2 = pd.read_parquet(out_path)
    r = df2.iloc[0]
    inst = r["instance"]
    logger.info("Sanity check — first row:")
    logger.info("  data_source         : %s", r["data_source"])
    logger.info("  instance_id         : %s", inst.get("instance_id"))
    logger.info("  docker_image        : %s", inst.get("docker_image"))
    logger.info("  repo                : %s", inst.get("repo"))
    logger.info("  base_commit         : %s", inst.get("base_commit")[:12] + "...")
    eoj_preview = inst.get("expected_output_json", "")[:200].replace("\n", " ")
    logger.info("  expected_output_json: %s...", eoj_preview)

    return 0


if __name__ == "__main__":
    sys.exit(main())
