#!/usr/bin/env python3
"""Preprocess a SWE dataset (SWE-Gym / SWE-Bench / variants) into the parquet
format expected by ``aws_kiro_infra`` trainer.

Produces one ``.parquet`` with columns::

    prompt      list[dict]   — [{"role":"user","content":"<problem_statement>"}]
    data_source str          — "swe-gym" | "swe-bench" | "swe-bench-verified" | ...
    ability     str          — "coding"
    instance    dict         — full SWE fields + pre-rendered ``eval_script``

See ``aws_kiro_infra/data/README.md`` for the full schema spec.

Dependencies (install inside the preprocessing env, NOT required at train time)::

    pip install datasets pandas pyarrow
    pip install git+https://github.com/SWE-Gym/SWE-Gym@main     # make_test_spec (swegym)
    # or
    pip install swebench                                         # make_test_spec (swebench)

Usage
-----
Default: SumanthRH/SWE-Gym-Subset (293 instances)::

    python3 preprocess_swegym.py \\
        --output-path /data/swegym_for_kiro/train.parquet

Smoke test (10 instances)::

    python3 preprocess_swegym.py --max-samples 10 \\
        --output-path /data/swegym_for_kiro/train_10.parquet

SWE-Bench Verified::

    python3 preprocess_swegym.py \\
        --data-source SumanthRH/SWE-bench_Verified \\
        --split test \\
        --data-source-tag swe-bench-verified \\
        --output-path /data/swebench_verified/eval.parquet
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("preprocess_swegym")

# ---------------------------------------------------------------------------
# Instance field normalization
# ---------------------------------------------------------------------------

_REQUIRED = [
    "instance_id", "repo", "base_commit", "problem_statement",
    "patch", "test_patch", "FAIL_TO_PASS", "PASS_TO_PASS", "version",
]
_OPTIONAL = ["hints_text", "created_at", "environment_setup_commit"]


def _coerce_list(value: Any) -> list:
    """Some HF datasets encode FAIL_TO_PASS / PASS_TO_PASS as JSON strings."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x) for x in value]
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, list):
                return [str(x) for x in parsed]
            return [str(parsed)]
        except json.JSONDecodeError:
            # Sometimes it's a newline-separated list
            return [line.strip() for line in stripped.splitlines() if line.strip()]
    # numpy arrays, etc.
    try:
        return [str(x) for x in list(value)]
    except TypeError:
        return [str(value)]


def build_instance_dict(example: dict) -> dict:
    inst = {}
    for key in _REQUIRED:
        if key not in example:
            raise KeyError(f"instance missing required field {key!r}: "
                           f"have {list(example.keys())}")
        inst[key] = example[key]
    inst["FAIL_TO_PASS"] = _coerce_list(inst["FAIL_TO_PASS"])
    inst["PASS_TO_PASS"] = _coerce_list(inst["PASS_TO_PASS"])
    for key in _OPTIONAL:
        if key in example:
            inst[key] = example[key]
    # Cast everything except the lists to str
    for k, v in list(inst.items()):
        if k in ("FAIL_TO_PASS", "PASS_TO_PASS"):
            continue
        if v is None:
            inst[k] = ""
        else:
            inst[k] = str(v)
    return inst


# ---------------------------------------------------------------------------
# eval_script rendering via swegym / swebench harness
# ---------------------------------------------------------------------------

def _get_make_test_spec(data_source_tag: str):
    """Return a ``make_test_spec`` callable.

    Tries swegym first (for swe-gym tag) then swebench (for everything else).
    Returns None if neither is installed.
    """
    tag = data_source_tag.lower()
    if "swe-gym" in tag:
        try:
            from swegym.harness.test_spec import make_test_spec
            logger.info("Using swegym.harness.test_spec.make_test_spec")
            return make_test_spec
        except ModuleNotFoundError:
            logger.warning("swegym not installed, falling back to swebench")
    try:
        from swebench.harness.test_spec.test_spec import make_test_spec
        logger.info("Using swebench.harness.test_spec.test_spec.make_test_spec")
        return make_test_spec
    except ModuleNotFoundError:
        try:
            from swebench.harness.test_spec import make_test_spec  # older API
            logger.info("Using swebench.harness.test_spec.make_test_spec (legacy)")
            return make_test_spec
        except ModuleNotFoundError:
            return None


def render_eval_script(instance: dict, make_test_spec) -> str:
    inst = copy.deepcopy(instance)
    # swegym/swebench convention
    inst["instance_id"] = inst["instance_id"].lower()
    if "version" not in inst and "base_commit" in inst:
        inst["version"] = inst["base_commit"]
    test_spec = make_test_spec(inst)
    script = getattr(test_spec, "eval_script", "")
    if not isinstance(script, str) or not script.strip():
        raise RuntimeError(f"make_test_spec returned empty eval_script for {inst['instance_id']}")
    return script


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-source", default="SumanthRH/SWE-Gym-Subset",
                    help="HuggingFace dataset ID (default: %(default)s)")
    ap.add_argument("--config", default="default", help="HF dataset config name")
    ap.add_argument("--split", default="train", help="HF split (default: %(default)s)")
    ap.add_argument("--data-source-tag", default="swe-gym",
                    help="Value to write into the 'data_source' column. "
                         "Must contain 'swe-gym' or 'swe-bench' so that "
                         "server.mini_swe_agent.get_docker_image_name picks "
                         "the right image name scheme.")
    ap.add_argument("--output-path", required=True,
                    help="Output .parquet path (directories will be created).")
    ap.add_argument("--max-samples", type=int, default=0,
                    help="Limit to first N samples. 0 = full split.")
    ap.add_argument("--skip-eval-script", action="store_true",
                    help="Do NOT render eval_script. Use only if you'll render "
                         "it later on the CPU agent pod (NOT recommended — the "
                         "agent pod image may not have swegym/swebench).")
    ap.add_argument("--prompt-template", default="user",
                    choices=["user", "raw"],
                    help="'user' (default) wraps problem_statement in a user "
                         "message list; 'raw' writes the raw string into prompt.")
    args = ap.parse_args()

    import pandas as pd
    import datasets

    # ----- Load -----
    logger.info("Loading %s [%s/%s] ...", args.data_source, args.config, args.split)
    ds = datasets.load_dataset(args.data_source, args.config)[args.split]
    total = len(ds)
    if args.max_samples > 0 and args.max_samples < total:
        ds = ds.select(range(args.max_samples))
        logger.info("Sliced to first %d of %d", args.max_samples, total)
    else:
        logger.info("Using all %d instances", total)

    # ----- Resolve make_test_spec -----
    make_test_spec = None
    if not args.skip_eval_script:
        make_test_spec = _get_make_test_spec(args.data_source_tag)
        if make_test_spec is None:
            logger.error(
                "Neither swegym nor swebench is installed. Install one of them "
                "first, OR pass --skip-eval-script (not recommended)."
            )
            return 2

    # ----- Convert -----
    rows = []
    n_errors = 0
    for i, example in enumerate(ds):
        iid = example.get("instance_id", f"instance_{i}")
        try:
            inst = build_instance_dict(example)
            if make_test_spec is not None:
                inst["eval_script"] = render_eval_script(inst, make_test_spec)
            else:
                inst["eval_script"] = ""

            problem_statement = inst["problem_statement"]
            if args.prompt_template == "user":
                prompt = [{"role": "user", "content": problem_statement}]
            else:
                prompt = problem_statement

            rows.append({
                "prompt":      prompt,
                "data_source": args.data_source_tag,
                "ability":     "coding",
                "instance":    inst,
            })
            if (i + 1) % 50 == 0:
                logger.info("  processed %d / %d  (last: %s)", i + 1, len(ds), iid)
        except Exception as e:
            n_errors += 1
            logger.warning("  skipping %s: %s", iid, e)

    logger.info("Converted %d rows (%d skipped due to error)", len(rows), n_errors)

    if not rows:
        logger.error("No rows produced; aborting.")
        return 3

    # ----- Write parquet -----
    out_path = Path(args.output_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_parquet(out_path, index=False)
    logger.info("Wrote %d rows → %s (%.1f MB)",
                len(df), out_path, out_path.stat().st_size / 1e6)

    # Sanity check
    logger.info("Sanity check — re-read first row:")
    df2 = pd.read_parquet(out_path)
    r = df2.iloc[0]
    logger.info("  columns          : %s", list(df2.columns))
    logger.info("  data_source      : %s", r["data_source"])
    logger.info("  ability          : %s", r["ability"])
    logger.info("  prompt[0].role   : %s", r["prompt"][0].get("role")
                if hasattr(r["prompt"], "__getitem__") else "<raw str>")
    inst_keys = list(r["instance"].keys()) if hasattr(r["instance"], "keys") else []
    logger.info("  instance keys    : %s", inst_keys)
    eval_preview = r["instance"].get("eval_script", "")[:120] if inst_keys else ""
    logger.info("  eval_script head : %s%s", eval_preview, "..." if eval_preview else "<MISSING>")

    return 0


if __name__ == "__main__":
    sys.exit(main())
