#!/usr/bin/env python3
"""Prepare immutable plans for the Milestone M stationary-rate study."""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Any

from tools.prepare_milestone_m_measurement import (
    TRACE_SHA256,
    _git_commit,
    frozen_placement,
    sha256_file,
)


RATE_SWEEP_RATES = (2.0, 4.0, 6.0, 8.0)
RATE_SWEEP_REPETITIONS = (1, 2, 3)
RATE_SWEEP_RANDOMIZATION_SEED = 20261002


def rate_sweep_plan(root: Path, trace: Path) -> dict[str, Any]:
    trace_sha = sha256_file(trace)
    if trace_sha != TRACE_SHA256:
        raise ValueError(f"frozen trace hash drift: {trace_sha}")
    windows = [
        {"request_rate": rate, "repetition": repetition}
        for rate in RATE_SWEEP_RATES
        for repetition in RATE_SWEEP_REPETITIONS
    ]
    random.Random(RATE_SWEEP_RANDOMIZATION_SEED).shuffle(windows)
    for order, window in enumerate(windows):
        window["order"] = order
    return {
        "schema_version": 1,
        "study": "milestone_m_afd_stationary_rate_sweep",
        "git_commit": _git_commit(root),
        "model": "Qwen/Qwen3-30B-A3B",
        "vllm_version": "0.26.0",
        "plugin_commit": "4c2541a53ec8ad507508073eac81ac9feecc6c9a",
        "topology": "2A2F",
        "attention_devices": [0, 1],
        "ffn_devices": [2, 3],
        "execution_mode": "FULL_DECODE_ONLY",
        "trace": "trace.jsonl",
        "trace_sha256": trace_sha,
        "requests": 300,
        "max_output_tokens": 128,
        "temperature": 0.0,
        "seed": 0,
        "placement": frozen_placement(root, "P_balanced"),
        "target_clocks_mhz": [2520, 2520, 2520, 2520],
        "rates": list(RATE_SWEEP_RATES),
        "repetitions": list(RATE_SWEEP_REPETITIONS),
        "randomization_seed": RATE_SWEEP_RANDOMIZATION_SEED,
        "windows": windows,
        "expected_runs": len(windows),
        "routing_profiling": False,
        "custom_kernel_timing": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    args = parser.parse_args()

    root = Path.cwd().resolve()
    trace = args.trace.resolve(strict=True)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest = rate_sweep_plan(root, trace)
    shutil.copyfile(trace, output / "trace.jsonl")
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(output), **manifest}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
