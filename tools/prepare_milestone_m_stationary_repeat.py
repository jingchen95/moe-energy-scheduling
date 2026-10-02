#!/usr/bin/env python3
"""Prepare the frozen 8 request/s Milestone M stationary repeat matrix."""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Any

from tools.prepare_milestone_m_measurement import (
    CONFIGURATIONS,
    TRACE_SHA256,
    _git_commit,
    frozen_placement,
    sha256_file,
)


SELECTED_REQUEST_RATE = 8.0
REPETITIONS = (1, 2, 3, 4, 5)
RANDOMIZATION_SEED = 20261003
RATE_SWEEP_JOB_ID = 270840
RATE_SWEEP_STUDY = "results/raw/milestone-m-rate-sweep-rerun-20261001T161703Z"


def stationary_repeat_plan(root: Path, trace: Path) -> dict[str, Any]:
    trace_sha = sha256_file(trace)
    if trace_sha != TRACE_SHA256:
        raise ValueError(f"frozen trace hash drift: {trace_sha}")
    indices = list(range(len(CONFIGURATIONS)))
    random.Random(RANDOMIZATION_SEED).shuffle(indices)
    rows = []
    for order, index in enumerate(indices):
        config_id, placement_name, ffn_clocks = CONFIGURATIONS[index]
        rows.append({
            "order": order,
            "config_id": config_id,
            "placement": frozen_placement(root, placement_name),
            "attention_clocks_mhz": [2520, 2520],
            "ffn_clocks_mhz": list(ffn_clocks),
            "target_clocks_mhz": [2520, 2520, *ffn_clocks],
            "repetitions": list(REPETITIONS),
        })
    return {
        "schema_version": 1,
        "study": "milestone_m_afd_stationary_repeat",
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
        "request_rate": SELECTED_REQUEST_RATE,
        "override_arrival_rate": True,
        "requests": 300,
        "max_output_tokens": 128,
        "temperature": 0.0,
        "seed": 0,
        "randomization_seed": RANDOMIZATION_SEED,
        "configuration_order": [row["config_id"] for row in rows],
        "configurations": rows,
        "repetitions": list(REPETITIONS),
        "expected_runs": len(rows) * len(REPETITIONS),
        "rate_selection": {
            "source_job_id": RATE_SWEEP_JOB_ID,
            "source_study": RATE_SWEEP_STUDY,
            "rule": "highest tested rate with completion throughput tracking offered load and no within-window TTFT accumulation",
            "selected_before_placement_repeat": True,
        },
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
    manifest = stationary_repeat_plan(root, trace)
    shutil.copyfile(trace, output / "trace.jsonl")
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(output), **manifest}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
