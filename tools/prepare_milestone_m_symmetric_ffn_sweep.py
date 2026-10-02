#!/usr/bin/env python3
"""Prepare the fixed-placement symmetric FFN frequency sweep."""

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


SYMMETRIC_FREQUENCIES_MHZ = (2520, 2220, 2070, 1635, 1005)
REPETITIONS = (1, 2, 3)
RANDOMIZATION_SEED = 20261004
REQUEST_RATE = 8.0


def symmetric_sweep_plan(root: Path, trace: Path) -> dict[str, Any]:
    trace_sha = sha256_file(trace)
    if trace_sha != TRACE_SHA256:
        raise ValueError(f"frozen trace hash drift: {trace_sha}")
    order = list(range(len(SYMMETRIC_FREQUENCIES_MHZ)))
    random.Random(RANDOMIZATION_SEED).shuffle(order)
    placement = frozen_placement(root, "P_balanced")
    rows = []
    for experiment_order, frequency_index in enumerate(order):
        frequency = SYMMETRIC_FREQUENCIES_MHZ[frequency_index]
        rows.append({
            "order": experiment_order,
            "config_id": f"S{frequency_index}",
            "frequency_mhz": frequency,
            "placement": placement,
            "attention_clocks_mhz": [2520, 2520],
            "ffn_clocks_mhz": [frequency, frequency],
            "target_clocks_mhz": [2520, 2520, frequency, frequency],
            "repetitions": list(REPETITIONS),
        })
    return {
        "schema_version": 1,
        "study": "milestone_m_afd_symmetric_ffn_frequency_sweep",
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
        "request_rate": REQUEST_RATE,
        "override_arrival_rate": True,
        "requests": 300,
        "max_output_tokens": 128,
        "temperature": 0.0,
        "seed": 0,
        "placement": placement,
        "frequencies_mhz": list(SYMMETRIC_FREQUENCIES_MHZ),
        "supported_clock_evidence": "results/raw/milestone-m-static-hh-20260930T081823Z/clocks-hh.json",
        "randomization_seed": RANDOMIZATION_SEED,
        "configuration_order": [row["config_id"] for row in rows],
        "configurations": rows,
        "repetitions": list(REPETITIONS),
        "conditioning_traces_per_configuration": 1,
        "expected_runs": len(rows) * len(REPETITIONS),
        "routing_profiling": False,
        "custom_kernel_timing": False,
        "placement_optimization": False,
        "dynamic_dvfs": False,
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
    manifest = symmetric_sweep_plan(root, trace)
    shutil.copyfile(trace, output / "trace.jsonl")
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(output), **manifest}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
