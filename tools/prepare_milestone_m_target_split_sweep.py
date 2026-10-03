#!/usr/bin/env python3
"""Prepare the fixed heterogeneous-frequency target-split experiment."""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from pathlib import Path
from typing import Any

from src.afd_static_placement import load_placement, sha256_file
from tools.prepare_milestone_m_measurement import (
    TRACE_SHA256,
    _git_commit,
    canonical_mapping_sha256,
)


RANDOMIZATION_SEED = 20261005
REQUEST_RATE = 8.0
REPETITIONS = (1, 2, 3)
ORIENTATIONS = {
    "MH": {"ffn_clocks_mhz": [1635, 2070], "slow_rank": 0},
    "HM": {"ffn_clocks_mhz": [2070, 1635], "slow_rank": 1},
}


def _placement_record(root: Path, row: dict[str, str]) -> dict[str, Any]:
    relative = Path(row["placement_file"])
    path = (root / relative).resolve(strict=True)
    mapping, observed_sha, observed_name = load_placement(str(path))
    if observed_sha != row["placement_sha256"]:
        raise ValueError(f"placement hash drift: {path}")
    if observed_name != row["placement"]:
        raise ValueError(f"placement name drift: {path}")
    canonical = canonical_mapping_sha256(mapping)
    if canonical != row["canonical_mapping_sha256"]:
        raise ValueError(f"canonical mapping hash drift: {path}")
    return {
        "name": observed_name,
        "path": str(relative),
        "sha256": observed_sha,
        "canonical_mapping_sha256": canonical,
    }


def target_split_plan(
    root: Path,
    trace: Path,
    placement_summary: Path,
) -> dict[str, Any]:
    trace_sha = sha256_file(trace)
    if trace_sha != TRACE_SHA256:
        raise ValueError(f"frozen trace hash drift: {trace_sha}")
    with placement_summary.open(newline="", encoding="utf-8") as stream:
        placement_rows = list(csv.DictReader(stream))
    if len(placement_rows) != 10:
        raise ValueError(f"expected 10 target-split placements, got {len(placement_rows)}")
    by_key = {
        (row["orientation"], float(row["target_slow_fraction"])): row
        for row in placement_rows
    }
    expected_keys = {
        (orientation, target)
        for orientation in ORIENTATIONS
        for target in (0.50, 0.475, 0.45, 0.425, 0.40)
    }
    if set(by_key) != expected_keys:
        raise ValueError("target-split placement set is incomplete")

    definitions = []
    for orientation in ("MH", "HM"):
        for target in (0.50, 0.475, 0.45, 0.425, 0.40):
            source = by_key[(orientation, target)]
            label = f"{target * 100:g}".replace(".", "p")
            orientation_spec = ORIENTATIONS[orientation]
            definitions.append({
                "config_id": f"{orientation}{label}",
                "orientation": orientation,
                "target_slow_fraction": target,
                "achieved_slow_fraction": float(source["achieved_slow_fraction"]),
                "slow_rank": orientation_spec["slow_rank"],
                "placement": _placement_record(root, source),
                "attention_clocks_mhz": [2520, 2520],
                "ffn_clocks_mhz": orientation_spec["ffn_clocks_mhz"],
                "target_clocks_mhz": [2520, 2520, *orientation_spec["ffn_clocks_mhz"]],
                "repetitions": list(REPETITIONS),
            })
    permutation = list(range(len(definitions)))
    random.Random(RANDOMIZATION_SEED).shuffle(permutation)
    configurations = []
    for order, definition_index in enumerate(permutation):
        configurations.append({"order": order, **definitions[definition_index]})

    symmetric_reference = root / "results/processed/milestone_m_symmetric_aggregate.csv"
    return {
        "schema_version": 1,
        "study": "milestone_m_afd_heterogeneous_target_split",
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
        "ffn_frequency_orientations_mhz": ORIENTATIONS,
        "target_slow_fractions": [0.50, 0.475, 0.45, 0.425, 0.40],
        "placement_construction": "direct frozen-profile target split; no frequency/H capacity proxy",
        "placement_summary": str(placement_summary.relative_to(root)),
        "placement_summary_sha256": sha256_file(placement_summary),
        "symmetric_reference": str(symmetric_reference.relative_to(root)),
        "symmetric_reference_sha256": sha256_file(symmetric_reference),
        "randomization_seed": RANDOMIZATION_SEED,
        "configuration_order": [row["config_id"] for row in configurations],
        "configurations": configurations,
        "repetitions": list(REPETITIONS),
        "conditioning_traces_per_configuration": 1,
        "server_lifecycle": "restart_per_configuration",
        "expected_runs": len(configurations) * len(REPETITIONS),
        "routing_profiling": False,
        "custom_kernel_timing": False,
        "learned_capacity_model": False,
        "dynamic_dvfs": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--placement-summary", type=Path, required=True)
    args = parser.parse_args()
    root = Path.cwd().resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    trace = args.trace.resolve(strict=True)
    summary = args.placement_summary.resolve(strict=True)
    plan = target_split_plan(root, trace, summary)
    shutil.copyfile(trace, output / "trace.jsonl")
    (output / "manifest.json").write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(output), **plan}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
