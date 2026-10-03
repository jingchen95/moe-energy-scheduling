#!/usr/bin/env python3
"""Prepare the one-allocation AFD FFN cost-characterization study."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import subprocess
from pathlib import Path
from typing import Any

from src.afd_static_placement import load_placement, sha256_file
from tools.prepare_milestone_m_measurement import (
    TRACE_SHA256,
    canonical_mapping_sha256,
)


RANDOMIZATION_SEED = 20261007
FREQUENCIES = (1005, 1635, 2070, 2220, 2520)


def _git_commit(root: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()


def _record(root: Path, path: Path) -> dict[str, Any]:
    absolute = path.resolve(strict=True)
    return {
        "path": str(absolute.relative_to(root)),
        "sha256": sha256_file(absolute),
        "bytes": absolute.stat().st_size,
    }


def _placement(root: Path, path: Path) -> dict[str, Any]:
    mapping, file_sha, name = load_placement(str(path.resolve(strict=True)))
    return {
        "name": name,
        "path": str(path.resolve().relative_to(root)),
        "sha256": file_sha,
        "canonical_mapping_sha256": canonical_mapping_sha256(mapping),
    }


def build_plan(root: Path, trace: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    if sha256_file(trace) != TRACE_SHA256:
        raise ValueError("frozen 300-request trace hash drift")
    balanced = _placement(root, root / "data/placements/milestone_m/P_balanced.json")
    split0 = _placement(
        root,
        root / "data/placements/milestone_m_target_splits/P_split_40_slow_rank0.json",
    )
    split1 = _placement(
        root,
        root / "data/placements/milestone_m_target_splits/P_split_40_slow_rank1.json",
    )
    reused_files = [
        root / "results/processed/milestone_m_symmetric_runs.csv",
        root / "results/processed/milestone_m_symmetric_aggregate.csv",
        root / "results/processed/milestone_m_target_split_runs.csv",
        root / "results/processed/milestone_m_target_split_aggregate.csv",
        root / "results/processed/milestone_m_target_split_placements.csv",
    ]
    raw_directories = [
        root / "results/raw/milestone-m-symmetric-restart-20261002T152801Z",
        root / "results/raw/milestone-m-target-splits-20261003T072138Z",
    ]
    missing_raw = [str(path) for path in raw_directories if not path.is_dir()]
    if missing_raw:
        raise FileNotFoundError(f"missing reused raw data: {missing_raw}")
    reuse = {
        "schema_version": 1,
        "primary_energy_power_data": [_record(root, path) for path in reused_files],
        "primary_energy_power_raw_directories": [
            str(path.resolve().relative_to(root)) for path in raw_directories
        ],
        "methodology_only_not_training_data": [
            _record(root, root / "src/moe_timing.py"),
            _record(root, root / "patches/vllm-0.15.1-moe-timing.patch"),
            _record(root, root / "docs/milestone_g_load_frequency_model.md"),
        ],
        "missing_before_this_study": [
            "per-invocation logical routing histogram",
            "AFD vLLM 0.26 rank-local FFN CUDA-event latency",
            "held-out lifecycle timing repeat at 1635 and 2070 MHz",
        ],
        "prohibited_reuse": (
            "Milestone G EP=4 colocated timing is not training data for AFD 2A2F"
        ),
    }

    definitions: list[dict[str, Any]] = []
    for frequency in FREQUENCIES:
        definitions.append({
            "config_id": f"balanced-f{frequency}-primary",
            "stage": "symmetric_primary",
            "frequency_mhz": frequency,
            "lifecycle": "primary",
            "placement": balanced,
        })
    for frequency in (1635, 2070):
        definitions.append({
            "config_id": f"balanced-f{frequency}-repeat",
            "stage": "symmetric_repeat",
            "frequency_mhz": frequency,
            "lifecycle": "heldout_repeat",
            "placement": balanced,
        })
    for frequency in (1635, 2070):
        for placement in (split0, split1):
            definitions.append({
                "config_id": f"{placement['name']}-f{frequency}",
                "stage": "distribution_generalization",
                "frequency_mhz": frequency,
                "lifecycle": "heldout_skew",
                "placement": placement,
            })
    order = list(range(len(definitions)))
    random.Random(RANDOMIZATION_SEED).shuffle(order)
    configurations = [
        {"order": index, **definitions[source]} for index, source in enumerate(order)
    ]
    plan = {
        "schema_version": 1,
        "study": "milestone_n_afd_ffn_cost_model",
        "git_commit": _git_commit(root),
        "model": "Qwen/Qwen3-30B-A3B",
        "vllm_version": "0.26.0",
        "plugin_commit": "4c2541a53ec8ad507508073eac81ac9feecc6c9a",
        "topology": "2A2F",
        "execution_mode": "FULL_DECODE_ONLY",
        "trace_sha256": TRACE_SHA256,
        "requests": 300,
        "request_rate": 8.0,
        "max_output_tokens": 128,
        "attention_clocks_mhz": [2520, 2520],
        "routing_stage": {
            "placement": balanced,
            "ffn_clocks_mhz": [2520, 2520],
            "separate_from_timing": True,
        },
        "timing_boundary": (
            "AFDQwen3MoeDecoderLayer.compute_ffn_output entry immediately before "
            "native Qwen3MoeSparseMoeBlock through returned output"
        ),
        "timing_boundary_includes": [
            "internal routing", "EP dispatch/all-to-all", "routed experts",
            "shared expert", "combine", "CUDA dependencies returning to caller stream",
        ],
        "timing_boundary_excludes": ["AFD input receive", "AFD output send"],
        "randomization_seed": RANDOMIZATION_SEED,
        "configuration_order": [row["config_id"] for row in configurations],
        "configurations": configurations,
        "expected_timing_runs": len(configurations),
        "energy_measurement_reused_not_rerun": True,
        "dynamic_dvfs": False,
        "joint_optimizer": False,
    }
    return plan, reuse


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    args = parser.parse_args()
    root = Path.cwd().resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    trace = args.trace.resolve(strict=True)
    plan, reuse = build_plan(root, trace)
    shutil.copyfile(trace, output / "trace.jsonl")
    shutil.copyfile(
        root / "configs/milestone_n_afd_ffn_cost_model.yaml",
        output / "resolved-config.yaml",
    )
    for name, value in (("manifest.json", plan), ("reuse-manifest.json", reuse)):
        (output / name).write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps({"output_dir": str(output), **plan}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
