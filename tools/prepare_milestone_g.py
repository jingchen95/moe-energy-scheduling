#!/usr/bin/env python3
"""Prepare the controlled Milestone G load/frequency matrix offline."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.expert_profile import validate_phase_counts
from src.placement import (
    controlled_load_placements,
    load_placement,
    placement_loads,
    save_placement,
)


FREQUENCIES_MHZ = (2520, 1635, 1005)
PLACEMENT_FREQUENCIES = (2520, 2520, 2520, 2520)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_phase_profile(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        total = np.asarray(archive["expert_counts"])
        if "prefill_expert_counts" not in archive or "decode_expert_counts" not in archive:
            raise SystemExit("Milestone G requires phase-separated expert counts")
        prefill, decode = validate_phase_counts(
            archive["prefill_expert_counts"], archive["decode_expert_counts"], total
        )
    return total.astype(np.int64), prefill, decode


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise SystemExit(f"output directory already exists: {args.output_dir}")
    total, prefill, decode = load_phase_profile(args.phase_counts)
    p1 = load_placement(args.p1_placement)
    placements = controlled_load_placements(total, base=p1)
    args.output_dir.mkdir(parents=True)

    placement_rows: list[dict[str, Any]] = []
    placement_paths: dict[str, str] = {}
    for placement in placements:
        path = args.output_dir / "placements" / f"{placement.name}.json"
        save_placement(path, placement)
        placement_paths[placement.name] = str(path.resolve())
        aggregate = placement_loads(total, placement)
        prefill_load = placement_loads(prefill, placement)
        decode_load = placement_loads(decode, placement)
        placement_rows.append({
            "name": placement.name,
            "alpha": placement.metadata["alpha"],
            "swaps": placement.metadata["swaps"],
            "path": str(path.resolve()),
            "sha256": sha256(path),
            "aggregate_load": aggregate.astype(int).tolist(),
            "prefill_load": prefill_load.astype(int).tolist(),
            "decode_load": decode_load.astype(int).tolist(),
            "aggregate_max_over_mean": float(aggregate.max() / aggregate.mean()),
            "prefill_max_over_mean": float(prefill_load.max() / prefill_load.mean()),
            "decode_max_over_mean": float(decode_load.max() / decode_load.mean()),
        })

    gpu0_decode = [int(row["decode_load"][0]) for row in placement_rows]
    if len(set(gpu0_decode)) != len(gpu0_decode):
        raise SystemExit(f"controlled placements do not separate GPU0 decode loads: {gpu0_decode}")
    if any(right <= left for left, right in zip(gpu0_decode, gpu0_decode[1:])):
        raise SystemExit(f"GPU0 decode load is not strictly ordered: {gpu0_decode}")

    matrix: list[dict[str, Any]] = []
    index = 0
    for row in placement_rows:
        for frequency in FREQUENCIES_MHZ:
            clocks = (frequency, 2520, 2520, 2520)
            matrix.append({
                "matrix_index": index,
                "placement": row["name"],
                "placement_file": row["path"],
                "placement_sha256": row["sha256"],
                "frequency_tuple": list(clocks),
                "gpu0_frequency_mhz": frequency,
                "frequency_label": {2520: "H", 1635: "M", 1005: "L"}[frequency],
                "trace": str(args.trace.resolve()),
                "trace_sha256": sha256(args.trace),
                "phase_counts": str(args.phase_counts.resolve()),
                "phase_counts_sha256": sha256(args.phase_counts),
                "predicted_aggregate_load": row["aggregate_load"],
                "predicted_prefill_load": row["prefill_load"],
                "predicted_decode_load": row["decode_load"],
            })
            index += 1

    result = {
        "schema_version": 1,
        "milestone": "G",
        "model": "Qwen/Qwen3-30B-A3B",
        "vllm_version": "0.15.1",
        "parallel": {"tp": 1, "dp": 4, "ep": 4, "all2all": "allgather_reducescatter"},
        "trace": str(args.trace.resolve()),
        "trace_sha256": sha256(args.trace),
        "phase_counts": str(args.phase_counts.resolve()),
        "phase_counts_sha256": sha256(args.phase_counts),
        "frequency_levels_mhz": list(FREQUENCIES_MHZ),
        "placements": placement_rows,
        "matrix_count": len(matrix),
        "matrix": matrix,
        "timing_definition": "CUDA event around FusedMoE dispatch + expert compute + combine on each rank",
        "output_divergence_is_not_a_model_feature": True,
    }
    write_json(args.output_dir / "load-levels.json", {
        "schema_version": 1,
        "gpu0_decode_load_order": gpu0_decode,
        "placements": placement_rows,
    })
    write_json(args.output_dir / "matrix-plan.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase-counts", type=Path, required=True)
    parser.add_argument("--p1-placement", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps({"output_dir": str(args.output_dir), "matrix_count": prepare(args)["matrix_count"]}, indent=2))


if __name__ == "__main__":
    main()
