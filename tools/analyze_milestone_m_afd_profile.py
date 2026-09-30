#!/usr/bin/env python3
"""Analyze frozen 2A2F AFD routing counts and generate candidate placements."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from src.afd_profile_analysis import (
    HIGH_MHZ,
    MEDIUM_MHZ,
    build_two_rank_candidates,
    layer_rank_loads,
    placement_diagnostics,
    profile_diagnostics,
)
from src.afd_static_placement import (
    combine_runtime_evidence,
    linear_default_mapping,
)
from src.placement import Placement, save_placement


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--ownership-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing empty CSV: {path}")
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _placement_row(
    counts: np.ndarray,
    placement: Placement,
    frequencies: tuple[int, int],
    path: Path,
) -> dict[str, object]:
    diagnostics = placement_diagnostics(counts, placement, frequencies)
    return {
        "placement": placement.name,
        "placement_file": str(path),
        "placement_sha256": sha256(path),
        "ffn0_clock_mhz": frequencies[0],
        "ffn1_clock_mhz": frequencies[1],
        "ffn0_load": diagnostics["aggregate_loads"][0],
        "ffn1_load": diagnostics["aggregate_loads"][1],
        "ffn0_fraction": diagnostics["aggregate_load_fractions"][0],
        "ffn1_fraction": diagnostics["aggregate_load_fractions"][1],
        "raw_max_over_mean": diagnostics["aggregate_raw_max_over_mean"],
        "normalized_max_over_mean": diagnostics[
            "aggregate_normalized_max_over_mean"
        ],
        "mean_layer_normalized_max_over_mean": diagnostics[
            "layer_normalized_max_over_mean_mean"
        ],
        "max_layer_normalized_max_over_mean": diagnostics[
            "layer_normalized_max_over_mean_max"
        ],
    }


def main() -> int:
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    with np.load(args.run_dir / "expert_counts.npz", allow_pickle=False) as archive:
        counts = np.asarray(archive["expert_counts"], dtype=np.int64)
        per_rank_counts = np.asarray(archive["per_rank_expert_counts"], dtype=np.int64)
    if counts.shape != (48, 128) or per_rank_counts.shape != (2, 48, 128):
        raise ValueError("routing-count archive has an unexpected shape")
    if not np.array_equal(counts, per_rank_counts.sum(axis=0)):
        raise ValueError("combined routing counts do not equal rank-local counts")

    expected_default = np.asarray(linear_default_mapping(), dtype=np.int64)
    evidence = sorted(args.ownership_dir.glob("ownership-*.json"))
    ownership = combine_runtime_evidence(evidence, expected_default)
    actual_mapping = np.asarray(ownership["actual_logical_to_rank"], dtype=np.int64)
    (output / "actual-default-ownership.json").write_text(
        json.dumps(ownership, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    diagnostics = profile_diagnostics(counts, actual_mapping)
    rank_loads = layer_rank_loads(counts, actual_mapping)
    layer_rows = []
    expert_layer = diagnostics["logical_expert"]
    default_rank = diagnostics["default_rank"]
    for layer in range(48):
        layer_rows.append({
            "layer": layer,
            "ffn0_load": int(rank_loads[layer, 0]),
            "ffn1_load": int(rank_loads[layer, 1]),
            "ffn0_fraction": default_rank["layer_fractions"][layer][0],
            "ffn1_fraction": default_rank["layer_fractions"][layer][1],
            "absolute_difference": default_rank["layer_absolute_difference"][layer],
            "rank_max_over_mean": default_rank["layer_max_over_mean"][layer],
            "rank_cv": default_rank["layer_cv"][layer],
            "expert_max_over_mean": expert_layer["layer_max_over_mean"][layer],
            "expert_cv": expert_layer["layer_cv"][layer],
        })
    _write_csv(output / "default-layer-rank-loads.csv", layer_rows)
    _write_csv(output / "aggregate-logical-expert-loads.csv", [
        {"logical_expert": expert, "aggregate_load": load}
        for expert, load in enumerate(expert_layer["aggregate_by_expert"])
    ])

    candidates = build_two_rank_candidates(counts)
    placements_dir = output / "placements"
    placements_dir.mkdir()
    frequency_by_name = {
        "P_balanced": (HIGH_MHZ, HIGH_MHZ),
        "P_MH": (MEDIUM_MHZ, HIGH_MHZ),
        "P_HM": (HIGH_MHZ, MEDIUM_MHZ),
    }
    placement_rows = []
    candidate_diagnostics = {}
    for name, placement in candidates.items():
        path = placements_dir / f"{name}.json"
        save_placement(path, placement)
        clocks = frequency_by_name[name]
        placement_rows.append(_placement_row(counts, placement, clocks, path))
        candidate_diagnostics[name] = placement_diagnostics(counts, placement, clocks)
    _write_csv(output / "candidate-placement-summary.csv", placement_rows)

    balanced_mh = placement_diagnostics(
        counts, candidates["P_balanced"], (MEDIUM_MHZ, HIGH_MHZ)
    )
    aligned_mh = candidate_diagnostics["P_MH"]
    pressure_reduction = 1.0 - (
        aligned_mh["aggregate_normalized_max_over_mean"]
        / balanced_mh["aggregate_normalized_max_over_mean"]
    )
    summary = {
        "schema_version": 1,
        "routing_counts": str((args.run_dir / "expert_counts.npz").resolve()),
        "routing_counts_sha256": sha256(args.run_dir / "expert_counts.npz"),
        "counts_shape": list(counts.shape),
        "rank_local_counts_shape": list(per_rank_counts.shape),
        "default_ownership": diagnostics,
        "candidates": candidate_diagnostics,
        "candidate_files": placement_rows,
        "opportunity": {
            "balanced_at_mh": balanced_mh,
            "aligned_mh": aligned_mh,
            "aggregate_normalized_max_over_mean_reduction_fraction": pressure_reduction,
            "selection_boundary": (
                "descriptive routed-token load/capacity proxy only; not execution time"
            ),
        },
    }
    (output / "profile-analysis.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "output_dir": str(output),
        "routing_counts_sha256": summary["routing_counts_sha256"],
        "default_rank_loads": diagnostics["default_rank"]["aggregate_loads"],
        "candidate_rows": placement_rows,
        "pressure_reduction_fraction": pressure_reduction,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
