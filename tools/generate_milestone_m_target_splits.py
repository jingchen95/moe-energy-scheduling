#!/usr/bin/env python3
"""Generate frozen frequency-independent AFD target-split placements."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from src.afd_static_placement import sha256_file
from src.afd_target_split import (
    TARGET_SLOW_FRACTIONS,
    mirror_placement,
    split_diagnostics,
    target_split_placement,
)
from src.placement import load_placement, save_placement
from tools.prepare_milestone_m_measurement import canonical_mapping_sha256


def _label(value: float) -> str:
    return f"{value * 100:g}".replace(".", "p")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expert-counts", type=Path, required=True)
    parser.add_argument("--base-placement", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--details-json", type=Path, required=True)
    args = parser.parse_args()

    with np.load(args.expert_counts.resolve(strict=True), allow_pickle=False) as archive:
        counts = np.asarray(archive["expert_counts"], dtype=np.int64)
    if counts.shape != (48, 128):
        raise ValueError(f"expected counts shape (48, 128), got {counts.shape}")
    base = load_placement(args.base_placement.resolve(strict=True))
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    details = []
    for target in TARGET_SLOW_FRACTIONS:
        label = _label(target)
        primary = target_split_placement(
            counts, base, target, name=f"P_split_{label}_slow_rank0"
        )
        candidates = (
            (primary, "MH", 0),
            (mirror_placement(primary, name=f"P_split_{label}_slow_rank1"), "HM", 1),
        )
        for placement, orientation, slow_rank in candidates:
            path = output / f"{placement.name}.json"
            placement.metadata["routing_counts_sha256"] = sha256_file(
                args.expert_counts.resolve()
            )
            save_placement(path, placement)
            diagnostics = split_diagnostics(
                counts,
                placement,
                slow_rank=slow_rank,
                target_slow_fraction=target,
            )
            relative_path = path.relative_to(Path.cwd().resolve())
            row = {
                "placement": placement.name,
                "orientation": orientation,
                "slow_rank": slow_rank,
                "target_slow_fraction": target,
                "achieved_slow_fraction": diagnostics["achieved_slow_fraction"],
                "layer_slow_fraction_min": diagnostics["layer_slow_fraction_min"],
                "layer_slow_fraction_mean": diagnostics["layer_slow_fraction_mean"],
                "layer_slow_fraction_max": diagnostics["layer_slow_fraction_max"],
                "layer_slow_fraction_max_abs_error": diagnostics[
                    "layer_slow_fraction_max_abs_error"
                ],
                "rank0_load": diagnostics["aggregate_rank_loads"][0],
                "rank1_load": diagnostics["aggregate_rank_loads"][1],
                "placement_file": str(relative_path),
                "placement_sha256": sha256_file(path),
                "canonical_mapping_sha256": canonical_mapping_sha256(
                    tuple(tuple(int(value) for value in layer)
                          for layer in placement.logical_to_rank.tolist())
                ),
            }
            rows.append(row)
            details.append({**row, **diagnostics})

    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.summary_csv.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    args.details_json.write_text(json.dumps({
        "schema_version": 1,
        "routing_counts_sha256": sha256_file(args.expert_counts.resolve()),
        "base_placement": base.name,
        "construction": "direct target routed-token split; no frequency-derived capacity",
        "placements": details,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"placements": rows}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
