#!/usr/bin/env python3
"""Build Milestone D tables and diagnostic figures from one controller directory."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any

from src.search import dvfs_sensitivity


def _runs(controller: Path) -> list[dict[str, Any]]:
    rows = []
    with (controller / "runs.tsv").open(newline="", encoding="utf-8") as stream:
        for entry in csv.DictReader(stream, delimiter="\t"):
            metrics = json.loads((Path(entry["run_dir"]) / "metrics.json").read_text())
            rows.append({**entry, **metrics})
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("controller", type=Path)
    parser.add_argument("--processed-dir", type=Path, default=Path("results/processed"))
    parser.add_argument("--figure", type=Path,
                        default=Path("results/figures/milestone_d_dvfs_sensitivity.pdf"))
    args = parser.parse_args()
    runs = _runs(args.controller)
    placement_stats = json.loads(
        (args.controller / "placements" / "placement_loads.json").read_text()
    )
    loads = {row["placement"]: row["per_gpu_load"] for row in placement_stats}
    indexed = {row["label"]: row for row in runs}
    placements = ("upstream_default_eplb", "diagnostic_load_concentrated")
    sensitivity = []
    for placement in placements:
        reference = indexed[f"{placement}-uniform-high"]
        for gpu in range(4):
            changed = indexed[f"{placement}-gpu{gpu}-medium"]
            row = dvfs_sensitivity(reference, changed, gpu)
            row.update({
                "placement": placement,
                "predicted_expert_load": loads[placement][gpu],
                "high_frequency_mhz": 2520,
                "downclock_frequency_mhz": 1635,
                "reference_run": reference["run_dir"],
                "perturbation_run": changed["run_dir"],
            })
            sensitivity.append(row)

    uniform = [row for row in runs if "-uniform-" in row["label"]
               and "-rep" not in row["label"]]
    uniform_fields = [
        "label", "placement", "clocks", "completed_requests", "failed_requests",
        "throughput_requests_s", "ttft_p99_ms", "tpot_p99_ms", "e2e_p99_ms",
        "total_gpu_energy_j", "joules_per_request", "joules_per_output_token",
        "client_send_delay_p99_ms", "run_dir",
    ]
    sensitivity_fields = [
        "placement", "gpu", "predicted_expert_load", "high_frequency_mhz",
        "downclock_frequency_mhz", "delta_ttft_p99_ms", "delta_tpot_p99_ms",
        "delta_e2e_p99_ms", "delta_total_energy_j", "delta_gpu_energy_j",
        "delta_throughput_requests_s", "reference_run", "perturbation_run",
    ]
    args.processed_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.processed_dir / "milestone_d_uniform.csv", uniform, uniform_fields)
    _write_csv(args.processed_dir / "milestone_d_sensitivity.csv", sensitivity,
               sensitivity_fields)

    repeatability = []
    for placement in placements:
        for kind in ("uniform-high", "gpu0-medium"):
            prefix = f"{placement}-{kind}"
            selected = [row for row in runs
                        if row["label"] == prefix or row["label"].startswith(prefix + "-rep")]
            repeatability.append({
                "placement": placement,
                "configuration": kind,
                "repetitions": len(selected),
                "tpot_p99_mean_ms": statistics.mean(row["tpot_p99_ms"] for row in selected),
                "tpot_p99_stdev_ms": statistics.stdev(row["tpot_p99_ms"] for row in selected),
                "energy_mean_j": statistics.mean(row["total_gpu_energy_j"] for row in selected),
                "energy_stdev_j": statistics.stdev(row["total_gpu_energy_j"] for row in selected),
            })
    (args.processed_dir / "milestone_d_summary.json").write_text(json.dumps({
        "controller": str(args.controller),
        "placement_statistics": placement_stats,
        "sensitivity": sensitivity,
        "repeatability": repeatability,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    import matplotlib.pyplot as plt
    markers = {"upstream_default_eplb": "o", "diagnostic_load_concentrated": "s"}
    labels = {"upstream_default_eplb": "P1 load-balanced",
              "diagnostic_load_concentrated": "P2 load-concentrated"}
    annotation_offsets = {
        ("delta_tpot_p99_ms", "upstream_default_eplb", 1): (4, -13),
        ("delta_tpot_p99_ms", "upstream_default_eplb", 2): (4, 6),
        ("delta_e2e_p99_ms", "upstream_default_eplb", 1): (4, -13),
        ("delta_e2e_p99_ms", "upstream_default_eplb", 2): (4, 5),
        ("delta_e2e_p99_ms", "diagnostic_load_concentrated", 1): (4, 4),
        ("delta_e2e_p99_ms", "diagnostic_load_concentrated", 2): (4, 14),
    }
    figure, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    for placement in placements:
        selected = [row for row in sensitivity if row["placement"] == placement]
        x = [row["predicted_expert_load"] for row in selected]
        for axis, metric, title in (
            (axes[0], "delta_tpot_p99_ms", "p99 TPOT penalty"),
            (axes[1], "delta_e2e_p99_ms", "p99 E2E penalty"),
        ):
            y = [row[metric] for row in selected]
            axis.scatter(x, y, marker=markers[placement], label=labels[placement])
            for row, x_value, y_value in zip(selected, x, y):
                offset = annotation_offsets.get((metric, placement, row["gpu"]), (4, 3))
                axis.annotate(f"GPU{row['gpu']}", (x_value, y_value),
                              xytext=offset, textcoords="offset points", fontsize=8)
            axis.set_xlabel("Predicted routed-token load")
            axis.set_ylabel(f"{title} (ms)")
            axis.grid(True, alpha=0.25)
    for axis in axes:
        axis.margins(x=0.04, y=0.15)
    axes[0].legend(frameon=False)
    figure.tight_layout()
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.figure)
    print(json.dumps({"uniform_rows": len(uniform),
                      "sensitivity_rows": len(sensitivity),
                      "figure": str(args.figure)}))


if __name__ == "__main__":
    main()
