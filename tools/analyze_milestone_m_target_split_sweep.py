#!/usr/bin/env python3
"""Compare heterogeneous target-split points with the symmetric frontier."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from tools.analyze_milestone_m_measurements import (
    AGGREGATE_METRICS,
    _run_row,
    _write_csv,
)


def _load_runs(study: Path, expected: int) -> list[dict[str, str]]:
    with (study / "runs.tsv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    complete = [row for row in rows if row["status"] == "complete"]
    if len(rows) != expected or len(complete) != expected:
        raise ValueError(f"expected {expected} complete runs, got {len(complete)}/{len(rows)}")
    return complete


def _aggregate(per_run: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    selected = [row for row in per_run if row["config_id"] == config["config_id"]]
    if sorted(row["repetition"] for row in selected) != list(config["repetitions"]):
        raise ValueError(f"incomplete repetitions for {config['config_id']}")
    result: dict[str, Any] = {
        "point": config["config_id"],
        "point_type": "heterogeneous_target_split",
        "configuration_order": config["order"],
        "orientation": config["orientation"],
        "ffn_clocks_mhz": "/".join(str(v) for v in config["ffn_clocks_mhz"]),
        "target_slow_fraction": config["target_slow_fraction"],
        "achieved_slow_fraction": config["achieved_slow_fraction"],
        "placement": config["placement"]["name"],
        "placement_sha256": config["placement"]["sha256"],
        "runs": len(selected),
        "completed_requests_total": sum(row["completed_requests"] for row in selected),
        "failed_requests_total": sum(row["failed_requests"] for row in selected),
    }
    for metric in AGGREGATE_METRICS:
        values = [float(row[metric]) for row in selected]
        result[f"{metric}_mean"] = mean(values)
        result[f"{metric}_sample_std"] = stdev(values)
        result[f"{metric}_values"] = ";".join(f"{value:.9f}" for value in values)
    return result


def _symmetric_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    result = []
    for row in rows:
        frequency = int(row["frequency_mhz"])
        result.append({
            "point": f"symmetric_{frequency}",
            "point_type": "symmetric",
            "orientation": "symmetric",
            "ffn_clocks_mhz": f"{frequency}/{frequency}",
            "target_slow_fraction": "",
            "achieved_slow_fraction": "",
            **{
                f"{metric}_{suffix}": float(row[f"{metric}_{suffix}"])
                for metric in AGGREGATE_METRICS
                for suffix in ("mean", "sample_std")
            },
        })
    return result


def pareto_points(rows: list[dict[str, Any]], latency_metric: str) -> set[str]:
    frontier = set()
    for candidate in rows:
        energy = float(candidate["total_gpu_energy_j_mean"])
        latency = float(candidate[f"{latency_metric}_mean"])
        dominated = any(
            float(other["total_gpu_energy_j_mean"]) <= energy
            and float(other[f"{latency_metric}_mean"]) <= latency
            and (
                float(other["total_gpu_energy_j_mean"]) < energy
                or float(other[f"{latency_metric}_mean"]) < latency
            )
            for other in rows if other is not candidate
        )
        if not dominated:
            frontier.add(str(candidate["point"]))
    return frontier


def comparison_rows(
    heterogeneous: list[dict[str, Any]], symmetric: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    references = {row["point"]: row for row in symmetric}
    high = references["symmetric_2070"]
    low = references["symmetric_1635"]
    rows = []
    for point in heterogeneous:
        row: dict[str, Any] = {
            "point": point["point"],
            "orientation": point["orientation"],
            "target_slow_fraction": point["target_slow_fraction"],
            "achieved_slow_fraction": point["achieved_slow_fraction"],
        }
        for reference_name, reference in (("symmetric_2070", high), ("symmetric_1635", low)):
            for metric in (
                "ttft_p99_ms", "tpot_p50_ms", "tpot_p95_ms", "tpot_p99_ms",
                "throughput_requests_s", "total_gpu_energy_j", "ffn_pool_energy_j",
            ):
                current = float(point[f"{metric}_mean"])
                baseline = float(reference[f"{metric}_mean"])
                row[f"{metric}_relative_to_{reference_name}"] = (
                    current - baseline
                ) / baseline
        row["lower_energy_than_symmetric_2070"] = (
            float(point["total_gpu_energy_j_mean"])
            < float(high["total_gpu_energy_j_mean"])
        )
        tpot = float(point["tpot_p99_ms_mean"])
        row["tpot_closer_to_symmetric_2070_than_1635"] = (
            abs(tpot - float(high["tpot_p99_ms_mean"]))
            < abs(tpot - float(low["tpot_p99_ms_mean"]))
        )
        rows.append(row)
    return rows


def _figure(path: Path, rows: list[dict[str, Any]], frontier: set[str]) -> None:
    width, height = 920, 620
    left, top, plot_width, plot_height = 90, 85, 760, 440
    xs = [float(row["tpot_p99_ms_mean"]) for row in rows]
    ys = [float(row["total_gpu_energy_j_mean"]) / 1000.0 for row in rows]
    x_pad = max((max(xs) - min(xs)) * 0.12, 0.5)
    y_pad = max((max(ys) - min(ys)) * 0.12, 0.3)
    x_min, x_max = min(xs) - x_pad, max(xs) + x_pad
    y_min, y_max = min(ys) - y_pad, max(ys) + y_pad
    pieces = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="50" y="37" font-family="sans-serif" font-size="22" font-weight="bold">Heterogeneous target splits vs symmetric FFN DVFS</text>',
        '<text x="50" y="62" font-family="sans-serif" font-size="13">Three-run means; outlined points are on the measured energy--TPOT-p99 frontier.</text>',
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#333"/>',
        f'<text x="{left + plot_width / 2}" y="580" text-anchor="middle" font-family="sans-serif" font-size="14">TPOT p99 (ms)</text>',
        f'<text x="30" y="{top + plot_height / 2}" transform="rotate(-90 30 {top + plot_height / 2})" text-anchor="middle" font-family="sans-serif" font-size="14">Total GPU energy (kJ)</text>',
    ]
    colors = {"symmetric": "#4c78a8", "MH": "#e45756", "HM": "#54a24b"}
    for row in rows:
        x_value = float(row["tpot_p99_ms_mean"])
        y_value = float(row["total_gpu_energy_j_mean"]) / 1000.0
        x = left + (x_value - x_min) / (x_max - x_min) * plot_width
        y = top + plot_height - (y_value - y_min) / (y_max - y_min) * plot_height
        orientation = str(row["orientation"])
        color = colors[orientation]
        stroke = "#111" if row["point"] in frontier else "none"
        stroke_width = 2 if row["point"] in frontier else 0
        pieces.append(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="6" fill="{color}" stroke="{stroke}" stroke-width="{stroke_width}"/>'
        )
        pieces.append(
            f'<text x="{x + 8:.2f}" y="{y - 7:.2f}" font-family="sans-serif" font-size="9">{row["point"]}</text>'
        )
    pieces.extend([
        '<rect x="610" y="32" width="10" height="10" fill="#4c78a8"/><text x="625" y="41" font-family="sans-serif" font-size="11">symmetric</text>',
        '<rect x="700" y="32" width="10" height="10" fill="#e45756"/><text x="715" y="41" font-family="sans-serif" font-size="11">MH</text>',
        '<rect x="755" y="32" width="10" height="10" fill="#54a24b"/><text x="770" y="41" font-family="sans-serif" font-size="11">HM</text>',
        "</svg>",
    ])
    path.write_text("\n".join(pieces) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument(
        "--symmetric-aggregate", type=Path,
        default=Path("results/processed/milestone_m_symmetric_aggregate.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/processed"))
    parser.add_argument(
        "--figure", type=Path,
        default=Path("results/figures/milestone_m_target_split_frontier.svg"),
    )
    args = parser.parse_args()
    study = args.study_dir.resolve(strict=True)
    plan = json.loads((study / "manifest.json").read_text())
    if json.loads((study / "result.json").read_text()).get("status") != "complete":
        raise ValueError("target-split study is not complete")
    plan_by_id = {row["config_id"]: row for row in plan["configurations"]}
    if len(plan_by_id) != 10:
        raise ValueError("expected ten target-split configurations")
    raw_runs = _load_runs(study, int(plan["expected_runs"]))
    per_run = [_run_row(row, plan_by_id) for row in raw_runs]
    for row in per_run:
        config = plan_by_id[row["config_id"]]
        row.update({
            "orientation": config["orientation"],
            "ffn_clocks_mhz": "/".join(str(v) for v in config["ffn_clocks_mhz"]),
            "target_slow_fraction": config["target_slow_fraction"],
            "achieved_slow_fraction": config["achieved_slow_fraction"],
        })
    per_run.sort(key=lambda row: (row["orientation"], -row["target_slow_fraction"], row["repetition"]))
    aggregates = [_aggregate(per_run, config) for config in plan_by_id.values()]
    aggregates.sort(key=lambda row: (row["orientation"], -row["target_slow_fraction"]))
    symmetric = _symmetric_rows(args.symmetric_aggregate.resolve(strict=True))
    combined = [*symmetric, *aggregates]
    tpot_frontier = pareto_points(combined, "tpot_p99_ms")
    ttft_frontier = pareto_points(combined, "ttft_p99_ms")
    frontier_rows = [{
        "point": row["point"],
        "point_type": row["point_type"],
        "orientation": row["orientation"],
        "ffn_clocks_mhz": row["ffn_clocks_mhz"],
        "target_slow_fraction": row["target_slow_fraction"],
        "achieved_slow_fraction": row["achieved_slow_fraction"],
        "ttft_p99_ms_mean": row["ttft_p99_ms_mean"],
        "tpot_p99_ms_mean": row["tpot_p99_ms_mean"],
        "total_gpu_energy_j_mean": row["total_gpu_energy_j_mean"],
        "ffn_pool_energy_j_mean": row["ffn_pool_energy_j_mean"],
        "throughput_requests_s_mean": row["throughput_requests_s_mean"],
        "energy_tpot_p99_pareto": row["point"] in tpot_frontier,
        "energy_ttft_p99_pareto": row["point"] in ttft_frontier,
    } for row in combined]
    comparisons = comparison_rows(aggregates, symmetric)
    primary_criterion_points = sorted(
        row["point"] for row in comparisons
        if row["lower_energy_than_symmetric_2070"]
        and row["tpot_closer_to_symmetric_2070_than_1635"]
    )
    tpot_by_split = {
        orientation: [
            {
                "target_slow_fraction": row["target_slow_fraction"],
                "tpot_p99_ms_mean": row["tpot_p99_ms_mean"],
                "total_gpu_energy_j_mean": row["total_gpu_energy_j_mean"],
            }
            for row in aggregates if row["orientation"] == orientation
        ]
        for orientation in ("MH", "HM")
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "milestone_m_target_split_runs.csv", per_run)
    _write_csv(args.output_dir / "milestone_m_target_split_aggregate.csv", aggregates)
    _write_csv(args.output_dir / "milestone_m_target_split_comparisons.csv", comparisons)
    _write_csv(args.output_dir / "milestone_m_target_split_frontier.csv", frontier_rows)
    summary = {
        "schema_version": 1,
        "study_dir": str(study),
        "validated_runs": len(per_run),
        "configuration_order": plan["configuration_order"],
        "aggregates": {row["point"]: row for row in aggregates},
        "comparisons": {row["point"]: row for row in comparisons},
        "tpot_frontier": sorted(tpot_frontier),
        "ttft_frontier": sorted(ttft_frontier),
        "heterogeneous_tpot_frontier_points": sorted(
            point for point in tpot_frontier if not point.startswith("symmetric_")
        ),
        "primary_criterion_points": primary_criterion_points,
        "tpot_and_energy_by_target_split": tpot_by_split,
        "interpretation_boundary": "measured motivation experiment; no learned capacity model or dynamic DVFS",
    }
    (args.output_dir / "milestone_m_target_split_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    _figure(args.figure, combined, tpot_frontier)
    print(json.dumps({
        "validated_runs": len(per_run),
        "heterogeneous_tpot_frontier_points": summary["heterogeneous_tpot_frontier_points"],
        "primary_criterion_points": primary_criterion_points,
        "tpot_frontier": summary["tpot_frontier"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
