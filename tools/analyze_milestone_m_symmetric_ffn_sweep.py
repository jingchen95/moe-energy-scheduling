#!/usr/bin/env python3
"""Analyze the fixed-placement symmetric FFN frequency sweep."""

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


EXPECTED_FREQUENCIES_MHZ = (2520, 2220, 2070, 1635, 1005)
RELATIVE_METRICS = (
    "ttft_p50_ms", "ttft_p95_ms", "ttft_p99_ms",
    "tpot_p50_ms", "tpot_p95_ms", "tpot_p99_ms",
    "throughput_requests_s", "total_gpu_energy_j", "ffn_pool_energy_j",
)


def _load_runs(study: Path, expected: int) -> list[dict[str, str]]:
    with (study / "runs.tsv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    complete = [row for row in rows if row["status"] == "complete"]
    if len(rows) != expected or len(complete) != expected:
        raise ValueError(
            f"expected {expected} complete runs, got {len(complete)}/{len(rows)}"
        )
    return complete


def _aggregate(
    per_run: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    selected = [row for row in per_run if row["config_id"] == config["config_id"]]
    expected_repetitions = list(config["repetitions"])
    if sorted(row["repetition"] for row in selected) != expected_repetitions:
        raise ValueError(f"incomplete repetitions for {config['config_id']}")
    result: dict[str, Any] = {
        "config_id": config["config_id"],
        "configuration_order": config["order"],
        "frequency_mhz": config["frequency_mhz"],
        "placement": config["placement"]["name"],
        "target_clocks_mhz": "/".join(
            str(value) for value in config["target_clocks_mhz"]
        ),
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


def relative_rows(aggregates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_frequency = {int(row["frequency_mhz"]): row for row in aggregates}
    baseline = by_frequency[2520]
    rows = []
    for frequency in sorted(by_frequency, reverse=True):
        current = by_frequency[frequency]
        row: dict[str, Any] = {
            "frequency_mhz": frequency,
            "baseline_frequency_mhz": 2520,
        }
        for metric in RELATIVE_METRICS:
            current_mean = float(current[f"{metric}_mean"])
            baseline_mean = float(baseline[f"{metric}_mean"])
            row[f"{metric}_mean"] = current_mean
            row[f"{metric}_sample_std"] = current[f"{metric}_sample_std"]
            row[f"{metric}_relative_to_hh"] = (
                (current_mean - baseline_mean) / baseline_mean
            )
        rows.append(row)
    return rows


def pareto_frequencies(
    aggregates: list[dict[str, Any]], latency_metric: str
) -> set[int]:
    """Return non-dominated frequencies for energy and the selected latency."""
    frontier: set[int] = set()
    for candidate in aggregates:
        candidate_energy = float(candidate["total_gpu_energy_j_mean"])
        candidate_latency = float(candidate[f"{latency_metric}_mean"])
        dominated = False
        for other in aggregates:
            if other is candidate:
                continue
            other_energy = float(other["total_gpu_energy_j_mean"])
            other_latency = float(other[f"{latency_metric}_mean"])
            if (
                other_energy <= candidate_energy
                and other_latency <= candidate_latency
                and (
                    other_energy < candidate_energy
                    or other_latency < candidate_latency
                )
            ):
                dominated = True
                break
        if not dominated:
            frontier.add(int(candidate["frequency_mhz"]))
    return frontier


def frontier_rows(aggregates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tpot_frontier = pareto_frequencies(aggregates, "tpot_p99_ms")
    ttft_frontier = pareto_frequencies(aggregates, "ttft_p99_ms")
    rows = []
    for aggregate in sorted(aggregates, key=lambda row: row["frequency_mhz"], reverse=True):
        frequency = int(aggregate["frequency_mhz"])
        rows.append({
            "frequency_mhz": frequency,
            "total_gpu_energy_j_mean": aggregate["total_gpu_energy_j_mean"],
            "total_gpu_energy_j_sample_std": aggregate["total_gpu_energy_j_sample_std"],
            "ffn_pool_energy_j_mean": aggregate["ffn_pool_energy_j_mean"],
            "ttft_p99_ms_mean": aggregate["ttft_p99_ms_mean"],
            "ttft_p99_ms_sample_std": aggregate["ttft_p99_ms_sample_std"],
            "tpot_p99_ms_mean": aggregate["tpot_p99_ms_mean"],
            "tpot_p99_ms_sample_std": aggregate["tpot_p99_ms_sample_std"],
            "throughput_requests_s_mean": aggregate["throughput_requests_s_mean"],
            "energy_tpot_p99_pareto": frequency in tpot_frontier,
            "energy_ttft_p99_pareto": frequency in ttft_frontier,
        })
    return rows


def _figure(path: Path, aggregates: list[dict[str, Any]]) -> None:
    ordered = sorted(aggregates, key=lambda row: row["frequency_mhz"])
    width, height = 1160, 620
    panel_width = 500
    lefts = (75, 635)
    pieces = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="55" y="38" font-family="sans-serif" font-size="23" font-weight="bold">Symmetric FFN frequency frontier at 8 requests/s</text>',
        '<text x="55" y="64" font-family="sans-serif" font-size="14">P_balanced; Attention GPUs fixed at 2520 MHz; points are three-run means.</text>',
    ]
    panels = (
        ("Total energy vs TPOT p99", "tpot_p99_ms_mean", "TPOT p99 (ms)"),
        ("Total energy vs TTFT p99", "ttft_p99_ms_mean", "TTFT p99 (ms)"),
    )
    energies = [float(row["total_gpu_energy_j_mean"]) / 1000.0 for row in ordered]
    energy_min, energy_max = min(energies), max(energies)
    energy_pad = max((energy_max - energy_min) * 0.15, 0.5)
    for panel_index, (title, latency_key, latency_label) in enumerate(panels):
        left = lefts[panel_index]
        top, plot_height = 115, 400
        latencies = [float(row[latency_key]) for row in ordered]
        latency_min, latency_max = min(latencies), max(latencies)
        latency_pad = max((latency_max - latency_min) * 0.15, 1.0)
        x_min, x_max = latency_min - latency_pad, latency_max + latency_pad
        y_min, y_max = energy_min - energy_pad, energy_max + energy_pad
        pieces.extend([
            f'<text x="{left + panel_width / 2}" y="95" text-anchor="middle" font-family="sans-serif" font-size="16" font-weight="bold">{title}</text>',
            f'<line x1="{left}" y1="{top + plot_height}" x2="{left + panel_width}" y2="{top + plot_height}" stroke="#333"/>',
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#333"/>',
            f'<text x="{left + panel_width / 2}" y="570" text-anchor="middle" font-family="sans-serif" font-size="13">{latency_label}</text>',
            f'<text x="{left - 48}" y="{top + plot_height / 2}" transform="rotate(-90 {left - 48} {top + plot_height / 2})" text-anchor="middle" font-family="sans-serif" font-size="13">Total GPU energy (kJ)</text>',
        ])
        for row in ordered:
            latency = float(row[latency_key])
            energy = float(row["total_gpu_energy_j_mean"]) / 1000.0
            x = left + (latency - x_min) / (x_max - x_min) * panel_width
            y = top + plot_height - (energy - y_min) / (y_max - y_min) * plot_height
            frequency = int(row["frequency_mhz"])
            pieces.extend([
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="7" fill="#4c78a8"/>',
                f'<text x="{x + 9:.2f}" y="{y - 9:.2f}" font-family="sans-serif" font-size="11">{frequency} MHz</text>',
            ])
        pieces.extend([
            f'<text x="{left}" y="540" font-family="sans-serif" font-size="10">{x_min:.1f}</text>',
            f'<text x="{left + panel_width}" y="540" text-anchor="end" font-family="sans-serif" font-size="10">{x_max:.1f}</text>',
            f'<text x="{left - 8}" y="{top + plot_height}" text-anchor="end" font-family="sans-serif" font-size="10">{y_min:.1f}</text>',
            f'<text x="{left - 8}" y="{top + 8}" text-anchor="end" font-family="sans-serif" font-size="10">{y_max:.1f}</text>',
        ])
    pieces.append("</svg>")
    path.write_text("\n".join(pieces) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/processed"))
    parser.add_argument(
        "--figure", type=Path,
        default=Path("results/figures/milestone_m_symmetric_ffn_frontier.svg"),
    )
    args = parser.parse_args()
    study = args.study_dir.resolve(strict=True)
    plan = json.loads((study / "manifest.json").read_text())
    result = json.loads((study / "result.json").read_text())
    if result.get("status") != "complete":
        raise ValueError("symmetric sweep result is not complete")
    if tuple(plan["frequencies_mhz"]) != EXPECTED_FREQUENCIES_MHZ:
        raise ValueError("unexpected symmetric frequency set")
    plan_by_id = {row["config_id"]: row for row in plan["configurations"]}
    if set(plan_by_id) != {f"S{index}" for index in range(5)}:
        raise ValueError("matrix does not contain exactly S0-S4")
    expected = int(plan["expected_runs"])
    per_run = [_run_row(row, plan_by_id) for row in _load_runs(study, expected)]
    for row in per_run:
        row["frequency_mhz"] = plan_by_id[row["config_id"]]["frequency_mhz"]
    per_run.sort(key=lambda row: (-row["frequency_mhz"], row["repetition"]))
    aggregate = [_aggregate(per_run, config) for config in plan_by_id.values()]
    aggregate.sort(key=lambda row: row["frequency_mhz"], reverse=True)
    relative = relative_rows(aggregate)
    frontier = frontier_rows(aggregate)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "milestone_m_symmetric_runs.csv", per_run)
    _write_csv(args.output_dir / "milestone_m_symmetric_aggregate.csv", aggregate)
    _write_csv(args.output_dir / "milestone_m_symmetric_relative.csv", relative)
    _write_csv(args.output_dir / "milestone_m_symmetric_frontier.csv", frontier)
    summary = {
        "schema_version": 1,
        "study_dir": str(study),
        "validated_runs": len(per_run),
        "placement": plan["placement"],
        "trace_sha256": plan["trace_sha256"],
        "request_rate": plan["request_rate"],
        "configuration_order": plan["configuration_order"],
        "aggregates": {row["config_id"]: row for row in aggregate},
        "relative_to_hh": {str(row["frequency_mhz"]): row for row in relative},
        "frontier": frontier,
        "interpretation_boundary": "descriptive measured frontier; no performance model or placement optimization",
    }
    (args.output_dir / "milestone_m_symmetric_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    _figure(args.figure, aggregate)
    print(json.dumps({
        "validated_runs": len(per_run),
        "frequencies_mhz": list(EXPECTED_FREQUENCIES_MHZ),
        "tpot_frontier_mhz": [
            row["frequency_mhz"] for row in frontier if row["energy_tpot_p99_pareto"]
        ],
        "ttft_frontier_mhz": [
            row["frequency_mhz"] for row in frontier if row["energy_ttft_p99_pareto"]
        ],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
