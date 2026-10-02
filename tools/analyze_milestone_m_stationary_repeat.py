#!/usr/bin/env python3
"""Analyze the five-repeat, stationary 8 request/s Milestone M matrix."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from tools.analyze_milestone_m_measurements import (
    AGGREGATE_METRICS,
    KEY_METRICS,
    _relative,
    _run_row,
    _write_csv,
    classify,
)


COMPARISONS = (
    ("primary_mh", "C3", "C2"),
    ("primary_hm", "C5", "C4"),
    ("control_hh", "C1", "C0"),
    ("fixed_dvfs_mh", "C2", "C0"),
    ("fixed_dvfs_hm", "C4", "C0"),
    ("aligned_mh_vs_hh", "C3", "C0"),
    ("aligned_hm_vs_hh", "C5", "C0"),
)


def classify_fixed_dvfs(
    comparisons: dict[str, dict[str, Any]],
) -> tuple[str, str]:
    rows = [comparisons["fixed_dvfs_mh"], comparisons["fixed_dvfs_hm"]]
    supported = all(
        row["total_gpu_energy_j_relative_change"] <= -0.05
        and row["tpot_p99_ms_relative_change"] <= 0.03
        and row["throughput_requests_s_relative_change"] >= -0.02
        for row in rows
    )
    if supported:
        return (
            "supported_at_tested_pair",
            "both physical orientations save at least 5% total energy with no more than 3% TPOT or 2% throughput regression",
        )
    mixed = any(row["total_gpu_energy_j_relative_change"] <= -0.02 for row in rows)
    if mixed:
        return "mixed", "at least one orientation saves energy, but the mirrored guardrails are not both met"
    return "not_supported", "neither orientation saves at least 2% total energy"


def _load_runs(study: Path, expected: int) -> list[dict[str, Any]]:
    with (study / "runs.tsv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    complete = [row for row in rows if row["status"] == "complete"]
    if len(rows) != expected or len(complete) != expected:
        raise ValueError(f"expected {expected} complete runs, got {len(complete)}/{len(rows)}")
    return complete


def _aggregate(
    per_run: list[dict[str, Any]], config: dict[str, Any], repetitions: list[int],
) -> dict[str, Any]:
    selected = sorted(
        (row for row in per_run if row["config_id"] == config["config_id"]),
        key=lambda row: int(row["repetition"]),
    )
    if [row["repetition"] for row in selected] != repetitions:
        raise ValueError(f"{config['config_id']} repetition set is incomplete")
    result: dict[str, Any] = {
        "config_id": config["config_id"],
        "configuration_order": config["order"],
        "placement": config["placement"]["name"],
        "target_clocks_mhz": "/".join(str(v) for v in config["target_clocks_mhz"]),
        "runs": len(selected),
        "completed_requests_total": sum(row["completed_requests"] for row in selected),
        "failed_requests_total": sum(row["failed_requests"] for row in selected),
    }
    for metric in AGGREGATE_METRICS:
        values = [float(row[metric]) for row in selected]
        result[f"{metric}_mean"] = mean(values)
        result[f"{metric}_sample_std"] = stdev(values)
        result[f"{metric}_values"] = ";".join(f"{value:.9f}" for value in values)
    for metric in ("ttft_p99_ms", "tpot_p99_ms", "total_gpu_energy_j", "throughput_requests_s"):
        values = [float(row[metric]) for row in selected]
        center_x = (len(values) + 1) / 2.0
        center_y = mean(values)
        numerator = sum(
            (rep - center_x) * (value - center_y)
            for rep, value in zip(repetitions, values)
        )
        denominator = math.sqrt(
            sum((rep - center_x) ** 2 for rep in repetitions)
            * sum((value - center_y) ** 2 for value in values)
        )
        result[f"{metric}_repetition_correlation"] = (
            numerator / denominator if denominator else 0.0
        )
    return result


def _comparison(
    label: str,
    treatment_id: str,
    baseline_id: str,
    aggregates: dict[str, dict[str, Any]],
    per_run: list[dict[str, Any]],
    repetitions: list[int],
) -> dict[str, Any]:
    treatment = aggregates[treatment_id]
    baseline = aggregates[baseline_id]
    row: dict[str, Any] = {
        "comparison": label,
        "treatment": treatment_id,
        "baseline": baseline_id,
    }
    for metric in KEY_METRICS:
        treatment_mean = float(treatment[f"{metric}_mean"])
        baseline_mean = float(baseline[f"{metric}_mean"])
        row[f"{metric}_treatment_mean"] = treatment_mean
        row[f"{metric}_baseline_mean"] = baseline_mean
        row[f"{metric}_relative_change"] = _relative(treatment_mean, baseline_mean)
        row[f"{metric}_treatment_sample_std"] = treatment[f"{metric}_sample_std"]
        row[f"{metric}_baseline_sample_std"] = baseline[f"{metric}_sample_std"]
        treatment_by_rep = {
            int(item["repetition"]): float(item[metric])
            for item in per_run if item["config_id"] == treatment_id
        }
        baseline_by_rep = {
            int(item["repetition"]): float(item[metric])
            for item in per_run if item["config_id"] == baseline_id
        }
        if set(treatment_by_rep) != set(repetitions) or set(baseline_by_rep) != set(repetitions):
            raise ValueError(f"{label} paired repetitions are incomplete")
        paired = [
            _relative(treatment_by_rep[rep], baseline_by_rep[rep])
            for rep in repetitions
        ]
        row[f"{metric}_paired_relative_change_mean"] = mean(paired)
        row[f"{metric}_paired_relative_change_sample_std"] = stdev(paired)
        row[f"{metric}_paired_relative_change_values"] = ";".join(
            f"{value:.9f}" for value in paired
        )
    return row


def _figure(path: Path, per_run: list[dict[str, Any]]) -> None:
    metrics = (
        ("TTFT p99 (ms)", "ttft_p99_ms", 1.0),
        ("TPOT p99 (ms)", "tpot_p99_ms", 1.0),
        ("Total energy (kJ)", "total_gpu_energy_j", 1000.0),
    )
    width, height = 1260, 780
    panel_w, panel_h = 380, 560
    left, top = 70, 100
    colors = ("#4c78a8", "#f58518", "#54a24b", "#e45756", "#72b7b2", "#b279a2")
    pieces = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="55" y="42" font-family="sans-serif" font-size="24" font-weight="bold">Milestone M stationary repeat: individual repetitions</text>',
        '<text x="55" y="70" font-family="sans-serif" font-size="14">Five 8 request/s repetitions per configuration; lines expose within-server order trends.</text>',
    ]
    for panel, (title, metric, divisor) in enumerate(metrics):
        x0 = left + panel * panel_w
        values = [float(row[metric]) / divisor for row in per_run]
        minimum, maximum = min(values), max(values)
        padding = max((maximum - minimum) * 0.1, maximum * 0.02)
        lo, hi = minimum - padding, maximum + padding
        y_base = top + panel_h
        pieces.append(f'<text x="{x0 + 150}" y="{top - 15}" text-anchor="middle" font-family="sans-serif" font-size="16" font-weight="bold">{title}</text>')
        pieces.append(f'<line x1="{x0}" y1="{y_base}" x2="{x0 + 300}" y2="{y_base}" stroke="#333"/>')
        pieces.append(f'<line x1="{x0}" y1="{top}" x2="{x0}" y2="{y_base}" stroke="#333"/>')
        for config_index in range(6):
            config_id = f"C{config_index}"
            rows = sorted(
                (row for row in per_run if row["config_id"] == config_id),
                key=lambda row: int(row["repetition"]),
            )
            points = []
            for row in rows:
                x = x0 + int(row["repetition"]) * 50
                value = float(row[metric]) / divisor
                y = y_base - (value - lo) / (hi - lo) * panel_h
                points.append((x, y))
            point_text = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
            pieces.append(f'<polyline points="{point_text}" fill="none" stroke="{colors[config_index]}" stroke-width="2"/>')
            for x, y in points:
                pieces.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="{colors[config_index]}"/>')
        for rep in range(1, 6):
            pieces.append(f'<text x="{x0 + rep * 50}" y="{y_base + 20}" text-anchor="middle" font-family="sans-serif" font-size="11">{rep}</text>')
    for index in range(6):
        x = 390 + index * 80
        pieces.append(f'<rect x="{x}" y="735" width="12" height="12" fill="{colors[index]}"/>')
        pieces.append(f'<text x="{x + 18}" y="746" font-family="sans-serif" font-size="12">C{index}</text>')
    pieces.append('</svg>')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(pieces) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/processed"))
    parser.add_argument(
        "--figure", type=Path,
        default=Path("results/figures/milestone_m_stationary_repetitions.svg"),
    )
    args = parser.parse_args()
    study = args.study_dir.resolve(strict=True)
    plan = json.loads((study / "manifest.json").read_text())
    if json.loads((study / "result.json").read_text())["status"] != "complete":
        raise ValueError("stationary repeat is not complete")
    repetitions = [int(value) for value in plan["repetitions"]]
    plan_by_id = {row["config_id"]: row for row in plan["configurations"]}
    if set(plan_by_id) != {f"C{index}" for index in range(6)}:
        raise ValueError("matrix does not contain exactly C0-C5")
    raw = _load_runs(study, int(plan["expected_runs"]))
    per_run = [_run_row(row, plan_by_id) for row in raw]
    per_run.sort(key=lambda row: (row["config_id"], row["repetition"]))
    aggregate_rows = [
        _aggregate(per_run, plan_by_id[f"C{index}"], repetitions)
        for index in range(6)
    ]
    aggregates = {row["config_id"]: row for row in aggregate_rows}
    comparison_rows = [
        _comparison(label, treatment, baseline, aggregates, per_run, repetitions)
        for label, treatment, baseline in COMPARISONS
    ]
    comparisons = {row["comparison"]: row for row in comparison_rows}
    placement_decision, placement_rationale = classify(comparisons)
    fixed_dvfs_decision, fixed_dvfs_rationale = classify_fixed_dvfs(comparisons)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "milestone_m_stationary_runs.csv", per_run)
    _write_csv(args.output_dir / "milestone_m_stationary_aggregate.csv", aggregate_rows)
    _write_csv(args.output_dir / "milestone_m_stationary_comparisons.csv", comparison_rows)
    summary = {
        "schema_version": 1,
        "study_dir": str(study),
        "request_rate": plan["request_rate"],
        "trace_sha256": plan["trace_sha256"],
        "configuration_order": plan["configuration_order"],
        "repetitions": repetitions,
        "validated_runs": len(per_run),
        "aggregates": aggregates,
        "comparisons": comparisons,
        "placement_effect_decision": placement_decision,
        "placement_effect_rationale": placement_rationale,
        "fixed_dvfs_effect_decision": fixed_dvfs_decision,
        "fixed_dvfs_effect_rationale": fixed_dvfs_rationale,
    }
    (args.output_dir / "milestone_m_stationary_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _figure(args.figure, per_run)
    print(json.dumps({
        "validated_runs": len(per_run),
        "placement_effect_decision": placement_decision,
        "fixed_dvfs_effect_decision": fixed_dvfs_decision,
        "outputs": {
            "runs": str(args.output_dir / "milestone_m_stationary_runs.csv"),
            "aggregate": str(args.output_dir / "milestone_m_stationary_aggregate.csv"),
            "comparisons": str(args.output_dir / "milestone_m_stationary_comparisons.csv"),
            "summary": str(args.output_dir / "milestone_m_stationary_summary.json"),
            "figure": str(args.figure),
        },
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
