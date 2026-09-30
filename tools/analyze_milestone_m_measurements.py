#!/usr/bin/env python3
"""Aggregate the fixed six-configuration Milestone M AFD experiment."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Any


KEY_METRICS = (
    "ttft_p99_ms",
    "tpot_p99_ms",
    "total_gpu_energy_j",
    "attention_pool_energy_j",
    "ffn_pool_energy_j",
    "throughput_requests_s",
)
AGGREGATE_METRICS = (
    "ttft_p50_ms", "ttft_p95_ms", "ttft_p99_ms",
    "tpot_p50_ms", "tpot_p95_ms", "tpot_p99_ms",
    "e2e_p50_ms", "e2e_p95_ms", "e2e_p99_ms",
    "throughput_requests_s", "throughput_output_tokens_s",
    "total_gpu_energy_j", "attention_pool_energy_j", "ffn_pool_energy_j",
    "joules_per_request", "client_send_delay_p99_ms",
)
COMPARISONS = (
    ("primary_mh", "C3", "C2"),
    ("primary_hm", "C5", "C4"),
    ("control_hh", "C1", "C0"),
    ("mh_vs_hh_reference", "C3", "C0"),
    ("hm_vs_hh_reference", "C5", "C0"),
)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def _load_runs(study: Path) -> list[dict[str, Any]]:
    with (study / "runs.tsv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    complete = [row for row in rows if row["status"] == "complete"]
    if len(rows) != 18 or len(complete) != 18:
        raise ValueError(f"expected 18 complete runs, got {len(complete)}/{len(rows)}")
    return complete


def _run_row(raw: dict[str, Any], plan_by_id: dict[str, Any]) -> dict[str, Any]:
    run_dir = Path(raw["run_dir"])
    metrics = json.loads((run_dir / "metrics.json").read_text())
    energy = json.loads((run_dir / "energy.json").read_text())
    metadata = json.loads((run_dir / "experiment_metadata.json").read_text())
    validation = json.loads((run_dir / "milestone-m-validation.json").read_text())
    config = plan_by_id[raw["config_id"]]
    if validation["status"] != "pass":
        raise ValueError(f"run did not pass validation: {run_dir}")
    target = list(config["target_clocks_mhz"])
    actual = [float(metrics[f"gpu{gpu}_avg_sm_clock"]) for gpu in range(4)]
    row: dict[str, Any] = {
        "config_id": raw["config_id"],
        "configuration_order": int(raw["order"]),
        "repetition": int(raw["repetition"]),
        "run_id": metrics["run_id"],
        "run_dir": str(run_dir),
        "placement": config["placement"]["name"],
        "placement_sha256": metadata["placement_sha256"],
        "actual_mapping_sha256": metadata["actual_mapping_sha256"],
        "trace_sha256": metadata["trace_sha256"],
        "experiment_plan_sha256": metadata["experiment_plan_sha256"],
        "git_commit": metadata["git_commit"],
        "plugin_commit": metadata["plugin_commit"],
        "vllm_version": metadata["vllm_version"],
        "execution_mode": metadata["execution_mode"],
        "completed_requests": metrics["completed_requests"],
        "failed_requests": metrics["failed_requests"],
        "output_tokens_total": metrics["output_tokens_total"],
        "duration_s": metrics["duration_s"],
        "energy_source": metrics["energy_source"],
        "target_clocks_mhz": "/".join(str(value) for value in target),
        "actual_clocks_mhz": "/".join(f"{value:.3f}" for value in actual),
        "attention_pool_energy_j": metrics["gpu0_energy_j"] + metrics["gpu1_energy_j"],
        "ffn_pool_energy_j": metrics["gpu2_energy_j"] + metrics["gpu3_energy_j"],
    }
    for key in (
        "ttft_p50_ms", "ttft_p95_ms", "ttft_p99_ms",
        "tpot_p50_ms", "tpot_p95_ms", "tpot_p99_ms",
        "e2e_p50_ms", "e2e_p95_ms", "e2e_p99_ms",
        "throughput_requests_s", "throughput_output_tokens_s",
        "total_gpu_energy_j", "joules_per_request", "joules_per_output_token",
        "client_send_delay_p99_ms",
    ):
        row[key] = metrics[key]
    for gpu in range(4):
        for suffix in ("energy_j", "avg_power_w", "avg_sm_util", "avg_memory_util", "avg_sm_clock"):
            row[f"gpu{gpu}_{suffix}"] = metrics[f"gpu{gpu}_{suffix}"]
    return row


def _aggregate(per_run: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    selected = [row for row in per_run if row["config_id"] == config["config_id"]]
    if sorted(row["repetition"] for row in selected) != [1, 2, 3]:
        raise ValueError(f"{config['config_id']} does not have repetitions 1,2,3")
    result: dict[str, Any] = {
        "config_id": config["config_id"],
        "configuration_order": config["order"],
        "placement": config["placement"]["name"],
        "target_clocks_mhz": "/".join(str(v) for v in config["target_clocks_mhz"]),
        "runs": 3,
        "completed_requests_total": sum(row["completed_requests"] for row in selected),
        "failed_requests_total": sum(row["failed_requests"] for row in selected),
    }
    for metric in AGGREGATE_METRICS:
        values = [float(row[metric]) for row in selected]
        result[f"{metric}_mean"] = mean(values)
        result[f"{metric}_sample_std"] = stdev(values)
        result[f"{metric}_values"] = ";".join(f"{value:.9f}" for value in values)
    return result


def _relative(treatment: float, baseline: float) -> float:
    if baseline == 0:
        raise ValueError("relative comparison baseline is zero")
    return (treatment - baseline) / baseline


def _comparison(
    label: str,
    treatment_id: str,
    baseline_id: str,
    aggregates: dict[str, dict[str, Any]],
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
    return row


def classify(comparisons: dict[str, dict[str, Any]]) -> tuple[str, str]:
    def acceptable(row: dict[str, Any]) -> bool:
        return (
            row["total_gpu_energy_j_relative_change"] <= 0.02
            and row["tpot_p99_ms_relative_change"] <= 0.02
            and row["ttft_p99_ms_relative_change"] <= 0.03
            and row["throughput_requests_s_relative_change"] >= -0.02
        )

    def repeatable_improvement(row: dict[str, Any], metric: str) -> bool:
        effect = row[f"{metric}_relative_change"]
        baseline = row[f"{metric}_baseline_mean"]
        variability = max(
            row[f"{metric}_treatment_sample_std"],
            row[f"{metric}_baseline_sample_std"],
        ) / baseline
        return effect <= -0.03 and abs(effect) > variability

    primary = [comparisons["primary_mh"], comparisons["primary_hm"]]
    control = comparisons["control_hh"]
    control_neutral = all(
        abs(control[f"{metric}_relative_change"]) <= 0.03
        for metric in (
            "ttft_p99_ms", "tpot_p99_ms", "total_gpu_energy_j",
            "throughput_requests_s",
        )
    )
    strong = all(
        acceptable(row)
        and (
            repeatable_improvement(row, "tpot_p99_ms")
            or repeatable_improvement(row, "total_gpu_energy_j")
        )
        for row in primary
    )
    if strong and control_neutral:
        return "A", "both mirrored comparisons pass the conservative 3% effect/variability rule and the H/H control is neutral"
    weak = any(
        acceptable(row)
        and (
            row["tpot_p99_ms_relative_change"] <= -0.01
            or row["total_gpu_energy_j_relative_change"] <= -0.01
        )
        for row in primary
    )
    if weak:
        return "B", "at least one mirrored comparison improves by 1%, but the strong mirrored/repeatability rule is not met"
    return "C", "neither mirrored comparison shows a practically meaningful latency/total-energy benefit"


def _figure(path: Path, aggregates: dict[str, dict[str, Any]]) -> None:
    pairs = (("MH", "C2", "C3"), ("HM", "C4", "C5"), ("HH control", "C0", "C1"))
    metrics = (
        ("TPOT p99 (ms)", "tpot_p99_ms"),
        ("Total energy (kJ)", "total_gpu_energy_j"),
        ("FFN energy (kJ)", "ffn_pool_energy_j"),
        ("Throughput (req/s)", "throughput_requests_s"),
    )
    width, height = 1200, 780
    panel_w, panel_h = 270, 560
    left, top = 75, 90
    colors = ("#4c78a8", "#e45756")
    pieces = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="60" y="42" font-family="sans-serif" font-size="24" font-weight="bold">Milestone M: fixed-frequency AFD placement comparison</text>',
        '<text x="60" y="68" font-family="sans-serif" font-size="14">Bars are three-run means; blue = balanced, red = frequency-aware placement.</text>',
    ]
    for panel, (title, metric) in enumerate(metrics):
        x0 = left + panel * panel_w
        scale_values = []
        for _label, baseline_id, treatment_id in pairs:
            divisor = 1000.0 if "energy" in metric else 1.0
            scale_values.extend([
                aggregates[baseline_id][f"{metric}_mean"] / divisor,
                aggregates[treatment_id][f"{metric}_mean"] / divisor,
            ])
        maximum = max(scale_values) * 1.15
        pieces.append(f'<text x="{x0 + 100}" y="{top}" text-anchor="middle" font-family="sans-serif" font-size="16" font-weight="bold">{title}</text>')
        y_base = top + panel_h
        pieces.append(f'<line x1="{x0}" y1="{y_base}" x2="{x0 + 220}" y2="{y_base}" stroke="#333"/>')
        for group, (label, baseline_id, treatment_id) in enumerate(pairs):
            divisor = 1000.0 if "energy" in metric else 1.0
            values = [
                aggregates[baseline_id][f"{metric}_mean"] / divisor,
                aggregates[treatment_id][f"{metric}_mean"] / divisor,
            ]
            for index, value in enumerate(values):
                bar_h = value / maximum * (panel_h - 70)
                x = x0 + 15 + group * 72 + index * 26
                y = y_base - bar_h
                pieces.append(f'<rect x="{x}" y="{y:.2f}" width="22" height="{bar_h:.2f}" fill="{colors[index]}"/>')
                pieces.append(f'<text x="{x + 11}" y="{y - 5:.2f}" text-anchor="middle" font-family="sans-serif" font-size="9">{value:.2f}</text>')
            pieces.append(f'<text x="{x0 + 39 + group * 72}" y="{y_base + 20}" text-anchor="middle" font-family="sans-serif" font-size="11">{label}</text>')
    pieces.append('</svg>')
    path.write_text("\n".join(pieces) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/processed"))
    parser.add_argument(
        "--figure", type=Path,
        default=Path("results/figures/milestone_m_afd_fixed_frequency.svg"),
    )
    args = parser.parse_args()
    study = args.study_dir.resolve(strict=True)
    plan = json.loads((study / "manifest.json").read_text())
    if json.loads((study / "result.json").read_text())["status"] != "complete":
        raise ValueError("matrix result is not complete")
    plan_by_id = {row["config_id"]: row for row in plan["configurations"]}
    if set(plan_by_id) != {f"C{index}" for index in range(6)}:
        raise ValueError("matrix does not contain exactly C0-C5")
    per_run = [_run_row(row, plan_by_id) for row in _load_runs(study)]
    per_run.sort(key=lambda row: (row["config_id"], row["repetition"]))
    aggregates_list = [
        _aggregate(per_run, plan_by_id[f"C{index}"]) for index in range(6)
    ]
    aggregates = {row["config_id"]: row for row in aggregates_list}
    comparison_rows = [
        _comparison(label, treatment, baseline, aggregates)
        for label, treatment, baseline in COMPARISONS
    ]
    comparisons = {row["comparison"]: row for row in comparison_rows}
    decision, rationale = classify(comparisons)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "milestone_m_runs.csv", per_run)
    _write_csv(output / "milestone_m_aggregate.csv", aggregates_list)
    _write_csv(output / "milestone_m_comparisons.csv", comparison_rows)
    summary = {
        "schema_version": 1,
        "study_dir": str(study),
        "execution_mode": plan["execution_mode"],
        "trace_sha256": plan["trace_sha256"],
        "configuration_order": plan["configuration_order"],
        "validated_runs": len(per_run),
        "aggregates": aggregates,
        "comparisons": comparisons,
        "decision": decision,
        "decision_rationale": rationale,
        "decision_rule": {
            "A": "both mirrors improve TPOT p99 or total energy by at least 3%, exceed run variability, avoid material regressions, and H/H control is within 3%",
            "B": "at least one mirror improves TPOT p99 or total energy by at least 1% without material regressions, but A is not met",
            "C": "otherwise",
        },
    }
    (output / "milestone_m_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    _figure(args.figure, aggregates)
    print(json.dumps({
        "decision": decision,
        "rationale": rationale,
        "validated_runs": len(per_run),
        "outputs": {
            "runs": str(output / "milestone_m_runs.csv"),
            "aggregate": str(output / "milestone_m_aggregate.csv"),
            "comparisons": str(output / "milestone_m_comparisons.csv"),
            "summary": str(output / "milestone_m_summary.json"),
            "figure": str(args.figure),
        },
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
