#!/usr/bin/env python3
"""Validate and summarize the 21-run Milestone L GPU experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable, Sequence

import yaml


LOWER_IS_BETTER = ("ttft_p99_ms", "tpot_p99_ms", "total_gpu_energy_j")
METRICS = LOWER_IS_BETTER + ("throughput_requests_s",)
COMPARISONS = (
    ("A_MHHH", "C4", "C5"),
    ("B_MMHH", "C6", "C7"),
    ("C_HHHH_align_MHHH", "C1", "C2"),
    ("C_HHHH_align_MMHH", "C1", "C3"),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(fields), extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def complete_controller(study: Path, task_index: int, config_id: str) -> Path:
    candidates = []
    pattern = f"task-{task_index:02d}-{config_id}-attempt-*"
    for path in sorted((study / "measured").glob(pattern)):
        result_path = path / "result.json"
        runs_path = path / "runs.tsv"
        if not result_path.is_file() or not runs_path.is_file():
            continue
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
            rows = read_tsv(runs_path)
        except (OSError, json.JSONDecodeError, csv.Error):
            continue
        complete = [row for row in rows if row.get("status") == "complete"]
        exit_status = (path / "exit-status.txt").read_text(encoding="utf-8").strip()
        reset_path = path / "final-clock-reset.json"
        reset = json.loads(reset_path.read_text(encoding="utf-8")) if reset_path.is_file() else []
        if (result.get("status") == "complete" and len(complete) == 3
                and exit_status == "0" and len(reset) == 4):
            candidates.append(path)
    if len(candidates) != 1:
        raise ValueError(
            f"expected one complete controller for {config_id}, found {candidates}"
        )
    return candidates[0]


def mean_request_tpot(run_dir: Path) -> float:
    values = []
    with (run_dir / "requests.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("status") == "ok" and row.get("tpot_ms") is not None:
                values.append(float(row["tpot_ms"]))
    if not values:
        raise ValueError(f"no valid request TPOT values in {run_dir}")
    return statistics.fmean(values)


def validate_run(
    run_dir: Path, plan: dict[str, Any], spec: dict[str, Any], repetition: int,
) -> dict[str, Any]:
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    config = yaml.safe_load((run_dir / "resolved_config.yaml").read_text(encoding="utf-8"))
    placement = spec["placement"]
    clocks = tuple(int(value) for value in spec["frequency_tuple"])
    if sha256_file(run_dir / "trace.jsonl") != plan["trace_sha256"]:
        raise ValueError(f"trace artifact mismatch: {run_dir}")
    if config.get("trace_sha256") != plan["trace_sha256"]:
        raise ValueError(f"trace config mismatch: {run_dir}")
    if config.get("placement_sha256") != placement["placement_sha256"]:
        raise ValueError(f"placement hash mismatch: {run_dir}")
    if config.get("placement_name") != placement["runtime_placement_name"]:
        raise ValueError(f"placement name mismatch: {run_dir}")
    if config.get("model") != "qwen3-30b-a3b" or config.get("vllm_version") != "0.15.1":
        raise ValueError(f"runtime identity mismatch: {run_dir}")
    if not math.isclose(float(config.get("request_rate")), 16.0):
        raise ValueError(f"request-rate mismatch: {run_dir}")
    decoding = config.get("decoding") or {}
    if (float(decoding.get("temperature")) != 0.0
            or int(decoding.get("seed")) != 0
            or decoding.get("max_output_tokens") != [128]):
        raise ValueError(f"decoding mismatch: {run_dir}")
    if (int(metrics.get("completed_requests", 0)) != 300
            or int(metrics.get("failed_requests", 1)) != 0
            or int(metrics.get("output_tokens_total", 0)) != 300 * 128):
        raise ValueError(f"request completion mismatch: {run_dir}")
    if metrics.get("energy_source") != "nvml_total_energy_counter":
        raise ValueError(f"cumulative NVML energy missing: {run_dir}")
    if float(metrics.get("total_gpu_energy_j", 0)) <= 0:
        raise ValueError(f"non-positive GPU energy: {run_dir}")
    if float(metrics.get("client_send_delay_p99_ms", math.inf)) > 50.0:
        raise ValueError(f"client scheduling invalid: {run_dir}")
    for gpu, target in enumerate(clocks):
        actual = float(metrics[f"gpu{gpu}_avg_sm_clock"])
        if abs(actual - target) > 25:
            raise ValueError(
                f"GPU{gpu} actual clock {actual} differs from target {target}: {run_dir}"
            )

    row: dict[str, Any] = {
        "run_id": metrics["run_id"],
        "run_dir": str(run_dir),
        "config_id": spec["config_id"],
        "placement_id": spec["placement_id"],
        "placement_name": placement["runtime_placement_name"],
        "placement_sha256": placement["placement_sha256"],
        "frequency_tuple": "|".join(str(value) for value in clocks),
        "repetition": repetition,
        "trace_sha256": plan["trace_sha256"],
        "config_sha256": sha256_file(run_dir / "resolved_config.yaml"),
        "completed_requests": metrics["completed_requests"],
        "failed_requests": metrics["failed_requests"],
        "ttft_p50_ms": metrics["ttft_p50_ms"],
        "ttft_p95_ms": metrics["ttft_p95_ms"],
        "ttft_p99_ms": metrics["ttft_p99_ms"],
        "tpot_p50_ms": metrics["tpot_p50_ms"],
        "tpot_p95_ms": metrics["tpot_p95_ms"],
        "tpot_p99_ms": metrics["tpot_p99_ms"],
        "tpot_mean_ms": mean_request_tpot(run_dir),
        "e2e_p50_ms": metrics["e2e_p50_ms"],
        "e2e_p95_ms": metrics["e2e_p95_ms"],
        "e2e_p99_ms": metrics["e2e_p99_ms"],
        "throughput_requests_s": metrics["throughput_requests_s"],
        "total_gpu_energy_j": metrics["total_gpu_energy_j"],
        "joules_per_request": metrics["joules_per_request"],
        "client_send_delay_p99_ms": metrics["client_send_delay_p99_ms"],
    }
    for gpu, target in enumerate(clocks):
        row[f"gpu{gpu}_target_clock_mhz"] = target
        row[f"gpu{gpu}_actual_clock_mhz"] = metrics[f"gpu{gpu}_avg_sm_clock"]
        row[f"gpu{gpu}_energy_j"] = metrics[f"gpu{gpu}_energy_j"]
        row[f"gpu{gpu}_avg_power_w"] = metrics[f"gpu{gpu}_avg_power_w"]
        row[f"gpu{gpu}_avg_sm_util"] = metrics[f"gpu{gpu}_avg_sm_util"]
        row[f"gpu{gpu}_avg_memory_util"] = metrics[f"gpu{gpu}_avg_memory_util"]
    return row


def aggregate_runs(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row["config_id"]), []).append(row)
    output = []
    aggregate_metrics = (
        "ttft_p99_ms", "tpot_p99_ms", "tpot_p50_ms", "tpot_mean_ms",
        "e2e_p99_ms", "total_gpu_energy_j", "joules_per_request",
        "throughput_requests_s", "client_send_delay_p99_ms",
    )
    for config_id, group in sorted(groups.items()):
        if len(group) != 3 or {int(row["repetition"]) for row in group} != {0, 1, 2}:
            raise ValueError(f"{config_id} does not contain repetitions 0/1/2")
        value: dict[str, Any] = {
            "config_id": config_id,
            "placement_id": group[0]["placement_id"],
            "placement_name": group[0]["placement_name"],
            "placement_sha256": group[0]["placement_sha256"],
            "frequency_tuple": group[0]["frequency_tuple"],
            "trace_sha256": group[0]["trace_sha256"],
            "repetitions": 3,
            "completed_requests_total": sum(int(row["completed_requests"]) for row in group),
            "failed_requests_total": sum(int(row["failed_requests"]) for row in group),
        }
        for metric in aggregate_metrics:
            samples = [float(row[metric]) for row in group]
            value[f"{metric}_values"] = "|".join(f"{sample:.9f}" for sample in samples)
            value[f"{metric}_mean"] = statistics.fmean(samples)
            value[f"{metric}_std"] = statistics.stdev(samples)
        for gpu in range(4):
            for metric in ("energy_j", "avg_power_w", "avg_sm_util", "actual_clock_mhz"):
                key = f"gpu{gpu}_{metric}"
                samples = [float(row[key]) for row in group]
                value[f"{key}_mean"] = statistics.fmean(samples)
                value[f"{key}_std"] = statistics.stdev(samples)
        output.append(value)
    if {row["config_id"] for row in output} != {f"C{index}" for index in range(1, 8)}:
        raise ValueError("aggregate configuration set is incomplete")
    return output


def compare_aggregates(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {str(row["config_id"]): row for row in rows}
    comparisons = []
    for name, original_id, aligned_id in COMPARISONS:
        original, aligned = by_id[original_id], by_id[aligned_id]
        value: dict[str, Any] = {
            "comparison": name,
            "original_config": original_id,
            "aligned_config": aligned_id,
            "frequency_tuple": original["frequency_tuple"],
            "original_placement": original["placement_id"],
            "aligned_placement": aligned["placement_id"],
        }
        if original["frequency_tuple"] != aligned["frequency_tuple"]:
            raise ValueError(f"comparison {name} does not use identical frequencies")
        for metric in METRICS:
            original_mean = float(original[f"{metric}_mean"])
            aligned_mean = float(aligned[f"{metric}_mean"])
            value[f"original_{metric}_mean"] = original_mean
            value[f"aligned_{metric}_mean"] = aligned_mean
            value[f"aligned_vs_original_{metric}_pct"] = (
                100.0 * (aligned_mean - original_mean) / original_mean
            )
            value[f"{metric}_benefit_pct"] = (
                100.0 * (original_mean - aligned_mean) / original_mean
                if metric in LOWER_IS_BETTER
                else 100.0 * (aligned_mean - original_mean) / original_mean
            )
        comparisons.append(value)
    return comparisons


def classify_result(comparisons: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Apply conservative effect-size gates; three repeats do not imply significance."""
    controls = [row for row in comparisons if str(row["comparison"]).startswith("C_HHHH")]
    heterogeneous = [row for row in comparisons if row not in controls]
    control_metrics = ("ttft_p99_ms", "tpot_p99_ms", "total_gpu_energy_j")
    maximum_control_change = max(
        abs(float(row[f"aligned_vs_original_{metric}_pct"]))
        for row in controls for metric in control_metrics
    )
    maximum_heterogeneous_benefit = max(
        float(row[f"{metric}_benefit_pct"])
        for row in heterogeneous for metric in control_metrics
    )
    if maximum_control_change >= 5.0:
        decision = "D_HHHH_CONTROL_NOT_NEUTRAL"
    elif maximum_heterogeneous_benefit >= 5.0:
        decision = "A_MEASURED_INTERACTION_CONFIRMED"
    elif maximum_heterogeneous_benefit >= 1.0:
        decision = "B_SMALL_MEASURED_EFFECT"
    else:
        decision = "C_NO_MEASURED_EFFECT"
    return {
        "decision": decision,
        "maximum_hhhh_control_absolute_change_pct": maximum_control_change,
        "maximum_heterogeneous_latency_or_energy_benefit_pct": maximum_heterogeneous_benefit,
        "effect_size_gates_pct": {"material": 5.0, "small": 1.0},
        "significance_claim": False,
    }


def write_figure(path: Path, aggregate: Sequence[dict[str, Any]]) -> None:
    by_id = {str(row["config_id"]): row for row in aggregate}
    panels = (
        ("MHHH", "C4", "C5"),
        ("MMHH", "C6", "C7"),
        ("HHHH controls", "C1", "C2", "C3"),
    )
    metrics = (
        ("TPOT p99 (ms)", "tpot_p99_ms_mean", "tpot_p99_ms_std"),
        ("TTFT p99 (ms)", "ttft_p99_ms_mean", "ttft_p99_ms_std"),
        ("Total GPU energy (J)", "total_gpu_energy_j_mean", "total_gpu_energy_j_std"),
    )
    width, height = 1120, 720
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#222}.title{font-size:20px;font-weight:700}.panel{font-size:14px;font-weight:700}.axis{font-size:10px}.note{font-size:11px;fill:#444}</style>',
        '<text x="560" y="28" text-anchor="middle" class="title">Original versus intact-group-aligned P1</text>',
    ]
    colors = ("#4e79a7", "#e15759", "#59a14f")
    for metric_index, (metric_label, mean_key, std_key) in enumerate(metrics):
        y0 = 65 + metric_index * 205
        svg.append(f'<text x="20" y="{y0 + 85}" text-anchor="middle" class="panel" transform="rotate(-90 20 {y0 + 85})">{metric_label}</text>')
        for panel_index, panel in enumerate(panels):
            name, *ids = panel
            x0 = 65 + panel_index * 350
            values = [float(by_id[config_id][mean_key]) for config_id in ids]
            maximum = max(values) * 1.15
            svg.append(f'<text x="{x0 + 145}" y="{y0}" text-anchor="middle" class="panel">{name}</text>')
            for index, config_id in enumerate(ids):
                value = float(by_id[config_id][mean_key])
                std = float(by_id[config_id][std_key])
                bar_height = 140 * value / maximum
                x = x0 + 45 + index * 78
                y = y0 + 158 - bar_height
                svg.append(f'<rect x="{x}" y="{y}" width="48" height="{bar_height}" fill="{colors[index]}"/>')
                svg.append(f'<line x1="{x + 24}" x2="{x + 24}" y1="{y - 140 * std / maximum}" y2="{y + 140 * std / maximum}" stroke="#222"/>')
                svg.append(f'<text x="{x + 24}" y="{y0 + 178}" text-anchor="middle" class="axis">{config_id}</text>')
                svg.append(f'<title>{config_id}: {value:.3f} +/- {std:.3f}</title>')
            svg.append(f'<line x1="{x0 + 25}" x2="{x0 + 280}" y1="{y0 + 158}" y2="{y0 + 158}" stroke="#444"/>')
    svg.append('<text x="560" y="698" text-anchor="middle" class="note">Bars are means across three runs; error bars are sample standard deviations.</text>')
    svg.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(svg) + "\n", encoding="utf-8")


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    study = args.study.resolve()
    plan = json.loads((study / "experiment-plan.json").read_text(encoding="utf-8"))
    if plan.get("study") != "real_gpu_p1_group_alignment_validation":
        raise ValueError("not a Milestone L study")
    if sha256_file(Path(plan["trace"])) != plan["trace_sha256"]:
        raise ValueError("frozen source trace hash mismatch")
    for placement in plan["placement_specs"].values():
        if sha256_file(Path(placement["placement_file"])) != placement["placement_sha256"]:
            raise ValueError(f"frozen placement hash mismatch: {placement['placement_id']}")
    rows = []
    controllers = []
    for spec in plan["configurations"]:
        controller = complete_controller(study, int(spec["task_index"]), spec["config_id"])
        controllers.append(str(controller))
        expected_repetitions = {int(run["repetition"]) for run in spec["runs"]}
        observed_repetitions = set()
        for entry in read_tsv(controller / "runs.tsv"):
            if entry.get("status") != "complete":
                continue
            repetition = int(entry["repetition"])
            observed_repetitions.add(repetition)
            rows.append(validate_run(Path(entry["run_dir"]), plan, spec, repetition))
        if observed_repetitions != expected_repetitions:
            raise ValueError(f"repetition mismatch for {spec['config_id']}")
    if len(rows) != 21:
        raise ValueError(f"expected 21 valid runs, found {len(rows)}")
    aggregate = aggregate_runs(rows)
    comparisons = compare_aggregates(aggregate)
    decision = classify_result(comparisons)
    write_csv(args.output_dir / "milestone_l_runs.csv", rows, list(rows[0]))
    write_csv(args.output_dir / "milestone_l_aggregate.csv", aggregate, list(aggregate[0]))
    write_csv(args.output_dir / "milestone_l_comparisons.csv", comparisons, list(comparisons[0]))
    write_figure(args.figure, aggregate)
    payload = {
        "schema_version": 1,
        "study": plan["study"],
        "study_directory": str(study),
        "plan_sha256": sha256_file(study / "experiment-plan.json"),
        "trace_sha256": plan["trace_sha256"],
        "validated_runs": len(rows),
        "aggregate_configurations": len(aggregate),
        "controllers": controllers,
        "comparisons": comparisons,
        "decision": decision,
        "fused_moe_timing_collected": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "milestone_l_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/processed"))
    parser.add_argument(
        "--figure", type=Path,
        default=Path("results/figures/milestone_l_alignment_validation.svg"),
    )
    args = parser.parse_args()
    analyze(args)


if __name__ == "__main__":
    main()
