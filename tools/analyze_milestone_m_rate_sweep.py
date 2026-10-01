#!/usr/bin/env python3
"""Analyze stationarity and queue growth in the Milestone M rate sweep."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any


SUMMARY_METRICS = (
    "throughput_requests_s",
    "ttft_p50_ms", "ttft_p95_ms", "ttft_p99_ms",
    "tpot_p50_ms", "tpot_p95_ms", "tpot_p99_ms",
    "e2e_p50_ms", "e2e_p95_ms", "e2e_p99_ms",
    "total_gpu_energy_j", "attention_pool_energy_j", "ffn_pool_energy_j",
    "client_send_delay_p99_ms",
)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _request_trend(path: Path) -> dict[str, float]:
    requests = [json.loads(line) for line in path.read_text().splitlines() if line]
    requests = [row for row in requests if row["status"] == "ok"]
    requests.sort(key=lambda row: float(row["scheduled_send_s"]))
    if len(requests) < 30:
        raise ValueError(f"too few completed requests in {path}")
    width = len(requests) // 3
    early = requests[:width]
    late = requests[-width:]

    def med(rows: list[dict[str, Any]], key: str) -> float:
        return float(median(float(row[key]) for row in rows))

    def slope(key: str) -> float:
        values = [float(row[key]) for row in requests]
        center_x = (len(values) - 1) / 2.0
        center_y = mean(values)
        denominator = sum((index - center_x) ** 2 for index in range(len(values)))
        return sum(
            (index - center_x) * (value - center_y)
            for index, value in enumerate(values)
        ) / denominator

    early_ttft = med(early, "ttft_ms")
    late_ttft = med(late, "ttft_ms")
    early_tpot = med(early, "tpot_ms")
    late_tpot = med(late, "tpot_ms")
    return {
        "early_ttft_median_ms": early_ttft,
        "late_ttft_median_ms": late_ttft,
        "late_minus_early_ttft_ms": late_ttft - early_ttft,
        "late_over_early_ttft": late_ttft / early_ttft,
        "ttft_slope_ms_per_request": slope("ttft_ms"),
        "early_tpot_median_ms": early_tpot,
        "late_tpot_median_ms": late_tpot,
        "late_minus_early_tpot_ms": late_tpot - early_tpot,
        "tpot_slope_ms_per_request": slope("tpot_ms"),
    }


def analyze(study: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    plan = json.loads((study / "manifest.json").read_text())
    result = json.loads((study / "result.json").read_text())
    if result.get("status") != "complete":
        raise ValueError("rate sweep result is not complete")
    with (study / "runs.tsv").open(newline="", encoding="utf-8") as stream:
        raw_rows = list(csv.DictReader(stream, delimiter="\t"))
    expected = int(plan["expected_runs"])
    if len(raw_rows) != expected or any(row["status"] != "complete" for row in raw_rows):
        raise ValueError(f"expected {expected} complete windows")

    per_run: list[dict[str, Any]] = []
    for raw in raw_rows:
        run_dir = Path(raw["run_dir"])
        metrics = json.loads((run_dir / "metrics.json").read_text())
        validation = json.loads((run_dir / "milestone-m-validation.json").read_text())
        if validation["status"] != "pass":
            raise ValueError(f"validation failed: {run_dir}")
        offered = float(raw["request_rate"])
        row: dict[str, Any] = {
            "order": int(raw["order"]),
            "request_rate": offered,
            "repetition": int(raw["repetition"]),
            "run_id": metrics["run_id"],
            "run_dir": str(run_dir),
            "completed_requests": int(metrics["completed_requests"]),
            "failed_requests": int(metrics["failed_requests"]),
            "achieved_over_offered": float(metrics["throughput_requests_s"]) / offered,
        }
        for key in SUMMARY_METRICS:
            if key == "attention_pool_energy_j":
                row[key] = float(metrics["gpu0_energy_j"]) + float(metrics["gpu1_energy_j"])
            elif key == "ffn_pool_energy_j":
                row[key] = float(metrics["gpu2_energy_j"]) + float(metrics["gpu3_energy_j"])
            else:
                row[key] = metrics[key]
        row.update(_request_trend(run_dir / "requests.jsonl"))
        per_run.append(row)
    per_run.sort(key=lambda row: row["order"])

    aggregate: list[dict[str, Any]] = []
    for rate in sorted({float(row["request_rate"]) for row in per_run}):
        selected = sorted(
            (row for row in per_run if float(row["request_rate"]) == rate),
            key=lambda row: int(row["repetition"]),
        )
        if [row["repetition"] for row in selected] != [1, 2, 3]:
            raise ValueError(f"rate {rate} is missing repetitions")
        output: dict[str, Any] = {
            "request_rate": rate,
            "runs": len(selected),
            "completed_requests_total": sum(row["completed_requests"] for row in selected),
            "failed_requests_total": sum(row["failed_requests"] for row in selected),
        }
        metrics = (
            *SUMMARY_METRICS,
            "achieved_over_offered",
            "late_minus_early_ttft_ms", "late_over_early_ttft",
            "ttft_slope_ms_per_request", "late_minus_early_tpot_ms",
            "tpot_slope_ms_per_request",
        )
        for metric in metrics:
            values = [float(row[metric]) for row in selected]
            output[f"{metric}_mean"] = mean(values)
            output[f"{metric}_sample_std"] = stdev(values)
            output[f"{metric}_values"] = ";".join(f"{value:.9f}" for value in values)
        aggregate.append(output)
    return per_run, aggregate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/processed"))
    args = parser.parse_args()
    per_run, aggregate = analyze(args.study_dir.resolve(strict=True))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "milestone_m_rate_sweep_runs.csv", per_run)
    _write_csv(args.output_dir / "milestone_m_rate_sweep_aggregate.csv", aggregate)
    print(json.dumps({
        "validated_runs": len(per_run),
        "rates": [row["request_rate"] for row in aggregate],
        "outputs": {
            "runs": str(args.output_dir / "milestone_m_rate_sweep_runs.csv"),
            "aggregate": str(args.output_dir / "milestone_m_rate_sweep_aggregate.csv"),
        },
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
