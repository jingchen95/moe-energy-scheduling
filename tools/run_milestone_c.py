#!/usr/bin/env python3
"""Calibrate semantic request rates, then run repeated common-load baselines."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.workload import load_trace, write_trace


def _run_benchmark(
    trace: Path, rate: float, args: argparse.Namespace, log: Any,
) -> dict[str, Any]:
    command = [
        sys.executable, "-m", "src.benchmark",
        "--trace", str(trace),
        "--request-rate", str(rate),
        "--base-url", args.base_url,
        "--model", args.model,
        "--tokenizer", args.tokenizer,
        "--output-root", str(args.output_root),
        "--local-files-only",
        "--server-pid-file", str(args.server_pid_file),
        "--server-log", str(args.server_log),
    ]
    result = subprocess.run(command, check=True, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log.write(f"$ {' '.join(command)}\n{result.stdout}\n")
    log.flush()
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    return json.loads(lines[-1])


def _rescale(source: Path, output: Path, rate: float, limit: int) -> None:
    rows = load_trace(source)[:limit]
    write_trace(output, [replace(row, arrival_offset_s=index / rate)
                         for index, row in enumerate(rows)])


def _is_sustainable(metrics: dict[str, Any], rate: float, expected: int) -> bool:
    return (
        metrics["completed_requests"] == expected
        and metrics["failed_requests"] == 0
        and float(metrics["throughput_requests_s"]) >= 0.85 * rate
        and float(metrics["client_send_delay_p99_ms"]) <= 50.0
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--general", type=Path, required=True)
    parser.add_argument("--code", type=Path, required=True)
    parser.add_argument("--math", type=Path, required=True)
    parser.add_argument("--rates", type=float, nargs="+", default=[0.5, 1, 2, 4, 6, 8, 10])
    parser.add_argument("--calibration-count", type=int, default=30)
    parser.add_argument("--baseline-count", type=int, default=200)
    parser.add_argument("--moderate-fraction", type=float, default=0.65)
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="qwen3-30b-a3b")
    parser.add_argument("--tokenizer", default="Qwen/Qwen3-30B-A3B")
    parser.add_argument("--output-root", type=Path, default=Path("results/raw"))
    parser.add_argument("--processed-root", type=Path, default=Path("results/processed"))
    parser.add_argument("--server-pid-file", type=Path, required=True)
    parser.add_argument("--server-log", type=Path, required=True)
    args = parser.parse_args()
    if args.rates != sorted(args.rates) or any(rate <= 0 for rate in args.rates):
        parser.error("rates must be positive and increasing")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    work_dir = args.output_root / f"milestone-c-orchestration-{stamp}"
    work_dir.mkdir(parents=True, exist_ok=False)
    sources = {"general": args.general, "code": args.code, "math": args.math}
    manifest: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "rates": args.rates,
        "calibration_count": args.calibration_count,
        "baseline_count": args.baseline_count,
        "moderate_fraction": args.moderate_fraction,
        "calibration": {},
        "baseline": {},
    }
    with (work_dir / "orchestration.log").open("x", encoding="utf-8") as log:
        sustainable: dict[str, float] = {}
        for category, source in sources.items():
            evaluations = []
            for rate in args.rates:
                trace = work_dir / f"calibration-{category}-r{rate:g}.jsonl"
                _rescale(source, trace, rate, args.calibration_count)
                outcome = _run_benchmark(trace, rate, args, log)
                metrics = outcome["metrics"]
                feasible = _is_sustainable(metrics, rate, args.calibration_count)
                evaluations.append({
                    "rate": rate, "sustainable": feasible,
                    "run_dir": outcome["run_dir"], "metrics": metrics,
                })
            feasible_rates = [row["rate"] for row in evaluations if row["sustainable"]]
            if not feasible_rates:
                raise RuntimeError(f"no sustainable calibration rate for {category}")
            sustainable[category] = max(feasible_rates)
            manifest["calibration"][category] = {
                "estimated_sustainable_rate": sustainable[category],
                "right_censored": sustainable[category] == args.rates[-1],
                "evaluations": evaluations,
            }

        common_rate = args.moderate_fraction * min(sustainable.values())
        manifest["common_moderate_rate"] = common_rate
        for category, source in sources.items():
            trace = work_dir / f"baseline-{category}-medium-r{common_rate:g}.jsonl"
            _rescale(source, trace, common_rate, args.baseline_count)
            manifest["baseline"][category] = {"trace": str(trace), "runs": []}
            for repetition in range(args.repetitions):
                outcome = _run_benchmark(trace, common_rate, args, log)
                manifest["baseline"][category]["runs"].append({
                    "repetition": repetition, "run_dir": outcome["run_dir"],
                    "metrics": outcome["metrics"],
                })

    args.processed_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.processed_root / f"milestone-c-{stamp}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path),
                      "common_moderate_rate": manifest["common_moderate_rate"]}))


if __name__ == "__main__":
    main()
