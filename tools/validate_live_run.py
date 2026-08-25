#!/usr/bin/env python3
"""Fail closed when a live benchmark run is unsuitable for research use."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    metrics = json.loads((args.run_dir / "metrics.json").read_text())
    energy = json.loads((args.run_dir / "energy.json").read_text())
    requests = [json.loads(line) for line in (args.run_dir / "requests.jsonl").read_text().splitlines()]
    trace = [json.loads(line) for line in (args.run_dir / "trace.jsonl").read_text().splitlines()]
    if metrics["failed_requests"] or metrics["completed_requests"] != len(trace):
        raise SystemExit("not every validation request completed")
    for row in requests:
        if row["status"] != "ok" or row.get("streamed_text_event_count", 0) <= 0:
            raise SystemExit(f"request {row['request_id']} did not stream successfully")
        if not row.get("ttft_ms") or row.get("tpot_ms") is None:
            raise SystemExit(f"request {row['request_id']} lacks TTFT/TPOT")
        if row["output_tokens"] <= 1 or row["e2e_ms"] < row["ttft_ms"]:
            raise SystemExit(f"request {row['request_id']} has implausible latency/tokens")
    offsets = [float(row["arrival_offset_s"]) for row in trace]
    intervals = [right - left for left, right in zip(offsets, offsets[1:]) if right > left]
    if intervals and metrics["client_send_delay_p99_ms"] >= min(intervals) * 1000:
        raise SystemExit("client send delay is not small relative to the arrival interval")
    if energy["total_gpu_energy_j"] <= 0 or any(energy[f"gpu{gpu}_energy_j"] <= 0 for gpu in range(4)):
        raise SystemExit("four-GPU cumulative-energy delta is not positive")
    with (args.run_dir / "gpu_samples.csv").open(newline="", encoding="utf-8") as stream:
        samples = list(csv.DictReader(stream))
    sample_times = [float(row["timestamp_monotonic_s"]) for row in samples]
    interval = float(metrics["duration_s"])
    tolerance = 0.5
    if min(sample_times) > energy["measurement_start_monotonic_s"] + tolerance:
        raise SystemExit("GPU samples do not cover the measurement start")
    if max(sample_times) < energy["measurement_end_monotonic_s"] - tolerance:
        raise SystemExit("GPU samples do not cover the measurement end")
    if interval <= 0 or not (args.run_dir / "expert_counts.npz").exists():
        raise SystemExit("measurement window or expert-profile output is missing")
    print(json.dumps({"status": "PASS", "run_dir": str(args.run_dir),
                      "completed": len(requests),
                      "send_delay_p99_ms": metrics["client_send_delay_p99_ms"]}))


if __name__ == "__main__":
    main()
