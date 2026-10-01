#!/usr/bin/env python3
"""Fail closed on one primary Milestone M AFD measurement."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import yaml


def validate_measured_run(
    run_dir: Path,
    *,
    trace_sha256: str,
    placement_sha256: str,
    mapping_sha256: str,
    target_clocks: list[int],
) -> dict[str, object]:
    metrics = json.loads((run_dir / "metrics.json").read_text())
    energy = json.loads((run_dir / "energy.json").read_text())
    config = yaml.safe_load((run_dir / "resolved_config.yaml").read_text())
    metadata = json.loads((run_dir / "experiment_metadata.json").read_text())
    if metrics["completed_requests"] != 300 or metrics["failed_requests"] != 0:
        raise ValueError("expected 300 completed requests and zero failures")
    if metrics["output_tokens_total"] != 300 * 128:
        raise ValueError("output-token total differs from the frozen workload")
    if energy["energy_source"] != "nvml_total_energy_counter":
        raise ValueError("cumulative NVML energy is required")
    if config["trace_sha256"] != trace_sha256:
        raise ValueError("trace hash mismatch")
    if config["placement_sha256"] != placement_sha256:
        raise ValueError("placement hash mismatch")
    if list(config["target_clocks_mhz"]) != target_clocks:
        raise ValueError("requested clock tuple mismatch")
    if metadata["actual_mapping_sha256"] != mapping_sha256:
        raise ValueError("runtime ownership mapping hash mismatch")
    if metadata["execution_mode"] != "FULL_DECODE_ONLY":
        raise ValueError("unexpected execution mode")
    actual = [float(metrics[f"gpu{gpu}_avg_sm_clock"]) for gpu in range(4)]
    for gpu, (observed, target) in enumerate(zip(actual, target_clocks)):
        if abs(observed - target) > 5:
            raise ValueError(
                f"GPU {gpu} average clock {observed} does not match {target} MHz"
            )
    if metrics["client_send_delay_p99_ms"] >= 62.5:
        raise ValueError("client p99 send delay exceeds one arrival interval")
    if config.get("arrival_schedule_source") == "request_rate_override":
        request_rate = float(config["request_rate"])
        requests = [
            json.loads(line)
            for line in (run_dir / "requests.jsonl").read_text().splitlines()
            if line
        ]
        for index, request in enumerate(requests):
            expected = index / request_rate
            if not math.isclose(
                float(request["scheduled_send_s"]), expected,
                rel_tol=0.0, abs_tol=1e-6,
            ):
                raise ValueError(
                    f"request {index} scheduled at {request['scheduled_send_s']}, "
                    f"expected {expected} for {request_rate} request/s"
                )
    return {
        "status": "pass",
        "run_id": metrics["run_id"],
        "completed_requests": metrics["completed_requests"],
        "failed_requests": metrics["failed_requests"],
        "output_tokens_total": metrics["output_tokens_total"],
        "energy_source": energy["energy_source"],
        "target_clocks_mhz": target_clocks,
        "actual_average_clocks_mhz": actual,
        "client_send_delay_p99_ms": metrics["client_send_delay_p99_ms"],
        "arrival_schedule_source": config.get("arrival_schedule_source", "trace"),
        "trace_sha256": trace_sha256,
        "placement_sha256": placement_sha256,
        "actual_mapping_sha256": mapping_sha256,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--trace-sha256", required=True)
    parser.add_argument("--placement-sha256", required=True)
    parser.add_argument("--mapping-sha256", required=True)
    parser.add_argument("--target-clocks", nargs=4, type=int, required=True)
    args = parser.parse_args()
    result = validate_measured_run(
        args.run_dir.resolve(strict=True),
        trace_sha256=args.trace_sha256,
        placement_sha256=args.placement_sha256,
        mapping_sha256=args.mapping_sha256,
        target_clocks=list(args.target_clocks),
    )
    output = args.run_dir / "milestone-m-validation.json"
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
