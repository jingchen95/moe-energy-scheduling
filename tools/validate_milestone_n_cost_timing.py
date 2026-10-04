#!/usr/bin/env python3
"""Fail-fast validation of one same-run AFD routing/timing profile."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.afd_cost_model import (
    join_same_run_cost_profile,
    load_same_run_cost_profile,
)


def validate_average_clocks(
    metrics: dict[str, object], frequency_mhz: int, *, tolerance_mhz: float = 5.0
) -> tuple[list[int], list[float]]:
    target_clocks = [2520, 2520, frequency_mhz, frequency_mhz]
    actual_clocks = [
        float(metrics[f"gpu{gpu}_avg_sm_clock"]) for gpu in range(4)
    ]
    for gpu, (actual, target) in enumerate(zip(actual_clocks, target_clocks)):
        if abs(actual - target) > tolerance_mhz:
            raise ValueError(
                f"GPU {gpu} average clock {actual} does not match {target} MHz"
            )
    return target_clocks, actual_clocks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timing-run", type=Path, required=True)
    parser.add_argument("--placement", type=Path, required=True)
    parser.add_argument("--config-id", required=True)
    parser.add_argument("--lifecycle", required=True)
    parser.add_argument("--frequency-mhz", type=int, required=True)
    args = parser.parse_args()
    profile_path = args.timing_run / "afd_same_run_cost_profile.npz"
    metrics = json.loads((args.timing_run / "metrics.json").read_text())
    profile = load_same_run_cost_profile(profile_path)
    rows, _ = join_same_run_cost_profile(
        profile,
        args.placement,
        config_id=args.config_id,
        lifecycle=args.lifecycle,
    )
    observed = set(int(value) for value in profile["frequency_mhz_by_rank"])
    if observed != {args.frequency_mhz}:
        raise ValueError(
            f"profile frequency mismatch: expected {args.frequency_mhz}, got {observed}"
        )
    target_clocks, actual_clocks = validate_average_clocks(
        metrics, args.frequency_mhz
    )
    by_rank = {
        str(rank): sum(int(row["physical_ffn_rank"]) == rank for row in rows)
        for rank in (0, 1)
    }
    result = {
        "status": "pass",
        "config_id": args.config_id,
        "joined_records": len(rows),
        "records_by_rank": by_rank,
        "profile_sha256": __import__("hashlib").sha256(
            profile_path.read_bytes()
        ).hexdigest(),
        "same_run_routing_and_timing": True,
        "cross_run_join": False,
        "local_boundary_present": True,
        "full_boundary_present": True,
        "target_clocks_mhz": target_clocks,
        "actual_average_clocks_mhz": actual_clocks,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
