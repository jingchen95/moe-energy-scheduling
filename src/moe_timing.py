"""Aggregate rank-local FusedMoE CUDA-event measurements."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable


def aggregate_timing_records(
    records_by_rank: dict[int, Iterable[dict[str, Any]]],
) -> dict[str, Any]:
    """Sum within each rank and use the slowest rank as the EP critical path."""
    per_rank: dict[str, dict[str, float | int]] = {}
    total_records = 0

    for rank, records in sorted(records_by_rank.items()):
        elapsed_ms = 0.0
        record_count = 0
        for record in records:
            value = float(record["elapsed_ms"])
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"rank {rank}: invalid elapsed_ms {value!r}")
            elapsed_ms += value
            record_count += 1
        per_rank[str(rank)] = {
            "record_count": record_count,
            "total_ms": elapsed_ms,
        }
        total_records += record_count

    rank_totals = [float(value["total_ms"]) for value in per_rank.values()]
    return {
        "record_count": total_records,
        "per_rank": per_rank,
        "max_rank_total_ms": max(rank_totals, default=0.0),
        "sum_rank_total_ms": sum(rank_totals),
    }


def load_timing_files(paths: Iterable[Path], expected_ranks: int) -> dict[str, Any]:
    records_by_rank: dict[int, list[dict[str, Any]]] = {}
    for path in paths:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict) or value.get("schema_version") != 1:
            raise ValueError(f"invalid MoE timing document in {path}")
        rank = int(value["rank"])
        if rank in records_by_rank:
            raise ValueError(f"duplicate MoE timing rank {rank}")
        records = value.get("records")
        if not isinstance(records, list):
            raise ValueError(f"invalid MoE timing records in {path}")
        records_by_rank[rank] = records

    expected = set(range(expected_ranks))
    observed = set(records_by_rank)
    if observed != expected:
        raise ValueError(
            f"expected MoE timing ranks {sorted(expected)}, found {sorted(observed)}"
        )

    result = aggregate_timing_records(records_by_rank)
    empty_ranks = [
        rank
        for rank, values in result["per_rank"].items()
        if int(values["record_count"]) == 0
    ]
    if empty_ranks:
        raise ValueError(f"no MoE timing records for ranks {empty_ranks}")
    return result


def fused_moe_time_share_percent(
    timing: dict[str, Any], active_request_wall_time_ms: float
) -> float:
    if active_request_wall_time_ms <= 0:
        raise ValueError("active request wall time must be positive")
    return (
        float(timing["max_rank_total_ms"])
        / active_request_wall_time_ms
        * 100.0
    )
