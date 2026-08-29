"""Post-processing for rank-local FusedMoE timing records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def _phase_split(record: dict[str, Any]) -> tuple[float, float, float]:
    elapsed = float(record["elapsed_ms"])
    token_rows = int(record.get("token_rows", 0) or 0)
    decode_rows = record.get("decode_rows")
    if record.get("phase") == "decode":
        return 0.0, elapsed, float(decode_rows or token_rows)
    if record.get("phase") == "prefill":
        return elapsed, 0.0, 0.0
    if record.get("phase") == "mixed" and token_rows > 0 and decode_rows is not None:
        fraction = min(1.0, max(0.0, float(decode_rows) / token_rows))
        return elapsed * (1.0 - fraction), elapsed * fraction, float(decode_rows)
    return elapsed, 0.0, 0.0


def aggregate_timing_records(
    records_by_rank: dict[int, Iterable[dict[str, Any]]],
) -> dict[str, Any]:
    """Aggregate rank-local records without double-counting synchronized ranks.

    The sum is retained for per-rank accounting, while the maximum rank elapsed
    time is the system-service proxy because EP ranks execute in a synchronized
    step. Mixed phase intervals are split by their decode-row fraction only for
    descriptive phase totals.
    """
    per_rank: dict[int, dict[str, float]] = {}
    per_layer_phase: dict[str, dict[str, float]] = {}
    record_count = 0
    for rank, records in records_by_rank.items():
        rank_totals = {"total_ms": 0.0, "prefill_ms": 0.0, "decode_ms": 0.0,
                       "decode_rows": 0.0, "token_rows": 0.0}
        for record in records:
            record_count += 1
            prefill_ms, decode_ms, decode_rows = _phase_split(record)
            total_ms = float(record["elapsed_ms"])
            rank_totals["total_ms"] += total_ms
            rank_totals["prefill_ms"] += prefill_ms
            rank_totals["decode_ms"] += decode_ms
            rank_totals["decode_rows"] += decode_rows
            rank_totals["token_rows"] += int(record.get("token_rows", 0) or 0)
            layer = str(record.get("layer", "unknown"))
            layer_totals = per_layer_phase.setdefault(
                layer, {"total_ms": 0.0, "prefill_ms": 0.0, "decode_ms": 0.0}
            )
            layer_totals["total_ms"] += total_ms
            layer_totals["prefill_ms"] += prefill_ms
            layer_totals["decode_ms"] += decode_ms
        per_rank[int(rank)] = rank_totals
    rank_total_values = [value["total_ms"] for value in per_rank.values()]
    return {
        "record_count": record_count,
        "per_rank": {str(rank): values for rank, values in sorted(per_rank.items())},
        "per_layer": per_layer_phase,
        "max_rank_total_ms": max(rank_total_values, default=0.0),
        "sum_rank_total_ms": sum(rank_total_values),
        "max_rank_prefill_ms": max(
            (value["prefill_ms"] for value in per_rank.values()), default=0.0
        ),
        "max_rank_decode_ms": max(
            (value["decode_ms"] for value in per_rank.values()), default=0.0
        ),
    }


def load_timing_files(paths: Iterable[Path]) -> dict[str, Any]:
    by_rank: dict[int, list[dict[str, Any]]] = {}
    for path in paths:
        value = json.loads(path.read_text(encoding="utf-8"))
        rank = int(value["rank"])
        by_rank[rank] = list(value.get("records") or [])
    return aggregate_timing_records(by_rank)
