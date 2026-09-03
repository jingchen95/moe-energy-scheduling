"""Pure latency and active-request-wall-time calculations."""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping


def latency_metrics(
    send_monotonic_s: float,
    first_token_monotonic_s: float | None,
    complete_monotonic_s: float,
    output_tokens: int | None,
) -> tuple[float | None, float | None, float]:
    """Return raw TTFT, TPOT, and end-to-end latency in milliseconds."""
    if complete_monotonic_s < send_monotonic_s:
        raise ValueError("completion precedes send")

    e2e_ms = (complete_monotonic_s - send_monotonic_s) * 1000.0
    if first_token_monotonic_s is None:
        return None, None, e2e_ms
    if not send_monotonic_s <= first_token_monotonic_s <= complete_monotonic_s:
        raise ValueError("first-token timestamp is outside request lifetime")

    ttft_ms = (first_token_monotonic_s - send_monotonic_s) * 1000.0
    tpot_ms = None
    if output_tokens is not None and output_tokens > 1:
        tpot_ms = (
            (complete_monotonic_s - first_token_monotonic_s)
            / (output_tokens - 1)
            * 1000.0
        )
    return ttft_ms, tpot_ms, e2e_ms


def active_request_time_ms(rows: Iterable[Mapping[str, Any]]) -> float:
    """Return the union of successful request lifetimes, excluding idle gaps."""
    intervals: list[tuple[float, float]] = []
    for row in rows:
        if row.get("status") != "ok":
            continue
        start_ms = float(row["actual_send_s"]) * 1000.0
        duration_ms = float(row["e2e_ms"])
        if not math.isfinite(start_ms) or not math.isfinite(duration_ms):
            raise ValueError("request interval must be finite")
        if duration_ms < 0:
            raise ValueError("request duration must be non-negative")
        intervals.append((start_ms, start_ms + duration_ms))

    if not intervals:
        return 0.0

    intervals.sort()
    current_start, current_end = intervals[0]
    total_ms = 0.0
    for next_start, next_end in intervals[1:]:
        if next_start <= current_end:
            current_end = max(current_end, next_end)
        else:
            total_ms += current_end - current_start
            current_start, current_end = next_start, next_end
    return total_ms + current_end - current_start
