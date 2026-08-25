"""Pure request, latency, and run-summary metric calculations."""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping


def percentile(values: Iterable[float], probability: float) -> float | None:
    """NumPy-compatible linear percentile without requiring NumPy."""
    ordered = sorted(float(value) for value in values if value is not None)
    if not ordered:
        return None
    if not 0 <= probability <= 100:
        raise ValueError("percentile must be in [0, 100]")
    if len(ordered) == 1:
        return ordered[0]
    rank = probability / 100.0 * (len(ordered) - 1)
    low = math.floor(rank)
    high = math.ceil(rank)
    fraction = rank - low
    return ordered[low] * (1.0 - fraction) + ordered[high] * fraction


def latency_metrics(
    send_monotonic_s: float,
    first_token_monotonic_s: float | None,
    complete_monotonic_s: float,
    output_tokens: int,
) -> tuple[float | None, float | None, float]:
    if complete_monotonic_s < send_monotonic_s:
        raise ValueError("completion precedes send")
    e2e_ms = (complete_monotonic_s - send_monotonic_s) * 1000.0
    if first_token_monotonic_s is None:
        return None, None, e2e_ms
    if not send_monotonic_s <= first_token_monotonic_s <= complete_monotonic_s:
        raise ValueError("first-token timestamp is outside request lifetime")
    ttft_ms = (first_token_monotonic_s - send_monotonic_s) * 1000.0
    tpot_ms = None
    if output_tokens > 1:
        tpot_ms = (
            (complete_monotonic_s - first_token_monotonic_s)
            / (output_tokens - 1) * 1000.0
        )
    return ttft_ms, tpot_ms, e2e_ms


def _latency_summary(prefix: str, values: list[float]) -> dict[str, float | None]:
    return {
        f"{prefix}_p50_ms": percentile(values, 50),
        f"{prefix}_p95_ms": percentile(values, 95),
        f"{prefix}_p99_ms": percentile(values, 99),
    }


def summarize_run(
    *,
    run_id: str,
    workload: str,
    request_rate: float,
    request_results: Iterable[Mapping[str, Any]],
    measurement_duration_s: float,
    energy_summary: Mapping[str, Any],
) -> dict[str, Any]:
    rows = list(request_results)
    completed = [row for row in rows if row.get("status") == "ok"]
    ttfts = [float(row["ttft_ms"]) for row in completed if row.get("ttft_ms") is not None]
    tpots = [float(row["tpot_ms"]) for row in completed if row.get("tpot_ms") is not None]
    e2es = [float(row["e2e_ms"]) for row in completed if row.get("e2e_ms") is not None]
    delays = [float(row["send_delay_ms"]) for row in rows if row.get("send_delay_ms") is not None]
    input_tokens = sum(int(row.get("prompt_tokens") or 0) for row in completed)
    output_tokens = sum(int(row.get("output_tokens") or 0) for row in completed)
    total_energy = float(energy_summary["total_gpu_energy_j"])
    summary: dict[str, Any] = {
        "run_id": run_id,
        "workload": workload,
        "request_rate": float(request_rate),
        "duration_s": float(measurement_duration_s),
        "submitted_requests": len(rows),
        "completed_requests": len(completed),
        "failed_requests": len(rows) - len(completed),
        "input_tokens_total": input_tokens,
        "output_tokens_total": output_tokens,
        "throughput_requests_s": (len(completed) / measurement_duration_s
                                   if measurement_duration_s > 0 else None),
        "throughput_output_tokens_s": (output_tokens / measurement_duration_s
                                        if measurement_duration_s > 0 else None),
        "joules_per_request": (total_energy / len(completed) if completed else None),
        "joules_per_output_token": (total_energy / output_tokens if output_tokens else None),
        "client_send_delay_p99_ms": percentile(delays, 99),
    }
    summary.update(_latency_summary("ttft", ttfts))
    summary.update(_latency_summary("tpot", tpots))
    summary.update(_latency_summary("e2e", e2es))
    summary.update(energy_summary)
    return summary
