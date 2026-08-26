"""Deterministic search and analysis helpers for placement-DVFS experiments."""

from __future__ import annotations

import itertools
import random
import statistics
from typing import Any, Iterable, Mapping, Sequence


DEFAULT_FREQUENCIES_MHZ = (2520, 1635, 1005)


def enumerate_frequency_tuples(
    levels: Sequence[int] = DEFAULT_FREQUENCIES_MHZ, *, seed: int | None = None,
) -> list[tuple[int, ...]]:
    """Return the complete four-GPU clock space, optionally in a seeded order."""
    normalized = tuple(int(value) for value in levels)
    if not normalized or len(set(normalized)) != len(normalized):
        raise ValueError("frequency levels must be non-empty and unique")
    if any(value <= 0 for value in normalized):
        raise ValueError("frequency levels must be positive")
    tuples = list(itertools.product(normalized, repeat=4))
    if seed is not None:
        random.Random(seed).shuffle(tuples)
    return tuples


def slo_feasible(
    row: Mapping[str, Any], *, ttft_limit_ms: float, tpot_limit_ms: float,
) -> bool:
    """Apply common absolute latency limits and reject incomplete runs."""
    if int(row.get("failed_requests", 0) or 0) != 0:
        return False
    ttft = row.get("ttft_p99_ms")
    tpot = row.get("tpot_p99_ms")
    return (ttft is not None and tpot is not None
            and float(ttft) <= ttft_limit_ms and float(tpot) <= tpot_limit_ms)


def select_min_energy(
    rows: Iterable[Mapping[str, Any]], *, ttft_limit_ms: float,
    tpot_limit_ms: float, energy_key: str = "total_gpu_energy_j",
) -> Mapping[str, Any] | None:
    """Select the minimum-energy feasible measured configuration deterministically."""
    feasible = [row for row in rows if slo_feasible(
        row, ttft_limit_ms=ttft_limit_ms, tpot_limit_ms=tpot_limit_ms,
    ) and row.get(energy_key) is not None]
    if not feasible:
        return None
    return min(feasible, key=lambda row: (
        float(row[energy_key]), float(row["tpot_p99_ms"]),
        tuple(int(value) for value in row.get("frequency_tuple", ())),
        str(row.get("run_id", row.get("label", ""))),
    ))


def pareto_frontier(
    rows: Iterable[Mapping[str, Any]], *, x_key: str = "tpot_p99_ms",
    y_key: str = "total_gpu_energy_j",
) -> list[Mapping[str, Any]]:
    """Return nondominated points for two objectives that are both minimized."""
    valid = [row for row in rows if row.get(x_key) is not None and row.get(y_key) is not None]
    ordered = sorted(valid, key=lambda row: (
        float(row[x_key]), float(row[y_key]), str(row.get("run_id", row.get("label", ""))),
    ))
    result: list[Mapping[str, Any]] = []
    best_y = float("inf")
    for row in ordered:
        value = float(row[y_key])
        if value < best_y:
            result.append(row)
            best_y = value
    return result


def _frequency_tuple(row: Mapping[str, Any]) -> tuple[int, ...]:
    value = row.get("frequency_tuple", row.get("target_clocks_mhz"))
    if value is None:
        raise ValueError("configuration has no frequency tuple")
    result = tuple(int(item) for item in value)
    if len(result) != 4:
        raise ValueError("frequency tuple must contain four clocks")
    return result


def _step_distance(
    left: Sequence[int], right: Sequence[int], levels: Sequence[int],
) -> int:
    positions = {int(value): index for index, value in enumerate(levels)}
    try:
        return sum(abs(positions[int(a)] - positions[int(b)]) for a, b in zip(left, right))
    except KeyError as error:
        raise ValueError(f"unknown frequency level {error.args[0]}") from error


def select_joint_candidates(
    rows: Sequence[Mapping[str, Any]], *, tpot_limits_ms: Sequence[float],
    ttft_limit_ms: float, levels: Sequence[int] = DEFAULT_FREQUENCIES_MHZ,
    max_candidates: int = 8,
) -> list[dict[str, Any]]:
    """Choose a bounded, deterministic candidate set from the Sequential frontier.

    For each SLO this admits the best feasible tuple, the closest lower-energy
    one-step neighbor, and the closest TPOT miss. Remaining slots come from the
    measured Pareto frontier ordered by distance to any registered boundary.
    """
    if not 1 <= max_candidates <= 8:
        raise ValueError("max_candidates must be in [1, 8]")
    usable = [row for row in rows if int(row.get("failed_requests", 0) or 0) == 0
              and row.get("total_gpu_energy_j") is not None
              and row.get("ttft_p99_ms") is not None and row.get("tpot_p99_ms") is not None]
    if not usable:
        return []
    selected: dict[tuple[int, ...], dict[str, Any]] = {}

    def add(row: Mapping[str, Any], reason: str) -> None:
        clocks = _frequency_tuple(row)
        if clocks not in selected and len(selected) < max_candidates:
            selected[clocks] = {"frequency_tuple": clocks, "reason": reason,
                                "sequential_row": row}

    for limit in sorted(float(value) for value in tpot_limits_ms):
        best = select_min_energy(usable, ttft_limit_ms=ttft_limit_ms,
                                 tpot_limit_ms=limit)
        if best is not None:
            add(best, f"best_b4_slo_{limit:g}")
            best_clocks = _frequency_tuple(best)
            neighbors = [row for row in usable
                         if _step_distance(_frequency_tuple(row), best_clocks, levels) == 1
                         and float(row["total_gpu_energy_j"]) < float(best["total_gpu_energy_j"])]
            if neighbors:
                add(min(neighbors, key=lambda row: (
                    max(0.0, float(row["tpot_p99_ms"]) - limit),
                    float(row["total_gpu_energy_j"]), _frequency_tuple(row),
                )), f"lower_energy_neighbor_slo_{limit:g}")
        misses = [row for row in usable if float(row["ttft_p99_ms"]) <= ttft_limit_ms
                  and float(row["tpot_p99_ms"]) > limit]
        if misses:
            add(min(misses, key=lambda row: (
                float(row["tpot_p99_ms"]) - limit,
                float(row["total_gpu_energy_j"]), _frequency_tuple(row),
            )), f"closest_tpot_miss_slo_{limit:g}")

    frontier = pareto_frontier(usable)
    for row in sorted(frontier, key=lambda item: (
        min(abs(float(item["tpot_p99_ms"]) - float(limit)) for limit in tpot_limits_ms),
        float(item["total_gpu_energy_j"]), _frequency_tuple(item),
    )):
        add(row, "sequential_pareto_boundary")
    for row in sorted(usable, key=lambda item: (
        min(abs(float(item["tpot_p99_ms"]) - float(limit)) for limit in tpot_limits_ms),
        float(item["total_gpu_energy_j"]), _frequency_tuple(item),
    )):
        add(row, "sequential_near_slo_boundary")
    return list(selected.values())


def paired_comparison(
    sequential: Mapping[str, Any], joint: Mapping[str, Any],
) -> dict[str, float]:
    """Compare placements at an identical four-GPU frequency tuple."""
    if _frequency_tuple(sequential) != _frequency_tuple(joint):
        raise ValueError("paired configurations must use identical frequencies")
    result = {}
    for key in ("ttft_p99_ms", "tpot_p99_ms", "e2e_p99_ms",
                "total_gpu_energy_j", "joules_per_request"):
        if sequential.get(key) is None or joint.get(key) is None:
            raise ValueError(f"missing paired metric {key}")
        result[f"delta_{key}"] = float(joint[key]) - float(sequential[key])
    return result


def summarize_repeatability(
    rows: Sequence[Mapping[str, Any]], *, ttft_limit_ms: float,
    tpot_limit_ms: float,
    metrics: Sequence[str] = ("ttft_p99_ms", "tpot_p99_ms",
                              "total_gpu_energy_j", "joules_per_request"),
) -> dict[str, Any]:
    """Summarize trials without averaging away individual SLO violations."""
    if not rows:
        raise ValueError("at least one repeat is required")
    summary: dict[str, Any] = {
        "trials": len(rows),
        "feasible_trials": sum(slo_feasible(row, ttft_limit_ms=ttft_limit_ms,
                                             tpot_limit_ms=tpot_limit_ms) for row in rows),
    }
    summary["robustly_slo_feasible"] = summary["feasible_trials"] == len(rows)
    for metric in metrics:
        values = [float(row[metric]) for row in rows if row.get(metric) is not None]
        if len(values) != len(rows):
            raise ValueError(f"missing repeatability metric {metric}")
        summary[f"{metric}_mean"] = statistics.mean(values)
        summary[f"{metric}_stdev"] = statistics.stdev(values) if len(values) > 1 else 0.0
        summary[f"{metric}_min"] = min(values)
        summary[f"{metric}_max"] = max(values)
    return summary


def dvfs_sensitivity(
    reference: Mapping[str, Any], perturbation: Mapping[str, Any], gpu: int,
) -> dict[str, float | int]:
    if not 0 <= gpu < 4:
        raise ValueError("gpu must be in [0, 3]")
    required = ("ttft_p99_ms", "tpot_p99_ms", "e2e_p99_ms",
                "total_gpu_energy_j", "throughput_requests_s", f"gpu{gpu}_energy_j")
    for key in required:
        if reference.get(key) is None or perturbation.get(key) is None:
            raise ValueError(f"missing sensitivity metric {key}")
    return {
        "gpu": gpu,
        "delta_ttft_p99_ms": float(perturbation["ttft_p99_ms"] - reference["ttft_p99_ms"]),
        "delta_tpot_p99_ms": float(perturbation["tpot_p99_ms"] - reference["tpot_p99_ms"]),
        "delta_e2e_p99_ms": float(perturbation["e2e_p99_ms"] - reference["e2e_p99_ms"]),
        "delta_total_energy_j": float(
            perturbation["total_gpu_energy_j"] - reference["total_gpu_energy_j"]
        ),
        "delta_gpu_energy_j": float(
            perturbation[f"gpu{gpu}_energy_j"] - reference[f"gpu{gpu}_energy_j"]
        ),
        "delta_throughput_requests_s": float(
            perturbation["throughput_requests_s"] - reference["throughput_requests_s"]
        ),
    }
