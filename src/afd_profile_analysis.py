"""Descriptive two-rank analysis for frozen AFD logical routing counts."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.expert_profile import validate_counts
from src.placement import Placement, frequency_aware_placement


HIGH_MHZ = 2520
MEDIUM_MHZ = 1635


def layer_rank_loads(
    counts: np.ndarray,
    logical_to_rank: np.ndarray,
    *,
    num_ranks: int = 2,
) -> np.ndarray:
    weights = validate_counts(counts)
    mapping = np.asarray(logical_to_rank)
    if mapping.shape != weights.shape:
        raise ValueError("mapping shape must match routing counts")
    if not np.issubdtype(mapping.dtype, np.integer):
        raise ValueError("mapping must contain integer rank IDs")
    if np.any(mapping < 0) or np.any(mapping >= num_ranks):
        raise ValueError("mapping contains an invalid rank")
    loads = np.zeros((weights.shape[0], num_ranks), dtype=np.int64)
    for rank in range(num_ranks):
        loads[:, rank] = np.where(mapping == rank, weights, 0).sum(axis=1)
    return loads


def _cv(values: np.ndarray, axis: int | None = None) -> np.ndarray | float:
    value = np.asarray(values, dtype=np.float64)
    mean = value.mean(axis=axis)
    std = value.std(axis=axis)
    if axis is None:
        return float(std / mean) if mean else 0.0
    return np.divide(std, mean, out=np.zeros_like(std), where=mean > 0)


def profile_diagnostics(
    counts: np.ndarray,
    default_mapping: np.ndarray,
) -> dict[str, Any]:
    weights = validate_counts(counts)
    if weights.shape != (48, 128):
        raise ValueError(f"expected routing counts shape (48, 128), got {weights.shape}")
    expert_aggregate = weights.sum(axis=0).astype(np.float64)
    expert_layer_mean = weights.mean(axis=1, dtype=np.float64)
    expert_layer_max = weights.max(axis=1).astype(np.float64)
    expert_layer_cv = np.asarray(_cv(weights, axis=1))
    expert_layer_max_mean = np.divide(
        expert_layer_max,
        expert_layer_mean,
        out=np.zeros_like(expert_layer_max),
        where=expert_layer_mean > 0,
    )
    rank_loads = layer_rank_loads(weights, default_mapping)
    rank_totals = rank_loads.sum(axis=0)
    rank_layer_mean = rank_loads.mean(axis=1, dtype=np.float64)
    rank_layer_max = rank_loads.max(axis=1).astype(np.float64)
    rank_layer_max_mean = np.divide(
        rank_layer_max,
        rank_layer_mean,
        out=np.zeros_like(rank_layer_max),
        where=rank_layer_mean > 0,
    )
    rank_layer_fractions = np.divide(
        rank_loads,
        rank_loads.sum(axis=1, keepdims=True),
        out=np.zeros_like(rank_loads, dtype=np.float64),
        where=rank_loads.sum(axis=1, keepdims=True) > 0,
    )
    rank_layer_cv = np.asarray(_cv(rank_loads, axis=1))
    return {
        "logical_expert": {
            "aggregate_by_expert": expert_aggregate.astype(np.int64).tolist(),
            "aggregate_max_over_mean": (
                float(expert_aggregate.max() / expert_aggregate.mean())
                if expert_aggregate.mean() else 0.0
            ),
            "aggregate_cv": _cv(expert_aggregate),
            "layer_max_over_mean": expert_layer_max_mean.tolist(),
            "layer_cv": expert_layer_cv.tolist(),
            "layer_max_over_mean_mean": float(expert_layer_max_mean.mean()),
            "layer_max_over_mean_max": float(expert_layer_max_mean.max()),
            "layer_cv_mean": float(expert_layer_cv.mean()),
        },
        "default_rank": {
            "aggregate_loads": rank_totals.astype(int).tolist(),
            "aggregate_fractions": (
                rank_totals / rank_totals.sum()
            ).tolist(),
            "aggregate_absolute_difference": int(abs(rank_totals[0] - rank_totals[1])),
            "aggregate_max_over_mean": (
                float(rank_totals.max() / rank_totals.mean())
                if rank_totals.mean() else 0.0
            ),
            "aggregate_cv": _cv(rank_totals),
            "layer_loads": rank_loads.astype(int).tolist(),
            "layer_fractions": rank_layer_fractions.tolist(),
            "layer_absolute_difference": np.abs(
                rank_loads[:, 0] - rank_loads[:, 1]
            ).astype(int).tolist(),
            "layer_max_over_mean": rank_layer_max_mean.tolist(),
            "layer_cv": rank_layer_cv.tolist(),
            "layer_max_over_mean_mean": float(rank_layer_max_mean.mean()),
            "layer_max_over_mean_max": float(rank_layer_max_mean.max()),
            "layer_cv_mean": float(rank_layer_cv.mean()),
        },
        "smoothing": {
            "mean_layer_cv_reduction_fraction": (
                float(1.0 - rank_layer_cv.mean() / expert_layer_cv.mean())
                if expert_layer_cv.mean() else 0.0
            ),
            "mean_layer_max_over_mean_reduction_fraction": (
                float(1.0 - rank_layer_max_mean.mean() / expert_layer_max_mean.mean())
                if expert_layer_max_mean.mean() else 0.0
            ),
        },
    }


def _renamed(placement: Placement, name: str, extra: dict[str, Any]) -> Placement:
    metadata = dict(placement.metadata)
    metadata.update(extra)
    result = Placement(name, placement.logical_to_rank.copy(), 2, metadata)
    result.validate()
    return result


def build_two_rank_candidates(counts: np.ndarray) -> dict[str, Placement]:
    weights = validate_counts(counts)
    if weights.shape != (48, 128):
        raise ValueError(f"expected routing counts shape (48, 128), got {weights.shape}")
    balanced_base = frequency_aware_placement(
        weights, (HIGH_MHZ, HIGH_MHZ), num_ranks=2
    )
    balanced = _renamed(
        balanced_base,
        "P_balanced",
        {"objective": "minimize per-layer maximum raw FFN-rank load"},
    )
    mh_base = frequency_aware_placement(
        weights, (MEDIUM_MHZ, HIGH_MHZ), num_ranks=2
    )
    mh = _renamed(
        mh_base,
        "P_MH",
        {
            "objective": "minimize per-layer maximum load/capacity pressure",
            "physical_ffn_frequencies_mhz": [MEDIUM_MHZ, HIGH_MHZ],
            "speed_proxy": "frequency / 2520; placement heuristic only",
        },
    )
    hm_mapping = 1 - mh.logical_to_rank
    hm = Placement(
        "P_HM",
        hm_mapping,
        2,
        {
            "algorithm": "exact physical-rank mirror of P_MH",
            "source_placement": "P_MH",
            "physical_ffn_frequencies_mhz": [HIGH_MHZ, MEDIUM_MHZ],
            "speed_factors": [1.0, MEDIUM_MHZ / HIGH_MHZ],
            "speed_proxy": "frequency / 2520; placement heuristic only",
            "replication": False,
        },
    )
    hm.validate()

    mh_loads = layer_rank_loads(weights, mh.logical_to_rank)
    hm_loads = layer_rank_loads(weights, hm.logical_to_rank)
    if np.any(mh_loads[:, 0] > mh_loads[:, 1]):
        raise ValueError("P_MH assigns more predicted load to the slower rank")
    if np.any(hm_loads[:, 1] > hm_loads[:, 0]):
        raise ValueError("P_HM assigns more predicted load to the slower rank")
    return {placement.name: placement for placement in (balanced, mh, hm)}


def placement_diagnostics(
    counts: np.ndarray,
    placement: Placement,
    frequencies_mhz: tuple[int, int],
) -> dict[str, Any]:
    loads = layer_rank_loads(counts, placement.logical_to_rank)
    speeds = np.asarray(frequencies_mhz, dtype=np.float64) / HIGH_MHZ
    pressure = loads / speeds[None, :]
    totals = loads.sum(axis=0)
    total_pressure = totals / speeds
    return {
        "placement": placement.name,
        "frequencies_mhz": list(frequencies_mhz),
        "aggregate_loads": totals.astype(int).tolist(),
        "aggregate_load_fractions": (totals / totals.sum()).tolist(),
        "aggregate_raw_max_over_mean": float(totals.max() / totals.mean()),
        "aggregate_normalized_pressure": total_pressure.tolist(),
        "aggregate_normalized_max_over_mean": float(
            total_pressure.max() / total_pressure.mean()
        ),
        "layer_loads": loads.astype(int).tolist(),
        "layer_normalized_pressure": pressure.tolist(),
        "layer_raw_max_over_mean_mean": float(
            np.mean(loads.max(axis=1) / loads.mean(axis=1))
        ),
        "layer_normalized_max_over_mean_mean": float(
            np.mean(pressure.max(axis=1) / pressure.mean(axis=1))
        ),
        "layer_normalized_max_over_mean_max": float(
            np.max(pressure.max(axis=1) / pressure.mean(axis=1))
        ),
    }
