"""Deterministic two-rank placements for explicit routed-token load splits."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.expert_profile import validate_counts
from src.placement import Placement


TARGET_SLOW_FRACTIONS = (0.50, 0.475, 0.45, 0.425, 0.40)


def _best_improving_swap(
    weights: np.ndarray,
    mapping: np.ndarray,
    target_load: float,
) -> tuple[int, int] | None:
    rank0 = np.flatnonzero(mapping == 0)
    rank1 = np.flatnonzero(mapping == 1)
    current_load = float(weights[rank0].sum())
    current_error = abs(current_load - target_load)
    candidate_loads = (
        current_load
        - weights[rank0].astype(np.float64)[:, None]
        + weights[rank1].astype(np.float64)[None, :]
    )
    errors = np.abs(candidate_loads - target_load)
    flat_index = int(np.argmin(errors))
    best_error = float(errors.flat[flat_index])
    if best_error >= current_error - 1e-12:
        return None
    row, column = np.unravel_index(flat_index, errors.shape)
    return int(rank0[row]), int(rank1[column])


def target_split_placement(
    counts: np.ndarray,
    base: Placement,
    target_rank0_fraction: float,
    *,
    name: str,
) -> Placement:
    """Target one per-layer rank-0 load fraction using cardinality-safe swaps.

    The construction is independent of GPU frequencies. Starting from the
    frozen balanced placement, it repeatedly applies the single expert swap
    that most reduces absolute routed-token error for that layer. Stable expert
    ordering and NumPy's first-minimum rule make ties deterministic.
    """
    weights = validate_counts(counts)
    base.validate()
    if base.num_ranks != 2 or base.logical_to_rank.shape != weights.shape:
        raise ValueError("base placement must be a matching two-rank mapping")
    target = float(target_rank0_fraction)
    if not 0.0 < target <= 0.5:
        raise ValueError("target rank-0 fraction must be in (0, 0.5]")

    mapping = base.logical_to_rank.copy()
    swaps_per_layer: list[int] = []
    for layer in range(weights.shape[0]):
        target_load = target * float(weights[layer].sum())
        swaps = 0
        while True:
            pair = _best_improving_swap(
                weights[layer], mapping[layer], target_load
            )
            if pair is None:
                break
            rank0_expert, rank1_expert = pair
            mapping[layer, rank0_expert] = 1
            mapping[layer, rank1_expert] = 0
            swaps += 1
            if swaps > base.experts_per_rank:
                raise RuntimeError("target-split local search did not converge")
        swaps_per_layer.append(swaps)

    placement = Placement(
        name,
        mapping,
        2,
        {
            "algorithm": "deterministic best-improving pair swaps from P_balanced",
            "base_placement": base.name,
            "target_rank0_routed_token_fraction": target,
            "frequency_independent": True,
            "swaps_per_layer": swaps_per_layer,
            "replication": False,
        },
    )
    placement.validate()
    return placement


def mirror_placement(placement: Placement, *, name: str) -> Placement:
    placement.validate()
    if placement.num_ranks != 2:
        raise ValueError("only two-rank placements can be mirrored")
    mirrored = Placement(
        name,
        1 - placement.logical_to_rank,
        2,
        {
            "algorithm": "exact physical-rank mirror",
            "source_placement": placement.name,
            "target_rank1_routed_token_fraction": placement.metadata[
                "target_rank0_routed_token_fraction"
            ],
            "frequency_independent": True,
            "replication": False,
        },
    )
    mirrored.validate()
    return mirrored


def split_diagnostics(
    counts: np.ndarray,
    placement: Placement,
    *,
    slow_rank: int,
    target_slow_fraction: float,
) -> dict[str, Any]:
    weights = validate_counts(counts)
    placement.validate()
    if slow_rank not in (0, 1):
        raise ValueError("slow_rank must be 0 or 1")
    mapping = placement.logical_to_rank
    layer_loads = np.stack(
        [np.where(mapping == rank, weights, 0).sum(axis=1) for rank in (0, 1)],
        axis=1,
    )
    layer_totals = layer_loads.sum(axis=1)
    layer_fractions = np.divide(
        layer_loads,
        layer_totals[:, None],
        out=np.zeros_like(layer_loads, dtype=np.float64),
        where=layer_totals[:, None] > 0,
    )
    totals = layer_loads.sum(axis=0)
    slow_fractions = layer_fractions[:, slow_rank]
    return {
        "placement": placement.name,
        "slow_rank": slow_rank,
        "target_slow_fraction": float(target_slow_fraction),
        "aggregate_rank_loads": totals.astype(int).tolist(),
        "aggregate_rank_fractions": (totals / totals.sum()).tolist(),
        "achieved_slow_fraction": float(totals[slow_rank] / totals.sum()),
        "layer_rank_loads": layer_loads.astype(int).tolist(),
        "layer_rank_fractions": layer_fractions.tolist(),
        "layer_slow_fraction_min": float(slow_fractions.min()),
        "layer_slow_fraction_mean": float(slow_fractions.mean()),
        "layer_slow_fraction_max": float(slow_fractions.max()),
        "layer_slow_fraction_max_abs_error": float(
            np.max(np.abs(slow_fractions - target_slow_fraction))
        ),
    }
