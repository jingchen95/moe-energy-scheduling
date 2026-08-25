"""Frozen logical-expert load profiles and descriptive diagnostics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def validate_counts(counts: np.ndarray) -> np.ndarray:
    value = np.asarray(counts)
    if value.ndim != 2 or value.shape[0] <= 0 or value.shape[1] <= 0:
        raise ValueError("expert counts must have shape [layers, experts]")
    if not np.issubdtype(value.dtype, np.integer):
        raise ValueError("expert counts must use an integer dtype")
    if np.any(value < 0):
        raise ValueError("expert counts must be non-negative")
    return value.astype(np.int64, copy=False)


def save_profile(
    path: Path, counts: np.ndarray, *, metadata: dict[str, Any] | None = None,
) -> None:
    value = validate_counts(counts)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        expert_counts=value,
        metadata_json=np.asarray(json.dumps(metadata or {}, sort_keys=True)),
    )


def load_profile(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    with np.load(path, allow_pickle=False) as archive:
        counts = validate_counts(archive["expert_counts"])
        metadata = json.loads(str(archive["metadata_json"].item()))
    return counts, metadata


def aggregate_rank_loads(counts: np.ndarray, logical_to_rank: np.ndarray) -> np.ndarray:
    weights = validate_counts(counts)
    mapping = np.asarray(logical_to_rank)
    if mapping.shape != weights.shape:
        raise ValueError("mapping shape must equal expert-count shape")
    if not np.issubdtype(mapping.dtype, np.integer) or np.any(mapping < 0):
        raise ValueError("mapping ranks must be non-negative integers")
    num_ranks = int(mapping.max()) + 1
    loads = np.zeros(num_ranks, dtype=np.int64)
    for rank in range(num_ranks):
        loads[rank] = weights[mapping == rank].sum()
    return loads


def load_diagnostics(counts: np.ndarray) -> dict[str, Any]:
    value = validate_counts(counts).astype(np.float64)
    means = value.mean(axis=1)
    maximums = value.max(axis=1)
    standard_deviations = value.std(axis=1)
    cvs = np.divide(standard_deviations, means, out=np.zeros_like(means), where=means > 0)
    ratios = np.divide(maximums, means, out=np.zeros_like(means), where=means > 0)
    return {
        "per_layer_mean": means.tolist(),
        "per_layer_max": maximums.tolist(),
        "per_layer_cv": cvs.tolist(),
        "per_layer_max_over_mean": ratios.tolist(),
        "aggregate": {
            "mean": float(value.mean()),
            "max": float(value.max()),
            "cv": float(value.std() / value.mean()) if value.mean() > 0 else 0.0,
            "max_over_mean": float(value.max() / value.mean()) if value.mean() > 0 else 0.0,
        },
    }
