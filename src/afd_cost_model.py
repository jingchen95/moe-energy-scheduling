"""Offline feature construction and simple empirical AFD FFN cost models."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from src.afd_static_placement import load_placement


@dataclass(frozen=True)
class BinnedCurve:
    centers: np.ndarray
    values: np.ndarray
    counts: np.ndarray

    def predict(self, n: np.ndarray) -> np.ndarray:
        if not len(self.centers):
            raise ValueError("cannot predict with an empty curve")
        return np.interp(n, self.centers, self.values)


def fit_binned_curve(
    n: np.ndarray, elapsed_ms: np.ndarray, *, bins: int = 12
) -> BinnedCurve:
    n = np.asarray(n, dtype=float)
    elapsed = np.asarray(elapsed_ms, dtype=float)
    if n.shape != elapsed.shape or not len(n):
        raise ValueError("n and elapsed_ms must be non-empty matching arrays")
    if np.any(n < 0) or np.any(elapsed <= 0):
        raise ValueError("invalid latency training values")
    edges = np.unique(np.quantile(n, np.linspace(0.0, 1.0, bins + 1)))
    if len(edges) == 1:
        return BinnedCurve(
            centers=np.asarray([float(edges[0])]),
            values=np.asarray([float(np.median(elapsed))]),
            counts=np.asarray([len(n)], dtype=int),
        )
    assignments = np.searchsorted(edges[1:-1], n, side="right")
    centers: list[float] = []
    values: list[float] = []
    counts: list[int] = []
    for index in range(len(edges) - 1):
        mask = assignments == index
        if not np.any(mask):
            continue
        centers.append(float(np.median(n[mask])))
        values.append(float(np.median(elapsed[mask])))
        counts.append(int(mask.sum()))
    order = np.argsort(centers)
    return BinnedCurve(
        centers=np.asarray(centers)[order],
        values=np.asarray(values)[order],
        counts=np.asarray(counts, dtype=int)[order],
    )


def prediction_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if actual.shape != predicted.shape or not len(actual):
        raise ValueError("actual and predicted must be non-empty matching arrays")
    error = predicted - actual
    absolute = np.abs(error)
    nonzero = np.abs(actual) > 1e-12
    return {
        "samples": int(len(actual)),
        "mae": float(np.mean(absolute)),
        "mape_percent": float(np.mean(absolute[nonzero] / np.abs(actual[nonzero])) * 100),
        "p95_absolute_error": float(np.percentile(absolute, 95)),
        "bias": float(np.mean(error)),
    }


def load_routing_trace(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        result = {key: np.asarray(archive[key]) for key in archive.files}
    required = {
        "layers", "call_sequences", "expert_histograms",
        "per_source_rank_expert_histograms", "token_rows_by_source_rank",
        "topk_assignments_by_source_rank",
    }
    if not required.issubset(result):
        raise ValueError(f"routing trace missing: {sorted(required - set(result))}")
    records = len(result["layers"])
    if result["expert_histograms"].shape != (records, 128):
        raise ValueError("routing trace must have 128 experts")
    if not np.array_equal(
        result["expert_histograms"].sum(axis=1),
        result["topk_assignments_by_source_rank"].sum(axis=0),
    ):
        raise ValueError("routing trace assignment totals mismatch")
    return result


def load_same_run_cost_profile(path: Path) -> dict[str, np.ndarray]:
    """Load routing features and both timing boundaries from one run."""
    with np.load(path, allow_pickle=False) as archive:
        result = {key: np.asarray(archive[key]) for key in archive.files}
    required = {
        "runtime_step_ids",
        "schema_version",
        "stages",
        "layers",
        "expert_histograms",
        "per_source_rank_expert_histograms",
        "token_rows_by_source_rank",
        "local_elapsed_ms_by_rank",
        "full_elapsed_ms_by_rank",
        "frequency_mhz_by_rank",
    }
    if not required.issubset(result):
        raise ValueError(
            f"same-run cost profile missing: {sorted(required - set(result))}"
        )
    if int(result["schema_version"]) != 3:
        raise ValueError("schema-3 shared runtime identity required; legacy profiles are engineering-only")
    records = len(result["layers"])
    if result["expert_histograms"].shape != (records, 128):
        raise ValueError("same-run profile must contain 128-expert histograms")
    for key in (
        "per_source_rank_expert_histograms",
        "token_rows_by_source_rank",
        "local_elapsed_ms_by_rank",
        "full_elapsed_ms_by_rank",
    ):
        expected = (2, records, 128) if key == "per_source_rank_expert_histograms" else (2, records)
        if result[key].shape != expected:
            raise ValueError(f"invalid {key} shape: {result[key].shape}")
    keys = np.stack(
        [result["runtime_step_ids"], result["stages"], result["layers"]], axis=1
    )
    if len(np.unique(keys, axis=0)) != records:
        raise ValueError("same-run profile contains duplicate invocation keys")
    if np.any(result["local_elapsed_ms_by_rank"] <= 0) or np.any(
        result["full_elapsed_ms_by_rank"] <= 0
    ):
        raise ValueError("same-run profile contains non-positive timing")
    return result


def placement_features(
    routing: dict[str, np.ndarray], mapping: np.ndarray
) -> dict[int, dict[str, np.ndarray]]:
    mapping = np.asarray(mapping, dtype=np.int64)
    if mapping.shape != (48, 128):
        raise ValueError(f"invalid placement shape {mapping.shape}")
    layers = routing["layers"].astype(np.int64)
    histograms = routing["expert_histograms"].astype(np.int64)
    output: dict[int, dict[str, np.ndarray]] = {}
    for rank in range(2):
        local_histograms = np.zeros_like(histograms)
        for layer in range(48):
            row_indices = np.flatnonzero(layers == layer)
            expert_indices = np.flatnonzero(mapping[layer] == rank)
            local_histograms[np.ix_(row_indices, expert_indices)] = histograms[
                np.ix_(row_indices, expert_indices)
            ]
        n = local_histograms.sum(axis=1)
        active = (local_histograms > 0).sum(axis=1)
        maximum = local_histograms.max(axis=1)
        squared = np.square(local_histograms, dtype=np.float64).sum(axis=1)
        concentration = np.divide(
            squared,
            np.square(n, dtype=np.float64),
            out=np.zeros_like(squared),
            where=n > 0,
        )
        output[rank] = {
            "n": n,
            "active_local_experts": active,
            "max_tokens_per_local_expert": maximum,
            "routing_concentration": concentration,
            "local_expert_histograms": local_histograms,
        }
    return output


def join_timing_run(
    routing: dict[str, np.ndarray],
    timing_paths: Iterable[Path],
    placement_path: Path,
    *,
    config_id: str,
    lifecycle: str,
    frequency_mhz: int,
) -> tuple[list[dict[str, Any]], dict[int, np.ndarray]]:
    mapping, _, placement_name = load_placement(str(placement_path))
    features = placement_features(routing, np.asarray(mapping, dtype=np.int64))
    route_keys = np.stack(
        [routing["call_sequences"], routing["layers"]], axis=1
    ).astype(np.int64)
    key_to_index = {tuple(key): index for index, key in enumerate(route_keys)}
    rows: list[dict[str, Any]] = []
    histograms: dict[int, np.ndarray] = {}
    seen_ranks: set[int] = set()
    for path in timing_paths:
        with np.load(path, allow_pickle=False) as archive:
            rank = int(archive["ffn_rank"])
            observed_frequency = int(archive["frequency_mhz"])
            layers = np.asarray(archive["layers"], dtype=np.int64)
            sequences = np.asarray(archive["call_sequences"], dtype=np.int64)
            token_rows = np.asarray(archive["token_rows"], dtype=np.int64)
            elapsed = np.asarray(archive["elapsed_ms"], dtype=float)
        if rank in seen_ranks or observed_frequency != frequency_mhz:
            raise ValueError("timing rank duplicate or frequency mismatch")
        indices = np.asarray(
            [key_to_index.get((sequence, layer), -1) for sequence, layer in zip(sequences, layers)],
            dtype=np.int64,
        )
        if np.any(indices < 0) or len(indices) != len(route_keys):
            raise ValueError("timing and routing layer/call sequences do not align exactly")
        if len(np.unique(indices)) != len(indices):
            raise ValueError("timing keys contain duplicates")
        expected_token_rows = routing["token_rows_by_source_rank"][rank, indices]
        if not np.array_equal(token_rows, expected_token_rows):
            raise ValueError("timing token rows do not match routing source-rank trace")
        rank_features = features[rank]
        histograms[rank] = rank_features["local_expert_histograms"][indices]
        for local_index, route_index in enumerate(indices):
            rows.append({
                "config_id": config_id,
                "lifecycle": lifecycle,
                "placement": placement_name,
                "physical_ffn_rank": rank,
                "physical_gpu": rank + 2,
                "frequency_mhz": frequency_mhz,
                "layer": int(layers[local_index]),
                "call_sequence": int(sequences[local_index]),
                "token_rows": int(token_rows[local_index]),
                "n": int(rank_features["n"][route_index]),
                "active_local_experts": int(
                    rank_features["active_local_experts"][route_index]
                ),
                "max_tokens_per_local_expert": int(
                    rank_features["max_tokens_per_local_expert"][route_index]
                ),
                "routing_concentration": float(
                    rank_features["routing_concentration"][route_index]
                ),
                "elapsed_ms": float(elapsed[local_index]),
            })
        seen_ranks.add(rank)
    if seen_ranks != {0, 1}:
        raise ValueError(f"missing timing ranks: {seen_ranks}")
    return rows, histograms


def join_same_run_cost_profile(
    profile: dict[str, np.ndarray],
    placement_path: Path,
    *,
    config_id: str,
    lifecycle: str,
) -> tuple[list[dict[str, Any]], dict[int, np.ndarray]]:
    """Build invocation rows without any cross-run timing/routing join."""
    mapping, _, placement_name = load_placement(str(placement_path))
    routing = {
        "layers": profile["layers"],
        "expert_histograms": profile["expert_histograms"],
    }
    features = placement_features(routing, np.asarray(mapping, dtype=np.int64))
    rows: list[dict[str, Any]] = []
    histograms: dict[int, np.ndarray] = {}
    for rank in range(2):
        rank_features = features[rank]
        histograms[rank] = rank_features["local_expert_histograms"]
        frequency = int(profile["frequency_mhz_by_rank"][rank])
        for index in range(len(profile["layers"])):
            rows.append({
                "config_id": config_id,
                "lifecycle": lifecycle,
                "placement": placement_name,
                "physical_ffn_rank": rank,
                "physical_gpu": rank + 2,
                "frequency_mhz": frequency,
                "runtime_step_id": int(profile["runtime_step_ids"][index]),
                "stage": int(profile["stages"][index]),
                "layer": int(profile["layers"][index]),
                "token_rows": int(
                    profile["token_rows_by_source_rank"][rank, index]
                ),
                "n": int(rank_features["n"][index]),
                "active_local_experts": int(
                    rank_features["active_local_experts"][index]
                ),
                "max_tokens_per_local_expert": int(
                    rank_features["max_tokens_per_local_expert"][index]
                ),
                "routing_concentration": float(
                    rank_features["routing_concentration"][index]
                ),
                "local_elapsed_ms": float(
                    profile["local_elapsed_ms_by_rank"][rank, index]
                ),
                "full_elapsed_ms": float(
                    profile["full_elapsed_ms_by_rank"][rank, index]
                ),
            })
    return rows, histograms


def fit_linear_power(load_rate: np.ndarray, power_w: np.ndarray) -> tuple[float, float]:
    x = np.asarray(load_rate, dtype=float)
    y = np.asarray(power_w, dtype=float)
    if x.shape != y.shape or len(x) < 2 or np.ptp(x) <= 0:
        raise ValueError("power fit needs at least two distinct load rates")
    slope, intercept = np.polyfit(x, y, 1)
    return float(intercept), float(slope)


def leave_one_group_out_power(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    keys = sorted({(int(row["rank"]), int(row["frequency_mhz"])) for row in rows})
    for rank, frequency in keys:
        subset = [
            row for row in rows
            if int(row["rank"]) == rank and int(row["frequency_mhz"]) == frequency
        ]
        groups = sorted({str(row["config_id"]) for row in subset})
        for group in groups:
            train = [row for row in subset if str(row["config_id"]) != group]
            test = [row for row in subset if str(row["config_id"]) == group]
            if len({float(row["load_rate"]) for row in train}) < 2:
                continue
            intercept, slope = fit_linear_power(
                np.asarray([row["load_rate"] for row in train]),
                np.asarray([row["avg_power_w"] for row in train]),
            )
            for row in test:
                predicted_power = intercept + slope * float(row["load_rate"])
                predictions.append({
                    **row,
                    "predicted_power_w": predicted_power,
                    "predicted_energy_j": predicted_power * float(row["duration_s"]),
                })
    return predictions


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"no rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
