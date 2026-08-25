"""Strict static expert mappings for the placement-DVFS coupling pilot."""

from __future__ import annotations

import json
import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from src.expert_profile import aggregate_rank_loads, validate_counts


@dataclass(frozen=True)
class Placement:
    name: str
    logical_to_rank: np.ndarray
    num_ranks: int = 4
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def num_layers(self) -> int:
        return int(self.logical_to_rank.shape[0])

    @property
    def num_experts(self) -> int:
        return int(self.logical_to_rank.shape[1])

    @property
    def experts_per_rank(self) -> int:
        return self.num_experts // self.num_ranks

    def validate(self) -> None:
        mapping = np.asarray(self.logical_to_rank)
        if mapping.ndim != 2 or not np.issubdtype(mapping.dtype, np.integer):
            raise ValueError("logical_to_rank must be a 2-D integer array")
        if self.num_ranks <= 0 or self.num_experts % self.num_ranks != 0:
            raise ValueError("experts must divide evenly across ranks")
        if np.any(mapping < 0) or np.any(mapping >= self.num_ranks):
            raise ValueError(f"EP ranks must be in [0, {self.num_ranks - 1}]")
        expected = self.experts_per_rank
        for layer, row in enumerate(mapping):
            counts = np.bincount(row, minlength=self.num_ranks)
            if not np.all(counts == expected):
                raise ValueError(
                    f"layer {layer}: expected {expected} experts per rank, got {counts.tolist()}"
                )

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema_version": 1,
            "name": self.name,
            "num_layers": self.num_layers,
            "num_experts": self.num_experts,
            "num_ranks": self.num_ranks,
            "experts_per_rank": self.experts_per_rank,
            "logical_to_rank": self.logical_to_rank.astype(int).tolist(),
            "metadata": self.metadata,
        }


def placement_from_dict(value: dict[str, Any]) -> Placement:
    if value.get("schema_version") != 1:
        raise ValueError("unsupported placement schema_version")
    placement = Placement(
        name=str(value["name"]),
        logical_to_rank=np.asarray(value["logical_to_rank"], dtype=np.int64),
        num_ranks=int(value["num_ranks"]),
        metadata=dict(value.get("metadata") or {}),
    )
    placement.validate()
    if placement.num_layers != int(value["num_layers"]):
        raise ValueError("num_layers does not match logical_to_rank")
    if placement.num_experts != int(value["num_experts"]):
        raise ValueError("num_experts does not match logical_to_rank")
    if placement.experts_per_rank != int(value["experts_per_rank"]):
        raise ValueError("experts_per_rank does not match mapping")
    return placement


def load_placement(path: Path) -> Placement:
    return placement_from_dict(json.loads(path.read_text(encoding="utf-8")))


def save_placement(path: Path, placement: Placement) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(placement.to_dict(), indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")


def linear_placement(num_layers: int, num_experts: int = 128, num_ranks: int = 4) -> Placement:
    if num_experts % num_ranks:
        raise ValueError("experts must divide evenly across ranks")
    row = np.repeat(np.arange(num_ranks), num_experts // num_ranks)
    return Placement("default_linear", np.tile(row, (num_layers, 1)), num_ranks)


def physical_to_rank_placement(
    physical_to_logical: np.ndarray, *, name: str, num_ranks: int = 4,
    metadata: dict[str, Any] | None = None,
) -> Placement:
    physical = np.asarray(physical_to_logical)
    if physical.ndim != 2 or not np.issubdtype(physical.dtype, np.integer):
        raise ValueError("physical_to_logical must be a 2-D integer array")
    layers, experts = physical.shape
    if experts % num_ranks:
        raise ValueError("physical experts must divide evenly across ranks")
    logical_to_rank = np.full((layers, experts), -1, dtype=np.int64)
    slots = experts // num_ranks
    for layer, row in enumerate(physical):
        if sorted(row.tolist()) != list(range(experts)):
            raise ValueError(f"layer {layer}: physical map is not a permutation")
        for physical_id, logical_id in enumerate(row):
            logical_to_rank[layer, logical_id] = physical_id // slots
    result = Placement(name, logical_to_rank, num_ranks, metadata or {})
    result.validate()
    return result


def upstream_balanced_placement(
    counts: np.ndarray, *, num_groups: int, num_ranks: int = 4,
    policy: Callable[..., Any] | None = None,
) -> Placement:
    weights = validate_counts(counts)
    if policy is None:
        import torch
        from vllm.distributed.eplb.policy.default import DefaultEplbPolicy
        policy = DefaultEplbPolicy.rebalance_experts
        weight_argument: Any = torch.from_numpy(weights)
    else:
        weight_argument = weights
    physical, _, _ = policy(
        weight_argument, weights.shape[1], num_groups, 1, num_ranks,
    )
    if hasattr(physical, "cpu"):
        physical = physical.cpu().numpy()
    return physical_to_rank_placement(
        np.asarray(physical), name="upstream_default_eplb", num_ranks=num_ranks,
        metadata={"policy": "vllm.distributed.eplb.policy.DefaultEplbPolicy",
                  "num_groups": num_groups, "replication": False},
    )


def diagnostic_concentrated_placement(
    counts: np.ndarray, *, num_ranks: int = 4,
) -> Placement:
    weights = validate_counts(counts)
    layers, experts = weights.shape
    if experts % num_ranks:
        raise ValueError("experts must divide evenly across ranks")
    slots = experts // num_ranks
    mapping = np.empty((layers, experts), dtype=np.int64)
    for layer in range(layers):
        # Stable sorting makes ties deterministic by logical expert ID.
        descending = np.argsort(-weights[layer], kind="stable")
        for position, expert in enumerate(descending):
            mapping[layer, expert] = position // slots
    result = Placement(
        "diagnostic_load_concentrated", mapping, num_ranks,
        {"algorithm": "stable descending load; contiguous groups of 32 per rank",
         "replication": False},
    )
    result.validate()
    return result


def placement_loads(counts: np.ndarray, placement: Placement) -> np.ndarray:
    placement.validate()
    return aggregate_rank_loads(counts, placement.logical_to_rank)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-groups", type=int, default=8)
    args = parser.parse_args()
    with np.load(args.counts, allow_pickle=False) as archive:
        counts = validate_counts(archive["expert_counts"])
    placements = [
        linear_placement(counts.shape[0], counts.shape[1], 4),
        upstream_balanced_placement(counts, num_groups=args.num_groups),
        diagnostic_concentrated_placement(counts),
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for placement in placements:
        output = args.output_dir / f"{placement.name}.json"
        save_placement(output, placement)
        loads = placement_loads(counts, placement)
        summary.append({
            "placement": placement.name,
            "file": str(output),
            "per_gpu_load": loads.tolist(),
            "max_over_mean": float(loads.max() / loads.mean()) if loads.mean() else 0.0,
        })
    (args.output_dir / "placement_loads.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
