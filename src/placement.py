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


def controlled_load_placements(
    counts: np.ndarray,
    *,
    base: Placement | None = None,
    concentrated: Placement | None = None,
    num_ranks: int = 4,
) -> list[Placement]:
    """Create four deterministic GPU0-load levels from two valid endpoints.

    The base endpoint is normally upstream EPLB and the concentrated endpoint
    is the diagnostic descending-load mapping.  For each layer, a fraction of
    the highest-load experts that move onto rank 0 in the concentrated mapping
    are swapped with the lowest-load rank-0 experts from the base mapping.
    Each swap preserves every rank's cardinality, while monotonically moving
    the controlled load axis toward the concentrated endpoint.
    """
    weights = validate_counts(counts)
    layers, experts = weights.shape
    if base is None:
        base = upstream_balanced_placement(weights, num_groups=8, num_ranks=num_ranks)
    if concentrated is None:
        concentrated = diagnostic_concentrated_placement(weights, num_ranks=num_ranks)
    base.validate()
    concentrated.validate()
    if (base.num_layers, base.num_experts, base.num_ranks) != (
        layers, experts, num_ranks
    ) or (concentrated.num_layers, concentrated.num_experts,
          concentrated.num_ranks) != (layers, experts, num_ranks):
        raise ValueError("controlled placement endpoints do not match counts")
    if num_ranks < 2:
        raise ValueError("controlled load levels require at least two ranks")

    levels = (
        ("LOW", 0.0),
        ("MEDIUM", 1.0 / 3.0),
        ("HIGH", 2.0 / 3.0),
        ("VERY_HIGH", 1.0),
    )
    placements: list[Placement] = []
    for name, alpha in levels:
        mapping = base.logical_to_rank.copy()
        swaps = 0
        for layer in range(layers):
            base_row = base.logical_to_rank[layer]
            target_row = concentrated.logical_to_rank[layer]
            incoming = [
                int(expert_id) for expert_id in range(experts)
                if base_row[expert_id] != 0 and target_row[expert_id] == 0
            ]
            outgoing = [
                int(expert_id) for expert_id in range(experts)
                if base_row[expert_id] == 0 and target_row[expert_id] != 0
            ]
            incoming.sort(key=lambda expert_id: (-int(weights[layer, expert_id]), expert_id))
            outgoing.sort(key=lambda expert_id: (int(weights[layer, expert_id]), expert_id))
            count = int(round(alpha * min(len(incoming), len(outgoing))))
            for incoming_id, outgoing_id in zip(incoming[:count], outgoing[:count]):
                source_rank = int(mapping[layer, incoming_id])
                if source_rank == 0 or int(mapping[layer, outgoing_id]) != 0:
                    raise ValueError("controlled placement swap lost rank-0 invariant")
                mapping[layer, incoming_id] = 0
                mapping[layer, outgoing_id] = source_rank
                swaps += 1
        placement = Placement(
            f"controlled_{name.lower()}",
            mapping,
            num_ranks,
            {
                "algorithm": "rank-0 interpolation between upstream EPLB and descending-load endpoints",
                "base_placement": base.name,
                "concentrated_placement": concentrated.name,
                "alpha": alpha,
                "swaps": swaps,
                "replication": False,
            },
        )
        placement.validate()
        placements.append(placement)
    return placements


def frequency_aware_placement(
    counts: np.ndarray, frequencies: list[int] | tuple[int, ...], *,
    speed_factors: list[float] | tuple[float, ...] | None = None,
    num_ranks: int = 4, max_swap_passes: int = 4,
    reference_frequency_mhz: int = 2520,
) -> Placement:
    """Balance frozen expert load against heterogeneous capacity per layer.

    The speed proxy defaults to ``frequency / 2520``. It is a
    pre-registered placement heuristic, not a performance-scaling model.
    """
    weights = validate_counts(counts)
    clocks = tuple(int(value) for value in frequencies)
    if len(clocks) != num_ranks or any(value <= 0 for value in clocks):
        raise ValueError("one positive frequency is required per rank")
    if speed_factors is None:
        if reference_frequency_mhz <= 0:
            raise ValueError("reference_frequency_mhz must be positive")
        speeds = tuple(value / reference_frequency_mhz for value in clocks)
    else:
        speeds = tuple(float(value) for value in speed_factors)
    if len(speeds) != num_ranks or any(value <= 0 for value in speeds):
        raise ValueError("one positive speed factor is required per rank")
    if max_swap_passes < 0:
        raise ValueError("max_swap_passes must be non-negative")
    layers, experts = weights.shape
    if experts % num_ranks:
        raise ValueError("experts must divide evenly across ranks")
    slots = experts // num_ranks
    mapping = np.full((layers, experts), -1, dtype=np.int64)
    before_objectives: list[float] = []
    after_objectives: list[float] = []
    total_swaps = 0
    tolerance = 1e-12

    for layer in range(layers):
        loads = np.zeros(num_ranks, dtype=np.float64)
        assigned: list[list[int]] = [[] for _ in range(num_ranks)]
        order = np.argsort(-weights[layer], kind="stable")
        for expert in order:
            eligible = [rank for rank in range(num_ranks) if len(assigned[rank]) < slots]
            rank = min(eligible, key=lambda value: (loads[value] / speeds[value], value))
            mapping[layer, expert] = rank
            assigned[rank].append(int(expert))
            loads[rank] += float(weights[layer, expert])
        before_objectives.append(float(max(loads[rank] / speeds[rank]
                                           for rank in range(num_ranks))))

        for _ in range(max_swap_passes):
            changed = False
            current = max(loads[rank] / speeds[rank] for rank in range(num_ranks))
            for left in range(num_ranks):
                for right in range(left + 1, num_ranks):
                    for left_expert in sorted(assigned[left]):
                        for right_expert in sorted(assigned[right]):
                            left_weight = float(weights[layer, left_expert])
                            right_weight = float(weights[layer, right_expert])
                            candidate_loads = loads.copy()
                            candidate_loads[left] += right_weight - left_weight
                            candidate_loads[right] += left_weight - right_weight
                            objective = max(candidate_loads[rank] / speeds[rank]
                                            for rank in range(num_ranks))
                            if objective < current - tolerance:
                                loads = candidate_loads
                                assigned[left].remove(left_expert)
                                assigned[left].append(right_expert)
                                assigned[right].remove(right_expert)
                                assigned[right].append(left_expert)
                                mapping[layer, left_expert] = right
                                mapping[layer, right_expert] = left
                                total_swaps += 1
                                changed = True
                                current = objective
                                break
                        if changed:
                            break
                    if changed:
                        break
                if changed:
                    break
            if not changed:
                break
        after_objectives.append(float(max(loads[rank] / speeds[rank]
                                          for rank in range(num_ranks))))

    result = Placement(
        "frequency_aware", mapping, num_ranks,
        {"algorithm": "capacity-normalized constrained LPT plus first-improvement swaps",
         "frequencies_mhz": list(clocks), "speed_factors": list(speeds),
         "reference_frequency_mhz": reference_frequency_mhz,
         "max_swap_passes": max_swap_passes, "local_swaps": total_swaps,
         "per_layer_before_objective": before_objectives,
         "per_layer_after_objective": after_objectives,
         "before_objective": max(before_objectives, default=0.0),
         "after_objective": max(after_objectives, default=0.0),
         "replication": False},
    )
    result.validate()
    raw_loads = placement_loads(weights, result).astype(float)
    result.metadata["per_rank_raw_load"] = raw_loads.tolist()
    result.metadata["per_rank_normalized_load"] = [
        float(raw_loads[rank] / speeds[rank]) for rank in range(num_ranks)
    ]
    normalized = np.asarray(result.metadata["per_rank_normalized_load"])
    result.metadata["normalized_max_over_mean"] = (
        float(normalized.max() / normalized.mean()) if normalized.mean() else 0.0
    )
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
