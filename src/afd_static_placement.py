"""Static per-layer expert ownership for the two-rank AFD FFN pool."""

from __future__ import annotations

import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Sequence


PLACEMENT_ENV = "AFD_STATIC_EXPERT_PLACEMENT_FILE"
EVIDENCE_ENV = "AFD_STATIC_EXPERT_PLACEMENT_EVIDENCE_DIR"
EXPECTED_LAYERS = 48
EXPECTED_EXPERTS = 128
EXPECTED_RANKS = 2


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_static_mapping(
    mapping: Sequence[Sequence[int]],
    *,
    num_layers: int = EXPECTED_LAYERS,
    num_experts: int = EXPECTED_EXPERTS,
    num_ranks: int = EXPECTED_RANKS,
) -> tuple[tuple[int, ...], ...]:
    if len(mapping) != num_layers:
        raise ValueError(f"expected {num_layers} layers, got {len(mapping)}")
    if num_ranks <= 0 or num_experts % num_ranks:
        raise ValueError("experts must divide evenly across ranks")
    expected_per_rank = num_experts // num_ranks
    normalized: list[tuple[int, ...]] = []
    for layer_idx, row in enumerate(mapping):
        if len(row) != num_experts:
            raise ValueError(
                f"layer {layer_idx}: expected {num_experts} experts, got {len(row)}"
            )
        if any(isinstance(rank, bool) or not isinstance(rank, int) for rank in row):
            raise ValueError(f"layer {layer_idx}: rank IDs must be integers")
        if any(rank < 0 or rank >= num_ranks for rank in row):
            raise ValueError(
                f"layer {layer_idx}: rank IDs must be in [0, {num_ranks - 1}]"
            )
        counts = [sum(rank == candidate for rank in row) for candidate in range(num_ranks)]
        if counts != [expected_per_rank] * num_ranks:
            raise ValueError(
                f"layer {layer_idx}: expected {expected_per_rank} experts/rank, "
                f"got {counts}"
            )
        normalized.append(tuple(row))
    return tuple(normalized)


def placement_document(
    name: str,
    mapping: Sequence[Sequence[int]],
    *,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    normalized = validate_static_mapping(mapping)
    return {
        "schema_version": 1,
        "name": name,
        "num_layers": EXPECTED_LAYERS,
        "num_experts": EXPECTED_EXPERTS,
        "num_ranks": EXPECTED_RANKS,
        "experts_per_rank": EXPECTED_EXPERTS // EXPECTED_RANKS,
        "logical_to_rank": [list(row) for row in normalized],
        "metadata": metadata or {},
    }


def write_placement(path: Path, document: dict[str, Any]) -> str:
    load_placement_document(document)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    return sha256_file(path)


def load_placement_document(value: dict[str, Any]) -> tuple[tuple[int, ...], ...]:
    if value.get("schema_version") != 1:
        raise ValueError("unsupported placement schema_version")
    declared = (
        value.get("num_layers"),
        value.get("num_experts"),
        value.get("num_ranks"),
        value.get("experts_per_rank"),
    )
    expected = (
        EXPECTED_LAYERS,
        EXPECTED_EXPERTS,
        EXPECTED_RANKS,
        EXPECTED_EXPERTS // EXPECTED_RANKS,
    )
    if declared != expected:
        raise ValueError(f"placement dimensions {declared} do not match {expected}")
    mapping = value.get("logical_to_rank")
    if not isinstance(mapping, list):
        raise ValueError("logical_to_rank must be a list")
    return validate_static_mapping(mapping)


@lru_cache(maxsize=None)
def load_placement(path: str) -> tuple[tuple[tuple[int, ...], ...], str, str]:
    placement_path = Path(path).expanduser().resolve(strict=True)
    value = json.loads(placement_path.read_text())
    mapping = load_placement_document(value)
    return mapping, sha256_file(placement_path), str(value.get("name", "unnamed"))


def balanced_round_robin_mapping() -> tuple[tuple[int, ...], ...]:
    return validate_static_mapping(
        [[(expert + layer) % EXPECTED_RANKS for expert in range(EXPECTED_EXPERTS)]
         for layer in range(EXPECTED_LAYERS)]
    )


def different_block_mapping() -> tuple[tuple[int, ...], ...]:
    return validate_static_mapping(
        [[((expert // 16) + layer) % EXPECTED_RANKS
          for expert in range(EXPECTED_EXPERTS)]
         for layer in range(EXPECTED_LAYERS)]
    )


def linear_default_mapping() -> tuple[tuple[int, ...], ...]:
    return validate_static_mapping(
        [[expert // (EXPECTED_EXPERTS // EXPECTED_RANKS)
          for expert in range(EXPECTED_EXPERTS)]
         for _ in range(EXPECTED_LAYERS)]
    )


def global_to_local_map(owner_row: Sequence[int], ep_rank: int) -> tuple[int, ...]:
    if ep_rank not in range(EXPECTED_RANKS):
        raise ValueError(f"invalid FFN EP rank {ep_rank}")
    local_ids = [-1] * len(owner_row)
    next_local = 0
    for logical_expert, owner in enumerate(owner_row):
        if owner == ep_rank:
            local_ids[logical_expert] = next_local
            next_local += 1
    expected = EXPECTED_EXPERTS // EXPECTED_RANKS
    if next_local != expected:
        raise ValueError(f"rank {ep_rank}: expected {expected} experts, got {next_local}")
    return tuple(local_ids)


def apply_runtime_placement(moe_block: Any, layer_idx: int) -> bool:
    """Install one layer's map into vLLM's live ExpertMapManager.

    RoutedExperts uses this same manager for checkpoint loading and kernel
    routing, so the mapping is installed before model weights are loaded.
    """
    raw_path = os.environ.get(PLACEMENT_ENV, "").strip()
    if not raw_path:
        return False
    mapping, _mapping_sha, _name = load_placement(raw_path)
    if not 0 <= layer_idx < len(mapping):
        raise ValueError(f"invalid MoE layer index {layer_idx}")
    if moe_block.ep_size != EXPECTED_RANKS:
        raise ValueError(
            f"static AFD placement requires EP={EXPECTED_RANKS}, got {moe_block.ep_size}"
        )

    import torch

    runner = moe_block.experts
    manager = runner.expert_map_manager
    if manager.global_num_experts != EXPECTED_EXPERTS:
        raise ValueError(
            f"expected {EXPECTED_EXPERTS} logical experts, "
            f"got {manager.global_num_experts}"
        )
    local_map = global_to_local_map(mapping[layer_idx], moe_block.ep_rank)
    old_map = manager.expert_map
    if old_map is None:
        raise ValueError("custom placement requires expert parallelism")
    manager._local_num_experts = EXPECTED_EXPERTS // EXPECTED_RANKS
    manager._expert_map = torch.tensor(
        local_map, dtype=torch.int32, device=old_map.device
    )
    manager._expert_mask = None
    manager._routing_tables = None
    runner.routed_experts.update_expert_map_info()
    actual = tuple(int(value) for value in runner.expert_map.cpu().tolist())
    if actual != local_map:
        raise RuntimeError(f"layer {layer_idx}: live expert map does not match request")
    return True


def _ownership_from_live_map(values: Iterable[int]) -> list[int]:
    return [index for index, local_id in enumerate(values) if int(local_id) >= 0]


def dump_runtime_ownership(model_layers: Iterable[Any]) -> Path | None:
    """Dump live post-construction ownership after checkpoint loading."""
    raw_dir = os.environ.get(EVIDENCE_ENV, "").strip()
    if not raw_dir:
        return None
    evidence_dir = Path(raw_dir).expanduser().resolve()
    evidence_dir.mkdir(parents=True, exist_ok=True)
    layer_records: list[dict[str, Any]] = []
    ep_rank: int | None = None
    for layer_idx, layer in enumerate(model_layers):
        moe_block = getattr(layer, "mlp", None)
        runner = getattr(moe_block, "experts", None)
        expert_map = getattr(runner, "expert_map", None)
        if expert_map is None:
            continue
        current_rank = int(moe_block.ep_rank)
        if ep_rank is None:
            ep_rank = current_rank
        elif ep_rank != current_rank:
            raise RuntimeError("one worker exposed inconsistent EP ranks")
        live_map = [int(value) for value in expert_map.detach().cpu().tolist()]
        owned = _ownership_from_live_map(live_map)
        if len(owned) != EXPECTED_EXPERTS // EXPECTED_RANKS:
            raise RuntimeError(
                f"layer {layer_idx}: runtime owns {len(owned)} experts"
            )
        layer_records.append({
            "layer": layer_idx,
            "global_to_local": live_map,
            "owned_logical_experts": owned,
        })
    if ep_rank is None or len(layer_records) != EXPECTED_LAYERS:
        raise RuntimeError(
            f"expected {EXPECTED_LAYERS} live MoE layers, got {len(layer_records)}"
        )
    placement_path = os.environ.get(PLACEMENT_ENV, "").strip()
    source_sha = sha256_file(Path(placement_path).resolve()) if placement_path else None
    record = {
        "schema_version": 1,
        "ep_rank": ep_rank,
        "pid": os.getpid(),
        "placement_mode": "custom" if placement_path else "default",
        "placement_file": str(Path(placement_path).resolve()) if placement_path else None,
        "placement_sha256": source_sha,
        "layers": layer_records,
    }
    output = evidence_dir / f"ownership-ep-rank-{ep_rank}-pid-{os.getpid()}.json"
    output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return output


def dump_model_runtime_ownership(model: Any) -> Path | None:
    """Avoid touching model internals unless runtime evidence was requested."""
    if not os.environ.get(EVIDENCE_ENV, "").strip():
        return None
    return dump_runtime_ownership(model.model.layers)


def combine_runtime_evidence(
    evidence_files: Sequence[Path],
    expected_mapping: Sequence[Sequence[int]],
) -> dict[str, Any]:
    mapping = validate_static_mapping(expected_mapping)
    if len(evidence_files) != EXPECTED_RANKS:
        raise ValueError(
            f"expected {EXPECTED_RANKS} ownership files, got {len(evidence_files)}"
        )
    records = [json.loads(path.read_text()) for path in evidence_files]
    ranks = [int(record["ep_rank"]) for record in records]
    if sorted(ranks) != list(range(EXPECTED_RANKS)):
        raise ValueError(f"expected one file per EP rank, got {ranks}")
    by_rank = {int(record["ep_rank"]): record for record in records}
    actual = [[-1] * EXPECTED_EXPERTS for _ in range(EXPECTED_LAYERS)]
    for rank, record in by_rank.items():
        layers = record.get("layers") or []
        if len(layers) != EXPECTED_LAYERS:
            raise ValueError(f"rank {rank}: expected {EXPECTED_LAYERS} layers")
        for layer_idx, layer_record in enumerate(layers):
            if int(layer_record["layer"]) != layer_idx:
                raise ValueError(f"rank {rank}: non-canonical layer order")
            live_map = layer_record["global_to_local"]
            if len(live_map) != EXPECTED_EXPERTS:
                raise ValueError(f"rank {rank} layer {layer_idx}: invalid map length")
            local_ids = sorted(int(value) for value in live_map if int(value) >= 0)
            if local_ids != list(range(EXPECTED_EXPERTS // EXPECTED_RANKS)):
                raise ValueError(
                    f"rank {rank} layer {layer_idx}: local slots are not 0..63"
                )
            for expert, local_id in enumerate(live_map):
                if int(local_id) >= 0:
                    if actual[layer_idx][expert] != -1:
                        raise ValueError(
                            f"layer {layer_idx} expert {expert}: duplicate ownership"
                        )
                    actual[layer_idx][expert] = rank
    normalized_actual = validate_static_mapping(actual)
    if normalized_actual != mapping:
        raise ValueError("runtime ownership does not match requested placement")
    canonical = json.dumps(actual, separators=(",", ":")).encode()
    return {
        "schema_version": 1,
        "status": "pass",
        "evidence_files": [str(path) for path in evidence_files],
        "actual_logical_to_rank": actual,
        "actual_mapping_sha256": hashlib.sha256(canonical).hexdigest(),
    }
