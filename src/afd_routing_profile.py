"""Profiling-only logical expert counts for the isolated AFD FFN workers."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np


PROFILE_DIR_ENV = "AFD_ROUTING_PROFILE_DIR"
EXPECTED_LAYERS = 48
EXPECTED_EXPERTS = 128
EXPECTED_RANKS = 2


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_profile_control(directory: Path, generation: str, active: bool) -> Path:
    """Atomically publish one profiling generation's desired state."""
    if not generation or any(character.isspace() for character in generation):
        raise ValueError("profile generation must be a non-empty token")
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "control.json"
    temporary = directory / f"control.{os.getpid()}.tmp"
    temporary.write_text(json.dumps({
        "schema_version": 1,
        "generation": generation,
        "active": bool(active),
        "updated_unix_s": time.time(),
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, target)
    return target


def _load_control(path: Path) -> tuple[str, bool]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != 1:
        raise ValueError("unsupported AFD routing profile control schema")
    generation = str(value.get("generation", ""))
    if not generation:
        raise ValueError("routing profile control has no generation")
    active = value.get("active")
    if not isinstance(active, bool):
        raise ValueError("routing profile active state must be boolean")
    return generation, active


class RuntimeRoutingProfiler:
    """Accumulate logical top-k IDs on the FFN worker's CUDA device.

    Only layer 0 polls the shared control file. A transition to active resets
    the device counters; a transition to inactive dumps them before the flush
    request is counted. This is intentionally profiling-only instrumentation.
    """

    def __init__(self, directory: Path, ep_rank: int) -> None:
        if ep_rank not in range(EXPECTED_RANKS):
            raise ValueError(f"invalid FFN rank {ep_rank}")
        self.directory = directory.resolve()
        self.control_path = self.directory / "control.json"
        self.ep_rank = ep_rank
        self.generation: str | None = None
        self.active = False
        self.counts: Any | None = None
        self.capture_calls = np.zeros(EXPECTED_LAYERS, dtype=np.int64)
        self.valid_assignments = np.zeros(EXPECTED_LAYERS, dtype=np.int64)
        self.dumped_generations: set[str] = set()

    def _reset(self, topk_ids: Any) -> None:
        import torch

        self.counts = torch.zeros(
            (EXPECTED_LAYERS, EXPECTED_EXPERTS),
            dtype=torch.int64,
            device=topk_ids.device,
        )
        self.capture_calls.fill(0)
        self.valid_assignments.fill(0)

    def _dump(self) -> Path:
        if self.generation is None or self.counts is None:
            raise RuntimeError("cannot dump an uninitialized routing profile")
        if self.generation in self.dumped_generations:
            raise RuntimeError(f"profile generation already dumped: {self.generation}")
        self.directory.mkdir(parents=True, exist_ok=True)
        counts = self.counts.detach().cpu().numpy().astype(np.int64, copy=False)
        target = self.directory / (
            f"routing-counts-rank-{self.ep_rank}-pid-{os.getpid()}-"
            f"{self.generation}.npz"
        )
        temporary = target.with_name(target.name + ".tmp.npz")
        np.savez_compressed(
            temporary,
            expert_counts=counts,
            ep_rank=np.asarray(self.ep_rank, dtype=np.int64),
            generation=np.asarray(self.generation),
            capture_calls=self.capture_calls,
            valid_assignments=self.valid_assignments,
        )
        os.replace(temporary, target)
        self.dumped_generations.add(self.generation)
        return target

    def _refresh(self, topk_ids: Any) -> None:
        if not self.control_path.is_file():
            return
        generation, requested_active = _load_control(self.control_path)
        if generation != self.generation:
            if self.active and self.counts is not None:
                self._dump()
            self.generation = generation
            self.active = requested_active
            if requested_active:
                self._reset(topk_ids)
            return
        if requested_active and not self.active:
            self._reset(topk_ids)
            self.active = True
        elif not requested_active and self.active:
            self._dump()
            self.active = False

    def capture(self, layer_idx: int, topk_ids: Any) -> None:
        if not 0 <= layer_idx < EXPECTED_LAYERS:
            raise ValueError(f"invalid routing profile layer {layer_idx}")
        if layer_idx == 0:
            self._refresh(topk_ids)
        if not self.active:
            return
        if self.counts is None:
            self._reset(topk_ids)
        import torch

        logical_ids = topk_ids.reshape(-1).to(dtype=torch.int64)
        logical_ids = logical_ids[
            (logical_ids >= 0) & (logical_ids < EXPECTED_EXPERTS)
        ]
        self.counts[layer_idx].add_(
            torch.bincount(logical_ids, minlength=EXPECTED_EXPERTS)
        )
        self.capture_calls[layer_idx] += 1
        self.valid_assignments[layer_idx] += int(logical_ids.numel())


_RUNTIME_PROFILERS: dict[tuple[str, int], RuntimeRoutingProfiler] = {}


def install_runtime_routing_profiler(moe_block: Any, layer_idx: int) -> bool:
    """Bind one native vLLM 0.26 logical-router capture callback."""
    raw_directory = os.environ.get(PROFILE_DIR_ENV, "").strip()
    if not raw_directory:
        return False
    if int(moe_block.ep_size) != EXPECTED_RANKS:
        raise ValueError(
            f"AFD routing profile requires EP={EXPECTED_RANKS}, got {moe_block.ep_size}"
        )
    directory = str(Path(raw_directory).expanduser().resolve())
    rank = int(moe_block.ep_rank)
    key = (directory, rank)
    profiler = _RUNTIME_PROFILERS.setdefault(
        key, RuntimeRoutingProfiler(Path(directory), rank)
    )
    moe_block.experts.router.set_capture_fn(
        lambda topk_ids, _layer=layer_idx, _profiler=profiler: _profiler.capture(
            _layer, topk_ids
        )
    )
    return True


def wait_for_rank_profiles(
    directory: Path,
    generation: str,
    *,
    expected_ranks: int = EXPECTED_RANKS,
    timeout_s: float = 120.0,
) -> list[Path]:
    deadline = time.monotonic() + timeout_s
    pattern = f"routing-counts-rank-*-pid-*-{generation}.npz"
    while time.monotonic() < deadline:
        paths = sorted(directory.glob(pattern))
        if len(paths) == expected_ranks:
            return paths
        if len(paths) > expected_ranks:
            raise ValueError(f"found too many routing profiles: {paths}")
        time.sleep(0.25)
    raise TimeoutError(
        f"expected {expected_ranks} rank profiles for {generation} in {directory}"
    )


def combine_rank_profiles(
    paths: Sequence[Path],
    generation: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if len(paths) != EXPECTED_RANKS:
        raise ValueError(f"expected {EXPECTED_RANKS} rank profiles, got {len(paths)}")
    per_rank = np.zeros(
        (EXPECTED_RANKS, EXPECTED_LAYERS, EXPECTED_EXPERTS), dtype=np.int64
    )
    seen: set[int] = set()
    rank_metadata: list[dict[str, Any]] = []
    for path in paths:
        with np.load(path, allow_pickle=False) as archive:
            rank = int(archive["ep_rank"])
            saved_generation = str(archive["generation"])
            counts = np.asarray(archive["expert_counts"], dtype=np.int64)
            capture_calls = np.asarray(archive["capture_calls"], dtype=np.int64)
            valid_assignments = np.asarray(
                archive["valid_assignments"], dtype=np.int64
            )
        if saved_generation != generation:
            raise ValueError(f"{path}: profile generation mismatch")
        if rank in seen or rank not in range(EXPECTED_RANKS):
            raise ValueError(f"invalid or duplicate rank {rank}")
        if counts.shape != (EXPECTED_LAYERS, EXPECTED_EXPERTS):
            raise ValueError(f"{path}: invalid counts shape {counts.shape}")
        if np.any(counts < 0) or capture_calls.shape != (EXPECTED_LAYERS,):
            raise ValueError(f"{path}: invalid routing profile metadata")
        if valid_assignments.shape != (EXPECTED_LAYERS,):
            raise ValueError(f"{path}: invalid assignment metadata")
        if not np.array_equal(counts.sum(axis=1), valid_assignments):
            raise ValueError(f"{path}: counts do not match valid assignments")
        seen.add(rank)
        per_rank[rank] = counts
        rank_metadata.append({
            "rank": rank,
            "path": str(path.resolve()),
            "sha256": _sha256(path),
            "capture_calls": capture_calls.tolist(),
            "valid_assignments": valid_assignments.tolist(),
        })
    if seen != set(range(EXPECTED_RANKS)):
        raise ValueError(f"missing rank profiles: {seen}")
    combined = per_rank.sum(axis=0)
    metadata = {
        "schema_version": 1,
        "generation": generation,
        "aggregation": "sum of disjoint 2A2F FFN/DP-rank logical routes",
        "shape": list(combined.shape),
        "rank_profiles": sorted(rank_metadata, key=lambda row: row["rank"]),
    }
    return combined, per_rank, metadata


def finalize_rank_profiles(
    directory: Path,
    generation: str,
    output_directory: Path,
    *,
    timeout_s: float = 120.0,
) -> dict[str, Any]:
    paths = wait_for_rank_profiles(directory, generation, timeout_s=timeout_s)
    combined, per_rank, metadata = combine_rank_profiles(paths, generation)
    output_directory.mkdir(parents=True, exist_ok=True)
    copied_paths = []
    for path in paths:
        copied = output_directory / path.name
        shutil.copy2(path, copied)
        copied_paths.append(copied)
    for row, copied in zip(metadata["rank_profiles"], copied_paths):
        row["path"] = str(copied.resolve())
    counts_path = output_directory / "expert_counts.npz"
    np.savez_compressed(
        counts_path,
        expert_counts=combined,
        per_rank_expert_counts=per_rank,
    )
    metadata["expert_counts"] = str(counts_path.resolve())
    metadata["expert_counts_sha256"] = _sha256(counts_path)
    metadata_path = output_directory / "routing_profile.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata
