"""Profiling-only CUDA-event timing for the AFD FFN layer boundary.

The timed interval begins immediately before the FFN role invokes the native
Qwen3 MoE block and ends when that call returns its output to the caller's CUDA
stream. No routed-load statistic is calculated in this path. CUDA events are
resolved only when an explicit inactive control transition is observed.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from src.afd_routing_profile import EXPECTED_LAYERS, EXPECTED_RANKS, _load_control


TIMING_DIR_ENV = "AFD_FFN_TIMING_DIR"
TIMING_CLOCKS_ENV = "AFD_FFN_CLOCKS_MHZ"
TIMING_PHYSICAL_GPUS_ENV = "AFD_FFN_PHYSICAL_GPUS"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class RuntimeFFNTimingProfiler:
    """Retain rank-local events until an explicit profile-window boundary."""

    def __init__(
        self,
        directory: Path,
        rank: int,
        frequency_mhz: int,
        physical_gpu: int,
    ) -> None:
        if rank not in range(EXPECTED_RANKS):
            raise ValueError(f"invalid FFN rank {rank}")
        self.directory = directory.resolve()
        self.control_path = self.directory / "control.json"
        self.rank = rank
        self.frequency_mhz = int(frequency_mhz)
        self.physical_gpu = int(physical_gpu)
        self.generation: str | None = None
        self.active = False
        self.call_sequence = np.zeros(EXPECTED_LAYERS, dtype=np.int64)
        self.records: list[dict[str, Any]] = []
        self.dumped_generations: set[str] = set()

    def _reset(self) -> None:
        self.call_sequence.fill(0)
        self.records.clear()

    def _dump(self) -> Path:
        if self.generation is None:
            raise RuntimeError("cannot dump timing without a generation")
        if self.generation in self.dumped_generations:
            raise RuntimeError(f"timing generation already dumped: {self.generation}")
        import torch

        torch.cuda.synchronize()
        count = len(self.records)
        layers = np.empty(count, dtype=np.int16)
        sequences = np.empty(count, dtype=np.int32)
        token_rows = np.empty(count, dtype=np.int32)
        elapsed_ms = np.empty(count, dtype=np.float32)
        for index, record in enumerate(self.records):
            layers[index] = int(record["layer"])
            sequences[index] = int(record["call_sequence"])
            token_rows[index] = int(record["token_rows"])
            elapsed_ms[index] = float(
                record["start"].elapsed_time(record["end"])
            )
        self.directory.mkdir(parents=True, exist_ok=True)
        target = self.directory / (
            f"ffn-timing-rank-{self.rank}-pid-{os.getpid()}-"
            f"{self.generation}.npz"
        )
        temporary = target.with_name(target.name + ".tmp.npz")
        np.savez_compressed(
            temporary,
            layers=layers,
            call_sequences=sequences,
            token_rows=token_rows,
            elapsed_ms=elapsed_ms,
            ffn_rank=np.asarray(self.rank, dtype=np.int64),
            physical_gpu=np.asarray(self.physical_gpu, dtype=np.int64),
            frequency_mhz=np.asarray(self.frequency_mhz, dtype=np.int64),
            generation=np.asarray(self.generation),
            boundary=np.asarray(
                "AFDQwen3MoeDecoderLayer.compute_ffn_output: entry before "
                "native Qwen3MoeSparseMoeBlock through returned output"
            ),
        )
        os.replace(temporary, target)
        self.records.clear()
        self.dumped_generations.add(self.generation)
        return target

    def _refresh(self) -> None:
        if not self.control_path.is_file():
            return
        generation, requested_active = _load_control(self.control_path)
        if generation != self.generation:
            if self.active:
                self._dump()
            self.generation = generation
            self.active = requested_active
            if requested_active:
                self._reset()
            return
        if requested_active and not self.active:
            self._reset()
            self.active = True
        elif not requested_active and self.active:
            self._dump()
            self.active = False

    def invoke(
        self,
        layer_idx: int,
        hidden_states: Any,
        function: Callable[[Any], Any],
    ) -> Any:
        if layer_idx == 0:
            self._refresh()
        if not self.active:
            return function(hidden_states)
        import torch

        sequence = int(self.call_sequence[layer_idx])
        self.call_sequence[layer_idx] += 1
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = function(hidden_states)
        end.record()
        self.records.append({
            "layer": layer_idx,
            "call_sequence": sequence,
            "token_rows": int(hidden_states.shape[0]),
            "start": start,
            "end": end,
        })
        return result


_RUNTIME_TIMERS: dict[tuple[str, int], RuntimeFFNTimingProfiler] = {}


def _parse_rank_values(name: str, expected: int) -> list[int]:
    raw = os.environ.get(name, "").strip()
    values = [int(value) for value in raw.split(",") if value.strip()]
    if len(values) != expected:
        raise ValueError(f"{name} must contain {expected} comma-separated values")
    return values


def run_timed_ffn(
    layer_idx: int,
    hidden_states: Any,
    function: Callable[[Any], Any],
    ep_rank: int,
) -> Any:
    """Invoke an FFN block directly unless profiling was explicitly enabled."""
    raw_directory = os.environ.get(TIMING_DIR_ENV, "").strip()
    if not raw_directory:
        return function(hidden_states)
    directory = str(Path(raw_directory).expanduser().resolve())
    clocks = _parse_rank_values(TIMING_CLOCKS_ENV, EXPECTED_RANKS)
    physical_gpus = _parse_rank_values(TIMING_PHYSICAL_GPUS_ENV, EXPECTED_RANKS)
    key = (directory, int(ep_rank))
    timer = _RUNTIME_TIMERS.setdefault(
        key,
        RuntimeFFNTimingProfiler(
            Path(directory), int(ep_rank), clocks[int(ep_rank)],
            physical_gpus[int(ep_rank)],
        ),
    )
    return timer.invoke(layer_idx, hidden_states, function)


def wait_for_timing_profiles(
    directory: Path,
    generation: str,
    *,
    timeout_s: float = 120.0,
) -> list[Path]:
    deadline = time.monotonic() + timeout_s
    pattern = f"ffn-timing-rank-*-pid-*-{generation}.npz"
    while time.monotonic() < deadline:
        paths = sorted(directory.glob(pattern))
        if len(paths) == EXPECTED_RANKS:
            return paths
        if len(paths) > EXPECTED_RANKS:
            raise ValueError(f"found too many FFN timing profiles: {paths}")
        time.sleep(0.25)
    raise TimeoutError(f"expected {EXPECTED_RANKS} FFN timing profiles")


def finalize_timing_profiles(
    directory: Path,
    generation: str,
    output_directory: Path,
    *,
    timeout_s: float = 120.0,
) -> dict[str, Any]:
    paths = wait_for_timing_profiles(directory, generation, timeout_s=timeout_s)
    output_directory.mkdir(parents=True, exist_ok=True)
    ranks: list[dict[str, Any]] = []
    seen: set[int] = set()
    for path in paths:
        with np.load(path, allow_pickle=False) as archive:
            rank = int(archive["ffn_rank"])
            saved_generation = str(archive["generation"])
            layers = np.asarray(archive["layers"])
            sequences = np.asarray(archive["call_sequences"])
            token_rows = np.asarray(archive["token_rows"])
            elapsed = np.asarray(archive["elapsed_ms"])
            frequency = int(archive["frequency_mhz"])
            physical_gpu = int(archive["physical_gpu"])
            boundary = str(archive["boundary"])
        if saved_generation != generation or rank in seen:
            raise ValueError(f"{path}: duplicate rank or generation mismatch")
        if not (layers.shape == sequences.shape == token_rows.shape == elapsed.shape):
            raise ValueError(f"{path}: inconsistent timing array shapes")
        if np.any(elapsed <= 0) or np.any(token_rows <= 0):
            raise ValueError(f"{path}: non-positive timing data")
        keys = np.stack([sequences, layers], axis=1)
        if len(np.unique(keys, axis=0)) != len(keys):
            raise ValueError(f"{path}: duplicate layer/call timing keys")
        copied = output_directory / path.name
        copied.write_bytes(path.read_bytes())
        ranks.append({
            "rank": rank,
            "physical_gpu": physical_gpu,
            "frequency_mhz": frequency,
            "records": int(len(layers)),
            "path": str(copied.resolve()),
            "sha256": _sha256(copied),
        })
        seen.add(rank)
    if seen != set(range(EXPECTED_RANKS)):
        raise ValueError(f"missing FFN timing rank: {seen}")
    metadata = {
        "schema_version": 1,
        "generation": generation,
        "timed_boundary": boundary,
        "cuda_event_resolution": "one torch.cuda.synchronize at explicit dump",
        "routing_statistics_inside_timed_region": False,
        "ranks": sorted(ranks, key=lambda row: row["rank"]),
    }
    path = output_directory / "ffn_timing.json"
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return metadata


def load_timing_arrays(paths: Sequence[Path]) -> dict[int, dict[str, np.ndarray]]:
    result: dict[int, dict[str, np.ndarray]] = {}
    for path in paths:
        with np.load(path, allow_pickle=False) as archive:
            rank = int(archive["ffn_rank"])
            result[rank] = {
                key: np.asarray(archive[key])
                for key in ("layers", "call_sequences", "token_rows", "elapsed_ms")
            }
    return result
