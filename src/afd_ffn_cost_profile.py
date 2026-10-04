"""Same-run AFD routing and local/full FFN timing instrumentation.

The FFN CUDA graph bypasses Python model hooks during replay.  This module
therefore registers CUDA events and a routing snapshot while each graph is
captured, then resolves those graph-owned objects at the beginning of the next
FFN service step.  The AFD worker already synchronizes CUDA after every step;
the resolver does not introduce another per-call synchronization.

The local interval wraps ``FusedMoEKernelModularImpl._fused_experts``.  It
includes local workspace preparation, assignment/permutation, and expert
kernels, but excludes EP dispatch and combine.
The full interval wraps the complete per-layer Qwen MoE block.  A copy of the
already-computed logical ``topk_ids`` is enqueued after the full interval, so
the routing feature is from the same invocation without a GPU bincount inside
either measured interval.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from src.afd_routing_profile import (
    EXPECTED_EXPERTS,
    EXPECTED_LAYERS,
    EXPECTED_RANKS,
    _load_control,
)


PROFILE_DIR_ENV = "AFD_FFN_TIMING_DIR"
PROFILE_CLOCKS_ENV = "AFD_FFN_CLOCKS_MHZ"
PROFILE_PHYSICAL_GPUS_ENV = "AFD_FFN_PHYSICAL_GPUS"
CHUNK_RECORDS = 1024
EXPECTED_TOP_K = 8
LOCAL_WITHIN_FULL_MIN_FRACTION = 0.99
TIMING_CONTAINMENT_TOLERANCE_MS = 0.01


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _graph_key_text(graph_key: object) -> str:
    return repr(graph_key)


def _cuda_timing_event(*, graph_capture: bool) -> Any:
    import torch

    # Internal events become graph-only cross-stream dependencies and do not
    # retain timestamps that elapsed_time() can query after replay.
    return torch.cuda.Event(enable_timing=True, external=graph_capture)


def _stage_index() -> int:
    from vllm.forward_context import get_forward_context

    context = get_forward_context()
    metadata = context.additional_kwargs.get("afd_metadata")
    if metadata is None:
        raise RuntimeError("AFD cost profiling requires afd_metadata")
    return int(metadata.stage_idx)


def valid_routing_rows(topk_rows: int, rank: int, dp_counts: Any | None) -> int:
    """Return real source rows, excluding fixed graph-buffer padding."""
    valid_rows = int(topk_rows)
    if dp_counts is not None:
        value = dp_counts[int(rank)]
        valid_rows = int(value.item() if hasattr(value, "item") else value)
    if not 0 < valid_rows <= int(topk_rows):
        raise ValueError(f"invalid routed row count {valid_rows}/{topk_rows}")
    return valid_rows


@dataclass
class _Invocation:
    layer: int
    stage: int
    token_rows: int
    full_start: Any
    full_end: Any | None = None
    local_start: Any | None = None
    local_end: Any | None = None
    topk_ids: Any | None = None
    routing_snapshot: Any | None = None


class RuntimeAFDCostProfiler:
    """One profiler instance per FFN worker process."""

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
        self.rank = int(rank)
        self.frequency_mhz = int(frequency_mhz)
        self.physical_gpu = int(physical_gpu)
        self.generation: str | None = None
        self.active = False
        self.dumped_generations: set[str] = set()
        self.step_sequence = 0
        self.mode = "inactive"
        self.capture_key: str | None = None
        self.capture_records: list[_Invocation] = []
        self.graph_templates: dict[str, tuple[_Invocation, ...]] = {}
        self.current_invocations: list[_Invocation] = []
        self.invocation_stack: list[_Invocation] = []
        self.pending_records: tuple[_Invocation, ...] | None = None
        self.pending_step: int | None = None
        self.rows: list[dict[str, Any]] = []
        self.histograms: list[np.ndarray] = []
        self.chunk_paths: list[Path] = []
        self.next_chunk = 0

    def begin_graph_capture(self, graph_key: object) -> None:
        if self.capture_key is not None or self.pending_records is not None:
            raise RuntimeError("nested or overlapping AFD graph profiling")
        self.mode = "capture"
        self.capture_key = _graph_key_text(graph_key)
        self.capture_records = []
        self.invocation_stack = []

    def end_graph_capture(self, graph_key: object) -> None:
        key = _graph_key_text(graph_key)
        if self.capture_key != key or self.invocation_stack:
            raise RuntimeError("AFD graph profile capture state mismatch")
        records = tuple(self.capture_records)
        self._validate_invocations(records, require_snapshot=True)
        previous = self.graph_templates.get(key)
        if previous is not None and len(previous) != len(records):
            raise RuntimeError(f"graph template changed for {key}")
        self.graph_templates[key] = records
        self.capture_key = None
        self.capture_records = []
        self.mode = "inactive"

    def prepare_runtime_step(self, run_mode: str, graph_key: object) -> None:
        # AFDFFNWorker synchronized CUDA after the preceding execute_model.
        self._resolve_pending_step()
        self._refresh_control()
        self.current_invocations = []
        self.invocation_stack = []
        if not self.active:
            self.mode = "inactive"
            return
        if run_mode == "replay":
            key = _graph_key_text(graph_key)
            try:
                template = self.graph_templates[key]
            except KeyError as exc:
                raise RuntimeError(f"missing cost-profile graph template: {key}") from exc
            self.mode = "replay"
            self.pending_records = template
            self.pending_step = self.step_sequence
            self.step_sequence += 1
        elif run_mode == "eager":
            self.mode = "eager"
        else:
            self.mode = "inactive"

    def finish_runtime_step(self, run_mode: str) -> None:
        if not self.active or run_mode != "eager":
            return
        if self.invocation_stack:
            raise RuntimeError("unfinished AFD eager profiling invocation")
        records = tuple(self.current_invocations)
        self._validate_invocations(records, require_snapshot=True)
        self.pending_records = records
        self.pending_step = self.step_sequence
        self.step_sequence += 1

    def begin_full(self, layer: int, hidden_states: Any) -> _Invocation | None:
        if self.mode not in {"capture", "eager"}:
            return None
        invocation = _Invocation(
            layer=int(layer),
            stage=_stage_index(),
            token_rows=int(hidden_states.shape[0]),
            full_start=_cuda_timing_event(graph_capture=self.mode == "capture"),
        )
        invocation.full_start.record()
        self.invocation_stack.append(invocation)
        return invocation

    def observe_topk(self, layer: int, topk_ids: Any) -> None:
        if self.mode not in {"capture", "eager"}:
            return
        if not self.invocation_stack:
            raise RuntimeError("routing callback has no active FFN invocation")
        invocation = self.invocation_stack[-1]
        if invocation.layer != int(layer) or invocation.topk_ids is not None:
            raise RuntimeError("routing callback layer/count mismatch")
        # FULL_DECODE_ONLY may replay fixed-shape graph buffers.  Use the same
        # CPU DP metadata contract as vLLM's RoutedExpertsCapturer so padded
        # rows never become apparent expert assignments.
        from vllm.forward_context import get_forward_context

        dp_metadata = get_forward_context().dp_metadata
        counts = (
            dp_metadata.num_tokens_across_dp_cpu
            if dp_metadata is not None else None
        )
        valid_rows = valid_routing_rows(
            int(topk_ids.shape[0]), self.rank, counts
        )
        invocation.token_rows = valid_rows
        invocation.topk_ids = topk_ids[:valid_rows]

    def run_local(self, function: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        if self.mode not in {"capture", "eager"}:
            return function(*args, **kwargs)
        if not self.invocation_stack:
            raise RuntimeError("local expert call has no active FFN invocation")
        invocation = self.invocation_stack[-1]
        if invocation.local_start is not None:
            raise RuntimeError("multiple local expert calls in one FFN invocation")
        graph_capture = self.mode == "capture"
        invocation.local_start = _cuda_timing_event(graph_capture=graph_capture)
        invocation.local_end = _cuda_timing_event(graph_capture=graph_capture)
        invocation.local_start.record()
        output = function(*args, **kwargs)
        invocation.local_end.record()
        return output

    def end_full(self, invocation: _Invocation | None) -> None:
        if invocation is None:
            return
        if not self.invocation_stack or self.invocation_stack.pop() is not invocation:
            raise RuntimeError("AFD FFN invocation stack mismatch")
        if invocation.topk_ids is None:
            raise RuntimeError("missing same-invocation routing metadata")
        invocation.full_end = _cuda_timing_event(
            graph_capture=self.mode == "capture"
        )
        invocation.full_end.record()
        # This copy is outside both measured intervals and avoids an extra
        # bincount/profiling kernel in the local expert interval.
        invocation.routing_snapshot = invocation.topk_ids.clone()
        if self.mode == "capture":
            self.capture_records.append(invocation)
        else:
            self.current_invocations.append(invocation)

    @staticmethod
    def _validate_invocations(
        records: tuple[_Invocation, ...], *, require_snapshot: bool
    ) -> None:
        if not records:
            raise RuntimeError("AFD cost profile step has no MoE invocations")
        keys = [(record.stage, record.layer) for record in records]
        if len(keys) != len(set(keys)):
            raise RuntimeError("duplicate stage/layer in AFD cost profile step")
        for record in records:
            if not 0 <= record.layer < EXPECTED_LAYERS:
                raise RuntimeError(f"invalid profiled layer {record.layer}")
            if record.full_end is None or record.local_start is None or record.local_end is None:
                raise RuntimeError("incomplete local/full timing boundary")
            if require_snapshot and record.routing_snapshot is None:
                raise RuntimeError("missing routing snapshot")

    def _refresh_control(self) -> None:
        if not self.control_path.is_file():
            return
        generation, requested_active = _load_control(self.control_path)
        if generation != self.generation:
            if self.active:
                self._dump()
            self.generation = generation
            self.active = requested_active
            if requested_active:
                self._reset_window()
            return
        if requested_active and not self.active:
            self._reset_window()
            self.active = True
        elif not requested_active and self.active:
            self._dump()
            self.active = False

    def _reset_window(self) -> None:
        self.step_sequence = 0
        self.rows.clear()
        self.histograms.clear()
        self.chunk_paths.clear()
        self.next_chunk = 0

    def _resolve_pending_step(self) -> None:
        if self.pending_records is None:
            return
        if self.pending_step is None:
            raise RuntimeError("pending AFD cost profile has no step id")
        for record in self.pending_records:
            topk = record.routing_snapshot.detach().cpu().numpy().reshape(-1)
            valid = topk[(topk >= 0) & (topk < EXPECTED_EXPERTS)]
            histogram = np.bincount(valid, minlength=EXPECTED_EXPERTS).astype(
                np.int32, copy=False
            )
            self.rows.append({
                "step_sequence": int(self.pending_step),
                "stage": int(record.stage),
                "layer": int(record.layer),
                "token_rows": int(record.token_rows),
                "topk_assignments": int(histogram.sum()),
                "local_elapsed_ms": float(
                    record.local_start.elapsed_time(record.local_end)
                ),
                "full_elapsed_ms": float(
                    record.full_start.elapsed_time(record.full_end)
                ),
            })
            self.histograms.append(histogram)
        self.pending_records = None
        self.pending_step = None
        if len(self.rows) >= CHUNK_RECORDS:
            self._flush_chunk()

    def _flush_chunk(self) -> None:
        if not self.rows:
            return
        if self.generation is None:
            raise RuntimeError("cannot flush AFD cost profile without generation")
        self.directory.mkdir(parents=True, exist_ok=True)
        target = self.directory / (
            f"afd-cost-rank-{self.rank}-pid-{os.getpid()}-"
            f"chunk-{self.next_chunk:06d}-{self.generation}.npz"
        )
        temporary = target.with_name(target.name + ".tmp.npz")
        np.savez_compressed(
            temporary,
            step_sequences=np.asarray(
                [row["step_sequence"] for row in self.rows], dtype=np.int32
            ),
            stages=np.asarray([row["stage"] for row in self.rows], dtype=np.int16),
            layers=np.asarray([row["layer"] for row in self.rows], dtype=np.int16),
            token_rows=np.asarray(
                [row["token_rows"] for row in self.rows], dtype=np.int32
            ),
            topk_assignments=np.asarray(
                [row["topk_assignments"] for row in self.rows], dtype=np.int32
            ),
            expert_histograms=np.stack(self.histograms).astype(np.int32, copy=False),
            local_elapsed_ms=np.asarray(
                [row["local_elapsed_ms"] for row in self.rows], dtype=np.float32
            ),
            full_elapsed_ms=np.asarray(
                [row["full_elapsed_ms"] for row in self.rows], dtype=np.float32
            ),
            ffn_rank=np.asarray(self.rank, dtype=np.int64),
            physical_gpu=np.asarray(self.physical_gpu, dtype=np.int64),
            frequency_mhz=np.asarray(self.frequency_mhz, dtype=np.int64),
            generation=np.asarray(self.generation),
            chunk_index=np.asarray(self.next_chunk, dtype=np.int64),
        )
        os.replace(temporary, target)
        self.chunk_paths.append(target)
        self.next_chunk += 1
        self.rows.clear()
        self.histograms.clear()

    def _dump(self) -> None:
        if self.generation is None:
            raise RuntimeError("cannot dump AFD cost profile without generation")
        if self.generation in self.dumped_generations:
            raise RuntimeError(f"generation already dumped: {self.generation}")
        self._resolve_pending_step()
        self._flush_chunk()
        if not self.chunk_paths:
            raise RuntimeError("AFD cost profile window produced no chunks")
        metadata = {
            "schema_version": 2,
            "generation": self.generation,
            "rank": self.rank,
            "physical_gpu": self.physical_gpu,
            "frequency_mhz": self.frequency_mhz,
            "chunk_count": len(self.chunk_paths),
            "chunks": [
                {"path": path.name, "sha256": _sha256(path)}
                for path in self.chunk_paths
            ],
            "local_boundary": (
                "FusedMoEKernelModularImpl._fused_experts including local "
                "workspace preparation, assignment/permutation, and expert "
                "kernels; excluding EP dispatch and combine"
            ),
            "full_boundary": (
                "Qwen3MoeSparseMoeBlock per-layer entry through returned output; "
                "includes router, EP dispatch, local experts, combine and dependencies"
            ),
            "routing_source": (
                "same-invocation logical topk_ids snapshot enqueued after full timer"
            ),
            "per_step_sync_added": False,
        }
        target = self.directory / (
            f"afd-cost-rank-{self.rank}-pid-{os.getpid()}-{self.generation}.json"
        )
        target.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        self.dumped_generations.add(self.generation)


_RUNTIME_PROFILERS: dict[tuple[str, int], RuntimeAFDCostProfiler] = {}


def _parse_rank_values(name: str, expected: int) -> list[int]:
    values = [
        int(value)
        for value in os.environ.get(name, "").split(",")
        if value.strip()
    ]
    if len(values) != expected:
        raise ValueError(f"{name} must contain {expected} comma-separated values")
    return values


def _profiler_for_rank(rank: int) -> RuntimeAFDCostProfiler | None:
    raw = os.environ.get(PROFILE_DIR_ENV, "").strip()
    if not raw:
        return None
    directory = str(Path(raw).expanduser().resolve())
    clocks = _parse_rank_values(PROFILE_CLOCKS_ENV, EXPECTED_RANKS)
    physical = _parse_rank_values(PROFILE_PHYSICAL_GPUS_ENV, EXPECTED_RANKS)
    key = (directory, int(rank))
    return _RUNTIME_PROFILERS.setdefault(
        key,
        RuntimeAFDCostProfiler(
            Path(directory), int(rank), clocks[int(rank)], physical[int(rank)]
        ),
    )


def install_runtime_cost_profiler(moe_block: Any, layer_idx: int) -> bool:
    """Attach routing observation and local-expert timing to one loaded layer."""
    if not os.environ.get(PROFILE_DIR_ENV, "").strip():
        return False
    profiler = _profiler_for_rank(int(moe_block.ep_rank))
    if profiler is None:
        return False
    if int(moe_block.ep_size) != EXPECTED_RANKS:
        raise ValueError(f"AFD cost profiling requires EP={EXPECTED_RANKS}")
    runner = moe_block.experts
    router = runner.router
    router.set_capture_fn(
        lambda topk_ids, _layer=layer_idx, _profiler=profiler: (
            _profiler.observe_topk(_layer, topk_ids)
        )
    )
    quant_method = runner._quant_method
    moe_kernel = getattr(quant_method, "moe_kernel", None)
    kernel_impl = getattr(moe_kernel, "impl", None)
    if kernel_impl is None or not hasattr(kernel_impl, "_fused_experts"):
        raise RuntimeError(
            "AFD cost profiler must be installed after MoE weights and the "
            "modular kernel are initialized"
        )
    if getattr(kernel_impl, "_afd_cost_profiler_installed", False):
        return False
    original_apply = kernel_impl._fused_experts

    def profiled_apply(*args: Any, **kwargs: Any) -> Any:
        return profiler.run_local(original_apply, *args, **kwargs)

    kernel_impl._fused_experts = profiled_apply
    kernel_impl._afd_cost_profiler_installed = True
    return True


def install_model_runtime_cost_profilers(model: Any) -> int:
    """Install cost profiling on every loaded MoE layer in an AFD FFN model."""
    if not os.environ.get(PROFILE_DIR_ENV, "").strip():
        return 0
    model_root = getattr(model, "model", None)
    layers = getattr(model_root, "layers", None)
    if layers is None:
        raise RuntimeError("AFD FFN model does not expose model.layers")
    installed = 0
    moe_layers = 0
    for layer_idx, layer in enumerate(layers):
        moe_block = getattr(layer, "mlp", None)
        if getattr(moe_block, "experts", None) is None:
            continue
        moe_layers += 1
        installed += int(install_runtime_cost_profiler(moe_block, layer_idx))
    if moe_layers != EXPECTED_LAYERS:
        raise RuntimeError(
            f"expected {EXPECTED_LAYERS} loaded MoE layers, got {moe_layers}"
        )
    return installed


def run_profiled_ffn(
    layer_idx: int,
    hidden_states: Any,
    function: Callable[[Any], Any],
    ep_rank: int,
) -> Any:
    profiler = _profiler_for_rank(int(ep_rank))
    if profiler is None:
        return function(hidden_states)
    invocation = profiler.begin_full(layer_idx, hidden_states)
    try:
        return function(hidden_states)
    finally:
        profiler.end_full(invocation)


def begin_graph_profile_capture(rank: int, graph_key: object) -> None:
    profiler = _profiler_for_rank(rank)
    if profiler is not None:
        profiler.begin_graph_capture(graph_key)


def end_graph_profile_capture(rank: int, graph_key: object) -> None:
    profiler = _profiler_for_rank(rank)
    if profiler is not None:
        profiler.end_graph_capture(graph_key)


def prepare_runtime_profile_step(rank: int, run_mode: str, graph_key: object) -> None:
    profiler = _profiler_for_rank(rank)
    if profiler is not None:
        profiler.prepare_runtime_step(run_mode, graph_key)


def finish_runtime_profile_step(rank: int, run_mode: str) -> None:
    profiler = _profiler_for_rank(rank)
    if profiler is not None:
        profiler.finish_runtime_step(run_mode)


def wait_for_cost_profile_metadata(
    directory: Path, generation: str, *, timeout_s: float = 120.0
) -> list[Path]:
    deadline = time.monotonic() + timeout_s
    pattern = f"afd-cost-rank-*-pid-*-{generation}.json"
    while time.monotonic() < deadline:
        paths = sorted(directory.glob(pattern))
        if len(paths) == EXPECTED_RANKS:
            return paths
        if len(paths) > EXPECTED_RANKS:
            raise ValueError(f"too many AFD cost profile metadata files: {paths}")
        time.sleep(0.25)
    raise TimeoutError("timed out waiting for AFD cost profiles")


def _load_rank_chunks(
    directory: Path, metadata_path: Path, output_directory: Path
) -> tuple[int, dict[str, np.ndarray], dict[str, Any]]:
    metadata = json.loads(metadata_path.read_text())
    rank = int(metadata["rank"])
    arrays: dict[str, list[np.ndarray]] = {
        key: []
        for key in (
            "step_sequences", "stages", "layers", "token_rows",
            "topk_assignments", "expert_histograms", "local_elapsed_ms",
            "full_elapsed_ms",
        )
    }
    copied_chunks: list[dict[str, Any]] = []
    for expected_index, chunk in enumerate(metadata["chunks"]):
        source = directory / chunk["path"]
        if _sha256(source) != chunk["sha256"]:
            raise ValueError(f"cost profile chunk hash mismatch: {source}")
        with np.load(source, allow_pickle=False) as archive:
            if int(archive["ffn_rank"]) != rank:
                raise ValueError(f"cost profile rank mismatch: {source}")
            if int(archive["chunk_index"]) != expected_index:
                raise ValueError(f"cost profile chunk sequence mismatch: {source}")
            for key in arrays:
                arrays[key].append(np.asarray(archive[key]))
        copied = output_directory / source.name
        copied.write_bytes(source.read_bytes())
        copied_chunks.append({"path": copied.name, "sha256": _sha256(copied)})
    combined = {key: np.concatenate(parts, axis=0) for key, parts in arrays.items()}
    records = len(combined["layers"])
    if combined["expert_histograms"].shape != (records, EXPECTED_EXPERTS):
        raise ValueError("invalid same-run routing histogram shape")
    histogram_totals = combined["expert_histograms"].sum(axis=1)
    if not np.array_equal(histogram_totals, combined["topk_assignments"]):
        raise ValueError("same-run routing assignment totals mismatch")
    expected_assignments = combined["token_rows"].astype(np.int64) * EXPECTED_TOP_K
    if not np.array_equal(histogram_totals, expected_assignments):
        raise ValueError(
            "same-run routing count does not equal "
            f"token_rows * top_k ({EXPECTED_TOP_K})"
        )
    local_elapsed = combined["local_elapsed_ms"].astype(np.float64)
    full_elapsed = combined["full_elapsed_ms"].astype(np.float64)
    if not np.all(np.isfinite(local_elapsed)) or not np.all(np.isfinite(full_elapsed)):
        raise ValueError("non-finite AFD cost timing")
    if np.any(local_elapsed <= 0) or np.any(full_elapsed <= 0):
        raise ValueError("non-positive AFD cost timing")
    containment = local_elapsed <= (
        full_elapsed + TIMING_CONTAINMENT_TOLERANCE_MS
    )
    containment_fraction = float(np.mean(containment))
    if containment_fraction < LOCAL_WITHIN_FULL_MIN_FRACTION:
        raise ValueError(
            "local AFD timing is not contained by the full timing boundary: "
            f"{containment_fraction:.6f} < {LOCAL_WITHIN_FULL_MIN_FRACTION:.6f}"
        )
    metadata["chunks"] = copied_chunks
    metadata["records"] = records
    metadata["validation"] = {
        "top_k": EXPECTED_TOP_K,
        "routing_assignments_equal_token_rows_times_top_k": True,
        "timings_finite_and_positive": True,
        "local_within_full_fraction": containment_fraction,
        "local_within_full_min_fraction": LOCAL_WITHIN_FULL_MIN_FRACTION,
        "timing_containment_tolerance_ms": TIMING_CONTAINMENT_TOLERANCE_MS,
    }
    return rank, combined, metadata


def finalize_same_run_cost_profiles(
    directory: Path,
    generation: str,
    output_directory: Path,
    *,
    timeout_s: float = 120.0,
) -> dict[str, Any]:
    """Validate/copy both rank profiles and write one combined same-run trace."""
    paths = wait_for_cost_profile_metadata(directory, generation, timeout_s=timeout_s)
    output_directory.mkdir(parents=True, exist_ok=True)
    per_rank: dict[int, dict[str, np.ndarray]] = {}
    metadata_by_rank: dict[int, dict[str, Any]] = {}
    rank_metadata: list[dict[str, Any]] = []
    for path in paths:
        rank, arrays, metadata = _load_rank_chunks(directory, path, output_directory)
        if rank in per_rank:
            raise ValueError(f"duplicate AFD cost profile rank {rank}")
        per_rank[rank] = arrays
        metadata_by_rank[rank] = metadata
        copied_metadata = output_directory / path.name
        copied_metadata.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        rank_metadata.append({
            "rank": rank,
            "path": copied_metadata.name,
            "sha256": _sha256(copied_metadata),
            "records": int(metadata["records"]),
        })
    if set(per_rank) != set(range(EXPECTED_RANKS)):
        raise ValueError(f"missing AFD cost profile rank: {set(per_rank)}")
    key_names = ("step_sequences", "stages", "layers")
    for key in key_names:
        if not np.array_equal(per_rank[0][key], per_rank[1][key]):
            raise ValueError(f"FFN ranks do not align in same run: {key}")
    keys = np.stack([per_rank[0][key] for key in key_names], axis=1)
    if len(np.unique(keys, axis=0)) != len(keys):
        raise ValueError("duplicate step/stage/layer keys in same-run profile")
    per_source_rank_histograms = np.stack([
        per_rank[0]["expert_histograms"].astype(np.int64),
        per_rank[1]["expert_histograms"].astype(np.int64),
    ])
    combined_histograms = per_source_rank_histograms.sum(axis=0)
    if not np.array_equal(
        combined_histograms,
        per_rank[0]["expert_histograms"].astype(np.int64)
        + per_rank[1]["expert_histograms"].astype(np.int64),
    ):
        raise ValueError("combined histogram does not equal both rank histograms")
    combined_token_rows = (
        per_rank[0]["token_rows"].astype(np.int64)
        + per_rank[1]["token_rows"].astype(np.int64)
    )
    if not np.array_equal(
        combined_histograms.sum(axis=1),
        combined_token_rows * EXPECTED_TOP_K,
    ):
        raise ValueError("combined routing count does not match both source ranks")
    output_path = output_directory / "afd_same_run_cost_profile.npz"
    np.savez_compressed(
        output_path,
        step_sequences=per_rank[0]["step_sequences"],
        stages=per_rank[0]["stages"],
        layers=per_rank[0]["layers"],
        expert_histograms=combined_histograms,
        per_source_rank_expert_histograms=per_source_rank_histograms,
        token_rows_by_source_rank=np.stack([
            per_rank[0]["token_rows"], per_rank[1]["token_rows"]
        ]),
        local_elapsed_ms_by_rank=np.stack([
            per_rank[0]["local_elapsed_ms"], per_rank[1]["local_elapsed_ms"]
        ]),
        full_elapsed_ms_by_rank=np.stack([
            per_rank[0]["full_elapsed_ms"], per_rank[1]["full_elapsed_ms"]
        ]),
        frequency_mhz_by_rank=np.asarray([
            int(metadata_by_rank[0]["frequency_mhz"]),
            int(metadata_by_rank[1]["frequency_mhz"]),
        ], dtype=np.int32),
    )
    result = {
        "schema_version": 2,
        "generation": generation,
        "records": int(len(keys)),
        "exact_same_run_rank_alignment": True,
        "profile": output_path.name,
        "profile_sha256": _sha256(output_path),
        "rank_profiles": sorted(rank_metadata, key=lambda row: row["rank"]),
        "timing_feature_ground_truth": "same invocation/run as CUDA events",
        "cross_run_join": False,
        "local_and_full_boundaries": True,
        "per_call_cuda_synchronize_added": False,
        "validation": {
            "top_k": EXPECTED_TOP_K,
            "per_rank_assignments_equal_token_rows_times_top_k": True,
            "combined_histogram_equals_sum_of_rank_histograms": True,
            "combined_assignments_equal_combined_token_rows_times_top_k": True,
            "timings_finite_and_positive": True,
            "local_within_full_fraction_by_rank": {
                str(rank): metadata_by_rank[rank]["validation"][
                    "local_within_full_fraction"
                ]
                for rank in range(EXPECTED_RANKS)
            },
        },
    }
    metadata_path = output_directory / "afd_same_run_cost_profile.json"
    metadata_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result
