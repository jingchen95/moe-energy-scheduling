import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.afd_cost_model import (
    fit_binned_curve,
    join_same_run_cost_profile,
    leave_one_group_out_power,
    load_same_run_cost_profile,
    placement_features,
    prediction_metrics,
)
from src.afd_ffn_cost_profile import (
    _cuda_timing_event,
    finalize_same_run_cost_profiles,
    install_model_runtime_cost_profilers,
    install_runtime_cost_profiler,
    valid_routing_rows,
)
from src.afd_routing_profile import combine_rank_routing_traces
from src.afd_static_placement import sha256_file
from tools.validate_milestone_n_cost_timing import validate_average_clocks
from tools.prepare_milestone_n_cost_model import (
    PROFILED_MINI_SMOKE_REQUESTS,
    _profiled_mini_smoke_bytes,
)


ROOT = Path(__file__).resolve().parents[1]


def test_cuda_graph_timing_uses_external_event_nodes(monkeypatch):
    calls = []
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(Event=lambda **kwargs: calls.append(kwargs) or object())
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    _cuda_timing_event(graph_capture=True)
    _cuda_timing_event(graph_capture=False)

    assert calls == [
        {"enable_timing": True, "external": True},
        {"enable_timing": True, "external": False},
    ]


class _FakeRouter:
    def set_capture_fn(self, function):
        self.capture_fn = function


class _FakeKernelImpl:
    def _fused_experts(self, value):
        return value + 1


def _fake_moe_block(*, loaded: bool = True):
    quant_method = SimpleNamespace(
        moe_kernel=(SimpleNamespace(impl=_FakeKernelImpl()) if loaded else None)
    )
    runner = SimpleNamespace(router=_FakeRouter(), _quant_method=quant_method)
    return SimpleNamespace(ep_rank=0, ep_size=2, experts=runner)


def test_runtime_cost_profiler_requires_post_load_modular_kernel(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("AFD_FFN_TIMING_DIR", str(tmp_path))
    monkeypatch.setenv("AFD_FFN_CLOCKS_MHZ", "2520,2520")
    monkeypatch.setenv("AFD_FFN_PHYSICAL_GPUS", "2,3")
    with np.testing.assert_raises_regex(RuntimeError, "after MoE weights"):
        install_runtime_cost_profiler(_fake_moe_block(loaded=False), 0)


def test_model_runtime_cost_profiler_installs_after_load_once(tmp_path, monkeypatch):
    monkeypatch.setenv("AFD_FFN_TIMING_DIR", str(tmp_path))
    monkeypatch.setenv("AFD_FFN_CLOCKS_MHZ", "2520,2520")
    monkeypatch.setenv("AFD_FFN_PHYSICAL_GPUS", "2,3")
    layers = [SimpleNamespace(mlp=_fake_moe_block()) for _ in range(48)]
    model = SimpleNamespace(model=SimpleNamespace(layers=layers))

    assert install_model_runtime_cost_profilers(model) == 48
    assert install_model_runtime_cost_profilers(model) == 0
    assert layers[0].mlp.experts._quant_method.moe_kernel.impl._fused_experts(2) == 3


def _write_trace_chunk(path: Path, rank: int, histograms: np.ndarray) -> None:
    records = len(histograms)
    np.savez_compressed(
        path,
        expert_histograms=histograms,
        layers=np.arange(records, dtype=np.int16),
        call_sequences=np.zeros(records, dtype=np.int32),
        token_rows=np.full(records, 2 + rank, dtype=np.int32),
        topk_assignments=histograms.sum(axis=1).astype(np.int32),
        ep_rank=np.asarray(rank),
        generation=np.asarray("g"),
        chunk_index=np.asarray(0),
    )


def test_routing_trace_combines_rank_histograms_by_exact_call(tmp_path):
    hist0 = np.zeros((48, 128), dtype=np.int64)
    hist1 = np.zeros((48, 128), dtype=np.int64)
    hist0[:, 0] = 4
    hist1[:, 1] = 6
    _write_trace_chunk(
        tmp_path / "routing-trace-rank-0-pid-10-chunk-000000-g.npz", 0, hist0
    )
    _write_trace_chunk(
        tmp_path / "routing-trace-rank-1-pid-11-chunk-000000-g.npz", 1, hist1
    )
    combined, metadata = combine_rank_routing_traces(tmp_path, "g")
    assert metadata["records"] == 48
    assert metadata["steps_per_layer"] == 1
    assert np.all(combined["expert_histograms"][:, 0] == 4)
    assert np.all(combined["expert_histograms"][:, 1] == 6)
    assert combined["token_rows_by_source_rank"].shape == (2, 48)


def test_placement_features_reconstruct_rank_local_assignment_counts():
    histograms = np.zeros((48, 128), dtype=np.int64)
    histograms[:, 0] = 7
    histograms[:, 64] = 3
    routing = {"layers": np.arange(48), "expert_histograms": histograms}
    mapping = np.tile(np.repeat([0, 1], 64), (48, 1))
    features = placement_features(routing, mapping)
    assert np.all(features[0]["n"] == 7)
    assert np.all(features[1]["n"] == 3)
    assert np.all(features[0]["active_local_experts"] == 1)
    assert np.all(features[1]["max_tokens_per_local_expert"] == 3)


def test_binned_curve_interpolates_and_reports_error():
    n = np.arange(1, 101)
    elapsed = 2.0 + 0.5 * n
    curve = fit_binned_curve(n, elapsed, bins=10)
    prediction = curve.predict(np.asarray([10.0, 50.0, 90.0]))
    metrics = prediction_metrics(np.asarray([7.0, 27.0, 47.0]), prediction)
    assert metrics["mae"] < 1.0
    assert metrics["p95_absolute_error"] < 1.0


def test_power_validation_holds_out_whole_configuration_groups():
    rows = []
    for config_index, load in enumerate((10.0, 20.0, 30.0, 40.0)):
        for repetition in range(3):
            power = 80.0 + 2.0 * load
            rows.append({
                "config_id": f"c{config_index}",
                "rank": 0,
                "frequency_mhz": 1635,
                "load_rate": load,
                "avg_power_w": power,
                "duration_s": 5.0,
                "window_energy_j": power * 5.0,
                "repetition": repetition,
            })
    predictions = leave_one_group_out_power(rows)
    assert len(predictions) == len(rows)
    assert max(abs(row["predicted_power_w"] - row["avg_power_w"])
               for row in predictions) < 1e-9


def test_patch_manifest_and_one_job_scope_are_reproducible():
    manifest = json.loads(
        (ROOT / "patches/afd-plugin-0.26.0-cost-model.json").read_text()
    )
    patch = ROOT / "patches" / manifest["patch_file"]
    assert hashlib.sha256(patch.read_bytes()).hexdigest() == manifest["patch_sha256"]
    launcher = (ROOT / "scripts/milestone_n_afd_ffn_cost_model.sbatch").read_text()
    assert "#SBATCH --exclusive" in launcher
    assert "--afd-timing-control-dir" in launcher
    assert "--routing-trace" not in launcher
    assert "balanced-f2520-primary" in (
        ROOT / "tools/prepare_milestone_n_cost_model.py"
    ).read_text()
    assert "--defer-current-verification" in launcher
    assert "run_profiled_mini_smoke" in launcher
    assert "profiled-mini-smoke-trace.jsonl" in launcher
    assert "manage_afd_cost_model_patch.sh revert" in launcher
    assert "pkill" not in launcher
    assert "killall" not in launcher
    common = (ROOT / "scripts/afd_common.sh").read_text()
    stop_body = common.split("afd_stop_process_group()", 1)[1].split(
        "afd_start_gpu_sampler()", 1
    )[0]
    assert 'pgrep -g "${pid}"' in stop_body
    assert 'kill -0 "${pid}"' not in stop_body
    assert '"${pgid}" != "${pid}"' in stop_body
    assert 'VLLM_PLUGINS=afd' in stop_body
    assert '"${state}" == Z*' in stop_body


@pytest.mark.skipif(sys.platform != "linux", reason="requires Linux PGID/procfs")
def test_cleanup_stops_group_when_leader_has_exited(tmp_path):
    child_file = tmp_path / "child.pid"
    environment = {
        **os.environ,
        "AFD_PROJECT_ROOT": str(ROOT),
        "VLLM_PLUGINS": "afd",
        "PYTHONPATH": str(ROOT),
    }
    leader = subprocess.Popen(
        [
            "setsid",
            "bash",
            "-c",
            f"trap '' HUP; sleep 60 & printf '%s\\n' $! > {child_file!s}",
        ],
        env=environment,
    )
    pgid = leader.pid
    try:
        leader.wait(timeout=5)
        deadline = time.monotonic() + 5
        while not child_file.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        child = int(child_file.read_text().strip())
        assert os.getpgid(child) == pgid
        command = (
            f"export AFD_PROJECT_ROOT={ROOT!s}; "
            f"export AFD_PLUGIN_ROOT={ROOT / 'third_party/afd-plugin'!s}; "
            f"source {ROOT / 'scripts/afd_common.sh'!s}; "
            f"afd_stop_process_group {pgid}"
        )
        subprocess.run(["bash", "-c", command], check=True, timeout=40)
        state = subprocess.run(
            ["ps", "-o", "stat=", "-p", str(child)],
            check=False,
            capture_output=True,
            text=True,
        ).stdout.strip()
        assert not state or state.startswith("Z")
    finally:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def test_config_has_same_run_local_and_full_instrumentation_and_no_optimizer():
    text = (ROOT / "configs/milestone_n_afd_ffn_cost_model.yaml").read_text()
    assert "routing_and_timing_same_invocation: true" in text
    assert "cross_run_routing_timing_join: false" in text
    assert "local_timing_boundary:" in text
    assert "full_timing_boundary:" in text
    assert "per_call_cuda_synchronize_added: false" in text
    assert "optimizer" not in text.lower()


def _write_same_run_rank(
    directory: Path,
    rank: int,
    generation: str,
    *,
    assignments: int | None = None,
    local_elapsed_ms: float | None = None,
    full_elapsed_ms: float | None = None,
) -> None:
    records = 48
    token_rows = 2 + rank
    assignment_count = assignments if assignments is not None else token_rows * 8
    histograms = np.zeros((records, 128), dtype=np.int32)
    histograms[:, rank] = assignment_count
    chunk = directory / (
        f"afd-cost-rank-{rank}-pid-{100 + rank}-chunk-000000-{generation}.npz"
    )
    np.savez_compressed(
        chunk,
        step_sequences=np.zeros(records, dtype=np.int32),
        stages=np.zeros(records, dtype=np.int16),
        layers=np.arange(records, dtype=np.int16),
        token_rows=np.full(records, token_rows, dtype=np.int32),
        topk_assignments=histograms.sum(axis=1),
        expert_histograms=histograms,
        local_elapsed_ms=np.full(
            records,
            0.5 + rank if local_elapsed_ms is None else local_elapsed_ms,
            dtype=np.float32,
        ),
        full_elapsed_ms=np.full(
            records,
            0.8 + rank if full_elapsed_ms is None else full_elapsed_ms,
            dtype=np.float32,
        ),
        ffn_rank=np.asarray(rank),
        physical_gpu=np.asarray(rank + 2),
        frequency_mhz=np.asarray(2520),
        generation=np.asarray(generation),
        chunk_index=np.asarray(0),
    )
    metadata = {
        "schema_version": 2,
        "generation": generation,
        "rank": rank,
        "physical_gpu": rank + 2,
        "frequency_mhz": 2520,
        "chunks": [{"path": chunk.name, "sha256": sha256_file(chunk)}],
    }
    (directory / f"afd-cost-rank-{rank}-pid-{100 + rank}-{generation}.json").write_text(
        json.dumps(metadata)
    )


def test_same_run_profile_combines_routing_and_two_timing_boundaries(tmp_path):
    control = tmp_path / "control"
    output = tmp_path / "output"
    control.mkdir()
    for rank in range(2):
        _write_same_run_rank(control, rank, "g")
    metadata = finalize_same_run_cost_profiles(control, "g", output, timeout_s=0.1)
    assert metadata["cross_run_join"] is False
    profile = load_same_run_cost_profile(output / "afd_same_run_cost_profile.npz")
    assert profile["local_elapsed_ms_by_rank"].shape == (2, 48)
    assert profile["full_elapsed_ms_by_rank"].shape == (2, 48)
    assert np.all(profile["expert_histograms"][:, 0] == 16)
    assert np.all(profile["expert_histograms"][:, 1] == 24)
    assert metadata["validation"][
        "per_rank_assignments_equal_token_rows_times_top_k"
    ] is True
    assert metadata["validation"][
        "combined_histogram_equals_sum_of_rank_histograms"
    ] is True


def test_same_run_profile_rejects_wrong_topk_count(tmp_path):
    control = tmp_path / "control"
    output = tmp_path / "output"
    control.mkdir()
    _write_same_run_rank(control, 0, "g", assignments=15)
    _write_same_run_rank(control, 1, "g")
    with np.testing.assert_raises_regex(ValueError, "token_rows \\* top_k"):
        finalize_same_run_cost_profiles(control, "g", output, timeout_s=0.1)


def test_same_run_profile_rejects_invalid_or_uncontained_timing(tmp_path):
    for name, local, full, pattern in (
        ("nan", float("nan"), 1.0, "non-finite"),
        ("outside", 2.0, 1.0, "not contained"),
    ):
        control = tmp_path / name / "control"
        output = tmp_path / name / "output"
        control.mkdir(parents=True)
        _write_same_run_rank(
            control, 0, "g", local_elapsed_ms=local, full_elapsed_ms=full
        )
        _write_same_run_rank(control, 1, "g")
        with np.testing.assert_raises_regex(ValueError, pattern):
            finalize_same_run_cost_profiles(control, "g", output, timeout_s=0.1)


def test_profiled_mini_smoke_trace_is_deterministic_first_eight(tmp_path):
    trace = tmp_path / "trace.jsonl"
    trace.write_text("".join(f'{{"request_id":"{i}"}}\n' for i in range(300)))
    smoke = _profiled_mini_smoke_bytes(trace)
    assert PROFILED_MINI_SMOKE_REQUESTS == 8
    assert smoke.decode().splitlines() == [
        f'{{"request_id":"{i}"}}' for i in range(8)
    ]


def test_same_run_join_uses_own_histogram_and_local_boundary(tmp_path):
    mapping = np.tile(np.repeat([0, 1], 64), (48, 1)).tolist()
    placement = tmp_path / "placement.json"
    placement.write_text(json.dumps({
        "schema_version": 1,
        "name": "test",
        "num_layers": 48,
        "num_experts": 128,
        "num_ranks": 2,
        "experts_per_rank": 64,
        "logical_to_rank": mapping,
    }))
    histograms = np.zeros((48, 128), dtype=np.int64)
    histograms[:, 0] = 7
    histograms[:, 64] = 3
    profile = {
        "step_sequences": np.zeros(48, dtype=np.int32),
        "stages": np.zeros(48, dtype=np.int16),
        "layers": np.arange(48, dtype=np.int16),
        "expert_histograms": histograms,
        "token_rows_by_source_rank": np.full((2, 48), 5, dtype=np.int32),
        "local_elapsed_ms_by_rank": np.vstack([
            np.full(48, 1.0), np.full(48, 2.0)
        ]),
        "full_elapsed_ms_by_rank": np.vstack([
            np.full(48, 3.0), np.full(48, 4.0)
        ]),
        "frequency_mhz_by_rank": np.asarray([1635, 1635]),
    }
    rows, _ = join_same_run_cost_profile(
        profile, placement, config_id="c", lifecycle="primary"
    )
    rank0 = [row for row in rows if row["physical_ffn_rank"] == 0]
    rank1 = [row for row in rows if row["physical_ffn_rank"] == 1]
    assert {row["n"] for row in rank0} == {7}
    assert {row["n"] for row in rank1} == {3}
    assert {row["local_elapsed_ms"] for row in rank0} == {1.0}
    assert {row["full_elapsed_ms"] for row in rank1} == {4.0}


def test_graph_routing_rows_exclude_fixed_buffer_padding():
    assert valid_routing_rows(64, 0, [17, 23]) == 17
    assert valid_routing_rows(64, 1, [17, 23]) == 23


def test_milestone_n_validates_average_clocks_under_measured_load():
    metrics = {
        "gpu0_avg_sm_clock": 2519.5,
        "gpu1_avg_sm_clock": 2520.0,
        "gpu2_avg_sm_clock": 1635.2,
        "gpu3_avg_sm_clock": 1634.8,
    }
    targets, actual = validate_average_clocks(metrics, 1635)
    assert targets == [2520, 2520, 1635, 1635]
    assert actual == [2519.5, 2520.0, 1635.2, 1634.8]

    metrics["gpu2_avg_sm_clock"] = 1629.0
    with np.testing.assert_raises_regex(ValueError, "GPU 2 average clock"):
        validate_average_clocks(metrics, 1635)
