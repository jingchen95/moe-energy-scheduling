import json
import asyncio
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.afd_profile_protocol import (
    AttentionProfileState, coordinate_profile_step, set_attention_profile_state, set_profile_state_rpc,
)
from src.afd_ffn_cost_profile import RuntimeAFDCostProfiler, _Invocation, finalize_same_run_cost_profiles
from src.afd_cost_model import load_same_run_cost_profile
from src.afd_cost_transaction import execution_fingerprint, mark_complete, read_complete, REQUIRED_FILES
from src.afd_static_placement import sha256_file


class _CPUCopy:
    def __init__(self, value):
        self.value = value
    def detach(self):
        return self
    def cpu(self):
        return self
    def numpy(self):
        return self.value


def _records():
    return tuple(_Invocation(
        layer=layer, stage=0, token_rows=2,
        full_start=SimpleNamespace(elapsed_time=lambda _: 1.0), full_end=object(),
        local_start=SimpleNamespace(elapsed_time=lambda _: 0.5), local_end=object(),
        routing_snapshot=_CPUCopy(np.zeros((2, 8), dtype=np.int32)),
    ) for layer in range(48))


def _payload(step, generation, active, **flags):
    return SimpleNamespace(runtime_step_id=step, profile_generation=generation,
                           profile_active=active, is_warmup=flags.get("warmup", False),
                           is_graph_capturing=flags.get("capture", False),
                           is_profile=flags.get("profile", False))


def _profilers(root):
    result = [RuntimeAFDCostProfiler(root, rank, 2520, rank + 2) for rank in range(2)]
    result[1].step_sequence = 1000
    for profiler in result:
        profiler.graph_templates[repr("graph")] = _records()
    return result


def test_skewed_files_have_no_effect_on_eight_generations(tmp_path):
    ranks = _profilers(tmp_path)
    for window in range(8):
        generation = f"generation-{window}"
        for offset, active in enumerate((True, True, False)):
            step = window * 3 + offset + 50
            for rank, profiler in enumerate(ranks):
                # Simulate the former one-step-displaced file observations.
                (tmp_path / "control.json").write_text(json.dumps({
                    "generation": "wrong", "active": bool(rank == 0),
                    "step": step + rank,
                }))
                profiler.prepare_runtime_step("replay", "graph", _payload(step, generation, active))
        output = tmp_path / f"out-{window}"
        final = finalize_same_run_cost_profiles(tmp_path, generation, output, timeout_s=.1)
        assert final["runtime_steps"] == 2
        assert final["first_runtime_step_id"] == window * 3 + 50
        assert final["last_runtime_step_id"] == window * 3 + 51
        profile = load_same_run_cost_profile(output / "afd_same_run_cost_profile.npz")
        assert not np.array_equal(*profile["diagnostic_step_sequences_by_rank"])
        assert final["exact_same_run_rank_alignment"]
    assert ranks[0].step_sequence == 16
    assert ranks[1].step_sequence == 1016
    assert ranks[0].last_runtime_step_id == ranks[1].last_runtime_step_id == 73


def test_active_generation_transition_resolves_previous_step_before_reset(tmp_path):
    for profiler in _profilers(tmp_path):
        profiler.prepare_runtime_step("replay", "graph", _payload(20, "A", True))
        profiler.prepare_runtime_step("replay", "graph", _payload(21, "B", True))
        profiler.prepare_runtime_step("replay", "graph", _payload(22, "B", False))
    a = finalize_same_run_cost_profiles(tmp_path, "A", tmp_path / "a", timeout_s=.1)
    b = finalize_same_run_cost_profiles(tmp_path, "B", tmp_path / "b", timeout_s=.1)
    assert a["first_runtime_step_id"] == a["last_runtime_step_id"] == 20
    assert b["first_runtime_step_id"] == b["last_runtime_step_id"] == 21


@pytest.mark.parametrize("flag", ["capture", "warmup", "profile"])
def test_excluded_steps_are_not_measurements_and_do_not_reset_window(tmp_path, flag):
    for profiler in _profilers(tmp_path):
        profiler.prepare_runtime_step("replay", "graph", _payload(40, "A", True))
        profiler.prepare_runtime_step("warmup", None, _payload(None, None, False, **{flag: True}))
        assert profiler.active
        profiler.prepare_runtime_step("replay", "graph", _payload(41, "A", True))
        profiler.prepare_runtime_step("replay", "graph", _payload(42, "A", False))
    result = finalize_same_run_cost_profiles(tmp_path, "A", tmp_path / "out", timeout_s=.1)
    assert result["runtime_steps"] == 2


def test_eager_records_use_payload_identity(tmp_path):
    for profiler in _profilers(tmp_path):
        profiler.prepare_runtime_step("eager", None, _payload(91, "E", True))
        profiler.current_invocations = list(_records())
        profiler.finish_runtime_step("eager")
        profiler.prepare_runtime_step("eager", None, _payload(92, "E", False))
    result = finalize_same_run_cost_profiles(tmp_path, "E", tmp_path / "out", timeout_s=.1)
    assert result["first_runtime_step_id"] == result["last_runtime_step_id"] == 91


def test_attention_identity_is_allocated_only_by_shared_dp_leader(monkeypatch):
    monkeypatch.setenv("AFD_FFN_TIMING_DIR", "/not/read/by/attention")
    runner0 = SimpleNamespace(vllm_config=SimpleNamespace(parallel_config=SimpleNamespace(tensor_parallel_size=1)))
    runner1 = SimpleNamespace(vllm_config=runner0.vllm_config,
                              _afd_cost_profile_state=AttentionProfileState(next_runtime_step_id=90000))
    wire = []
    group0 = SimpleNamespace(world_size=2, rank_in_group=0,
                             broadcast_object=lambda value, src: wire.append(value) or value)
    group1 = SimpleNamespace(world_size=2, rank_in_group=1,
                             broadcast_object=lambda value, src: wire.pop(0))
    for window in range(3):
        for active in (True, False):
            for runner in (runner0, runner1):
                set_attention_profile_state(runner, json.dumps({"generation": f"g{window}", "active": active}))
            a = coordinate_profile_step(runner0, excluded=False, group=group0)
            b = coordinate_profile_step(runner1, excluded=False, group=group1)
            assert a == b
            assert a["runtime_step_id"] == window * 2 + int(not active)
    assert runner0._afd_cost_profile_state.next_runtime_step_id == 6
    assert runner1._afd_cost_profile_state.next_runtime_step_id == 90000
    assert coordinate_profile_step(runner0, excluded=True, group=group0) == {}


def test_nonprofiled_path_is_noop(monkeypatch):
    monkeypatch.delenv("AFD_FFN_TIMING_DIR", raising=False)
    assert coordinate_profile_step(object(), excluded=False) == {}
    with pytest.raises(RuntimeError, match="not enabled"):
        set_attention_profile_state(object(), '{"generation":"g","active":true}')


def test_profile_rpc_requires_matching_acknowledgement(monkeypatch):
    from src import benchmark
    calls = []
    async def rpc(url, method, *, args):
        value = json.loads(args[0])
        calls.append((url, method, value))
        return [{**value, "schema_version": 3}]
    monkeypatch.setattr(benchmark, "collective_rpc", rpc)
    asyncio.run(set_profile_state_rpc("http://127.0.0.1", "g", True))
    asyncio.run(set_profile_state_rpc("http://127.0.0.1", "g", False))
    assert [row[2]["active"] for row in calls] == [True, False]
    assert all(row[1] == "set_afd_cost_profile_state" for row in calls)
    async def stale(*args, **kwargs):
        return [{"generation": "wrong", "active": True, "schema_version": 3}]
    monkeypatch.setattr(benchmark, "collective_rpc", stale)
    with pytest.raises(RuntimeError, match="acknowledgement"):
        asyncio.run(set_profile_state_rpc("http://127.0.0.1", "g", True))


def test_missing_rank_fails_closed(tmp_path):
    profiler = _profilers(tmp_path)[0]
    profiler.prepare_runtime_step("replay", "graph", _payload(50, "g", True))
    profiler.prepare_runtime_step("replay", "graph", _payload(51, "g", False))
    with pytest.raises(TimeoutError):
        finalize_same_run_cost_profiles(tmp_path, "g", tmp_path / "out", timeout_s=.01)


@pytest.mark.parametrize("kind", ["runtime_id", "duplicate", "missing_layer", "generation", "interior", "legacy", "warmup"])
def test_finalizer_rejects_identity_damage_without_repair(tmp_path, kind):
    ranks = _profilers(tmp_path)
    for step in range(10, 14):
        for profiler in ranks:
            profiler.prepare_runtime_step("replay", "graph", _payload(step, "g", step < 13))
    metadata_path = next(tmp_path.glob("afd-cost-rank-1-pid-*-g.json"))
    metadata = json.loads(metadata_path.read_text())
    chunk = tmp_path / metadata["chunks"][0]["path"]
    with np.load(chunk, allow_pickle=False) as data:
        arrays = {key: data[key].copy() for key in data.files}
    if kind == "runtime_id":
        arrays["runtime_step_ids"] += 1
    elif kind == "duplicate":
        arrays["layers"][47] = 46
    elif kind == "missing_layer":
        for key in ("runtime_step_ids", "step_sequences", "run_modes", "stages", "layers", "token_rows", "topk_assignments", "expert_histograms", "local_elapsed_ms", "full_elapsed_ms"):
            arrays[key] = arrays[key][:-1]
    elif kind == "generation":
        arrays["generation"] = np.asarray("other")
    elif kind == "interior":
        arrays["runtime_step_ids"][48:] += 1
    elif kind == "legacy":
        metadata["schema_version"] = 2
    elif kind == "warmup":
        arrays["run_modes"][0] = "warmup"
    np.savez_compressed(chunk, **arrays)
    metadata["chunks"][0]["sha256"] = sha256_file(chunk)
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        finalize_same_run_cost_profiles(tmp_path, "g", tmp_path / "rejected", timeout_s=.1)
    assert not (tmp_path / "rejected/afd_same_run_cost_profile.npz").exists()


def _complete_fixture(root):
    attempt = root / "attempt-1"
    run = attempt / "runs/g"
    run.mkdir(parents=True)
    for name in REQUIRED_FILES:
        (run / name).write_text("{}")
    (run / "trace.jsonl").write_text("trace")
    (run / "expert_placement.json").write_text("placement")
    (run / "git_commit.txt").write_text("source\n")
    (run / "metrics.json").write_text(json.dumps({"completed_requests": 300, "failed_requests": 0,
        **{f"gpu{g}_avg_sm_clock": 2520 for g in range(4)}}))
    (run / "afd_same_run_cost_finalize.json").write_text(json.dumps({"schema_version": 3,
        "identity_source": "attention_dp_rank0_cpu_broadcast", "exact_same_run_rank_alignment": True,
        "profile_sha256": sha256_file(run / "afd_same_run_cost_profile.npz")}))
    for name in ("cost-timing-validation.json", "live-validation.json"):
        (run / name).write_text('{"status":"PASS"}')
    (attempt / "runtime-ownership.json").write_text('{"actual_mapping_sha256":"mapping"}')
    (attempt / "compute-apps-after.csv").write_text("")
    (attempt / "server").mkdir()
    for name in ("attention.log", "ffn.log", "attention-command.txt", "ffn-command.txt"):
        (attempt / "server" / name).write_text("server")
    definition = {"frequency_mhz": 2520, "placement": {
        "sha256": sha256_file(run / "expert_placement.json"), "canonical_mapping_sha256": "mapping"}}
    plan = {"git_commit": "source", "trace_sha256": sha256_file(run / "trace.jsonl"), "request_rate": 8}
    fingerprint = execution_fingerprint(plan, definition, "patch")
    return attempt, run, plan, definition, fingerprint


def test_transactional_resume_checks_identical_fingerprint_and_all_artifacts(tmp_path):
    attempt, run, plan, definition, fingerprint = _complete_fixture(tmp_path)
    assert read_complete(tmp_path, fingerprint) is None
    mark_complete(tmp_path, run, attempt, fingerprint, definition)
    assert read_complete(tmp_path, fingerprint) == run
    for key in fingerprint:
        with pytest.raises(ValueError, match="fingerprint"):
            read_complete(tmp_path, {**fingerprint, key: "changed"})
    changed = execution_fingerprint({**plan, "request_rate": 16}, definition, "patch")
    with pytest.raises(ValueError, match="fingerprint"):
        read_complete(tmp_path, changed)
    (run / "requests.jsonl").write_text("changed")
    with pytest.raises(ValueError, match="hash drift"):
        read_complete(tmp_path, fingerprint)


@pytest.mark.parametrize("damage", ["requests", "clocks", "legacy", "ownership", "missing", "shutdown"])
def test_incomplete_configuration_cannot_receive_complete_marker(tmp_path, damage):
    attempt, run, _, definition, fingerprint = _complete_fixture(tmp_path)
    if damage in ("requests", "clocks"):
        path = run / "metrics.json"
        value = json.loads(path.read_text())
        value["completed_requests" if damage == "requests" else "gpu2_avg_sm_clock"] = 299
        path.write_text(json.dumps(value))
    elif damage == "legacy":
        (run / "afd_same_run_cost_finalize.json").write_text('{"schema_version":2}')
    elif damage == "ownership":
        (attempt / "runtime-ownership.json").write_text('{"actual_mapping_sha256":"wrong"}')
    elif damage == "missing":
        (run / "requests.jsonl").unlink()
    elif damage == "shutdown":
        (attempt / "compute-apps-after.csv").write_text("worker")
    with pytest.raises(ValueError):
        mark_complete(tmp_path, run, attempt, fingerprint, definition)
    assert not (tmp_path / "COMPLETE.json").exists()


def test_launcher_has_one_allocation_eight_windows_and_strict_resume():
    root = Path(__file__).resolve().parents[1]
    text = (root / "scripts/milestone_n_afd_ffn_cost_model.sbatch").read_text()
    assert "for generation in {0..7}" in text
    assert text.index("phase-a:shutdown") < text.index('while IFS=')
    assert "src.afd_cost_transaction check" in text
    assert "src.afd_cost_transaction mark" in text
    assert "attempt-${SLURM_JOB_ID}" in text
    assert "write_profile_control" not in text
    assert "sbatch " not in text.split("set -Eeuo pipefail", 1)[1]
