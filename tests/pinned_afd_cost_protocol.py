"""Explicitly run with the pinned AFD interpreter and cost patch applied."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from afd_plugin.connectors.metadata import AFDControlPayload, AFDDPMetadata, encode_control_payload, decode_control_payload
from afd_plugin.v1.worker.attention_metadata import AFDMetadataProviderMixin
from afd_plugin.v1.worker.attention_worker import AFDAttentionWorker


@pytest.mark.parametrize("mode", ["eager", "replay", "capture", "warmup", "memory_profile"])
@pytest.mark.parametrize("active", [False, True])
def test_payload_round_trip_and_excluded_modes(mode, active):
    excluded = mode in ("capture", "warmup", "memory_profile")
    kwargs = dict(is_graph_capturing=mode == "capture", is_warmup=mode == "warmup",
                  is_graph_replaying=mode == "replay", is_profile=mode == "memory_profile")
    if active and excluded:
        with pytest.raises(ValueError, match="real serving"):
            AFDControlPayload({}, runtime_step_id=8, profile_generation="g", profile_active=True, **kwargs)
        return
    original = AFDControlPayload({0: AFDDPMetadata(torch.tensor([13, 64], dtype=torch.int32))},
        runtime_step_id=None if excluded else 8, profile_generation="g" if not excluded else None,
        profile_active=active, **kwargs)
    restored = decode_control_payload(encode_control_payload(original))
    assert restored.runtime_step_id == original.runtime_step_id
    assert restored.profile_generation == original.profile_generation
    assert restored.profile_active == original.profile_active
    assert torch.equal(restored.dp_metadata_list[0].num_tokens_across_dp_cpu,
                       original.dp_metadata_list[0].num_tokens_across_dp_cpu)


def test_old_wire_format_defaults_and_exact_nonprofiled_encoding():
    payload = AFDControlPayload({}, False, False)
    wire = json.loads(encode_control_payload(payload))
    assert "runtime_step_id" not in wire
    restored = decode_control_payload(json.dumps(wire).encode())
    assert restored.runtime_step_id is None and restored.profile_generation is None
    assert not restored.profile_active


def test_attention_rpc_and_send_payload_use_the_protocol(monkeypatch):
    from src import afd_profile_protocol
    monkeypatch.setenv("AFD_FFN_TIMING_DIR", "/unused")
    worker = object.__new__(AFDAttentionWorker)
    worker.model_runner = SimpleNamespace()
    assert worker.set_afd_cost_profile_state('{"generation":"g","active":true}')["active"]
    sent = []
    runner = object.__new__(AFDMetadataProviderMixin)
    runner._is_warmup = False
    runner._afd_is_profile = False
    runner.connector = SimpleNamespace(control_plane=SimpleNamespace(
        update_state_from_dp_metadata=lambda payload: None,
        send_dp_metadata_list=sent.append))
    monkeypatch.setattr(afd_profile_protocol, "coordinate_profile_step",
        lambda *a, **k: {"runtime_step_id": 123, "profile_generation": "g", "profile_active": True})
    runner.send_dp_metadata(AFDDPMetadata(torch.tensor([13, 64], dtype=torch.int32)), None)
    assert sent[0].runtime_step_id == 123 and sent[0].profile_active


@pytest.mark.parametrize("enabled", [True, False])
def test_startup_memory_profile_is_excluded_but_idle_dp_dummy_is_serving(monkeypatch, enabled):
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    from afd_plugin.v1.worker.attention_model_runner import AFDAttentionModelRunner
    if enabled:
        monkeypatch.setenv("AFD_FFN_TIMING_DIR", "/unused")
    else:
        monkeypatch.delenv("AFD_FFN_TIMING_DIR", raising=False)
    runner = object.__new__(AFDAttentionModelRunner)
    runner._afd_pending_metadata = None
    runner._afd_is_profile = False
    runner._afd_is_graph_capturing = False
    observed = []
    def dummy(self, *args):
        observed.append(self._afd_is_profile)
        return None, None
    monkeypatch.setattr(GPUModelRunner, "_dummy_run", dummy)
    runner._dummy_run(64, is_profile=True)
    runner._dummy_run(1, is_profile=False)
    assert observed == [enabled, False]
    assert not runner._afd_is_profile


def _gloo_rank(rank, rendezvous, output):
    import os
    import torch.distributed as dist
    from vllm.distributed.parallel_state import GroupCoordinator
    from src.afd_profile_protocol import AttentionProfileState, coordinate_profile_step, set_attention_profile_state

    os.environ["AFD_FFN_TIMING_DIR"] = "/unused"
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2)
    try:
        group = object.__new__(GroupCoordinator)
        group.world_size, group.rank_in_group = 2, rank
        group.ranks, group.cpu_group, group.mq_broadcaster = [0, 1], dist.group.WORLD, None
        runner = SimpleNamespace(vllm_config=SimpleNamespace(parallel_config=SimpleNamespace(tensor_parallel_size=1)),
            _afd_cost_profile_state=AttentionProfileState(next_runtime_step_id=0 if rank == 0 else 9999))
        values = []
        for generation in range(8):
            for active in (True, False):
                set_attention_profile_state(runner, json.dumps({"generation": f"g{generation}", "active": active}))
                values.append(coordinate_profile_step(runner, excluded=False, group=group))
        Path(output, f"rank-{rank}.json").write_text(json.dumps(values))
    finally:
        dist.destroy_process_group()


def test_real_two_process_native_gloo_broadcast(tmp_path):
    # Exercise the pinned GroupCoordinator implementation, not a fake counter.
    torch.multiprocessing.spawn(_gloo_rank, args=(str(tmp_path / "rendezvous"), str(tmp_path)), nprocs=2, join=True)
    a, b = [json.loads((tmp_path / f"rank-{rank}.json").read_text()) for rank in (0, 1)]
    assert a == b
    assert [row["runtime_step_id"] for row in a] == list(range(16))
