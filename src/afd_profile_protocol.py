"""Profiling-only shared Attention step identity and RPC window control."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any


PROFILER_SCHEMA_VERSION = 3
IDENTITY_SOURCE = "attention_dp_rank0_cpu_broadcast"


@dataclass
class AttentionProfileState:
    generation: str | None = None
    active: bool = False
    next_runtime_step_id: int = 0


def _state(runner: Any) -> AttentionProfileState:
    if not hasattr(runner, "_afd_cost_profile_state"):
        runner._afd_cost_profile_state = AttentionProfileState()
    return runner._afd_cost_profile_state


def set_attention_profile_state(runner: Any, encoded: str) -> dict[str, Any]:
    if not os.environ.get("AFD_FFN_TIMING_DIR", "").strip():
        raise RuntimeError("AFD cost protocol is not enabled in this launcher")
    value = json.loads(encoded)
    generation, active = value.get("generation"), value.get("active")
    if not isinstance(generation, str) or not generation or any(c.isspace() for c in generation):
        raise ValueError("profile generation must be a non-empty token")
    if type(active) is not bool:
        raise ValueError("profile active must be boolean")
    state = _state(runner)
    state.generation, state.active = generation, active
    # Never reset runtime identity at a window boundary.
    return {"generation": generation, "active": active,
            "schema_version": PROFILER_SCHEMA_VERSION}


def coordinate_profile_step(runner: Any, *, excluded: bool, group: Any = None) -> dict[str, Any]:
    """Called once per existing synchronized Attention DP control send.

    Only DP rank 0 allocates identity. Both senders carry its broadcast value,
    not independent transaction/call counters. This CPU collective precedes
    AFD control transmission and all timed CUDA intervals.
    """
    if not os.environ.get("AFD_FFN_TIMING_DIR", "").strip() or excluded:
        return {}
    if group is None:
        from vllm.distributed.parallel_state import get_dp_group

        group = get_dp_group()
    if group.world_size != 2 or int(runner.vllm_config.parallel_config.tensor_parallel_size) != 1:
        raise RuntimeError("cost-profile protocol supports only Attention DP=2 / TP=1")
    state = _state(runner)
    value = None
    if group.rank_in_group == 0:
        value = {
            "runtime_step_id": state.next_runtime_step_id,
            "profile_generation": state.generation,
            "profile_active": state.active,
        }
        state.next_runtime_step_id += 1
    value = group.broadcast_object(value, src=0)
    if not isinstance(value, dict) or type(value.get("runtime_step_id")) is not int:
        raise RuntimeError("invalid shared Attention runtime step identity")
    if value["runtime_step_id"] < 0:
        raise RuntimeError("negative shared runtime step identity")
    return value


async def set_profile_state_rpc(base_url: str, generation: str, active: bool) -> list[Any]:
    from src.benchmark import collective_rpc

    expected = {"generation": generation, "active": active,
                "schema_version": PROFILER_SCHEMA_VERSION}
    results = await collective_rpc(
        base_url, "set_afd_cost_profile_state",
        args=[json.dumps({"generation": generation, "active": active})],
    )
    # Pinned DPLBAsyncMPClient awaits every engine's collective RPC, then returns
    # the first engine's result. An error on any engine propagates to the client.
    if not results or any(result != expected for result in results):
        raise RuntimeError(f"profile-state RPC acknowledgement mismatch: {results}")
    return results
