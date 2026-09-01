from __future__ import annotations

from tools.analyze_milestone_i import (
    aggregate_repetitions,
    classify_opportunity,
    pareto_frontier,
    select_oracles,
)
from tools.prepare_milestone_i import _runtime_placement_name, targeted_frequency_tuples


def _rep(placement: str, clocks: tuple[int, ...], repetition: int,
         energy: float, tpot: float, ttft: float = 100.0) -> dict:
    return {
        "placement": placement, "placement_name": placement,
        "placement_sha256": placement, "frequency_tuple": list(clocks),
        "repetition": repetition, "trace_sha256": "trace", "failed_requests": 0,
        "total_gpu_energy_j": energy, "joules_per_request": energy / 300,
        "ttft_p99_ms": ttft, "tpot_p99_ms": tpot, "tpot_p50_ms": tpot - 1,
        "tpot_mean_ms": tpot - 0.5, "throughput_requests_s": 2.0,
    }


def test_targeted_frequency_set_is_exact() -> None:
    rows = targeted_frequency_tuples()
    assert len(rows) == len(set(rows)) == 11
    assert (2520,) * 4 in rows
    assert (1635,) * 4 in rows
    assert (1005,) * 4 in rows
    assert sum(row.count(1005) == 1 and row.count(1635) == 3 for row in rows) == 4
    assert sum(row.count(1635) == 1 and row.count(1005) == 3 for row in rows) == 4


def test_legacy_p1_plan_gets_runtime_label() -> None:
    assert _runtime_placement_name({"policy": "P1", "name": "upstream_eplb"}) == (
        "upstream_default_eplb"
    )


def test_aggregate_repetitions_reports_sample_std() -> None:
    rows = [_rep("P1", (2520,) * 4, i, 100.0 + 10 * i, 10.0 + i) for i in range(3)]
    result = aggregate_repetitions(rows)[0]
    assert result["total_gpu_energy_j_mean"] == 110.0
    assert result["total_gpu_energy_j_std"] == 10.0
    assert result["tpot_p99_ms_mean"] == 11.0


def test_pareto_and_oracle_use_measured_means() -> None:
    rows = []
    for placement, clocks, energy, tpot in [
        ("P1", (2520,) * 4, 120.0, 10.0),
        ("P1", (1635,) * 4, 90.0, 12.0),
        ("P2", (2520,) * 4, 125.0, 11.0),
        ("P2", (1635,) * 4, 80.0, 11.5),
    ]:
        rows.extend(_rep(placement, clocks, i, energy, tpot) for i in range(3))
    aggregate = aggregate_repetitions(rows)
    frontier = pareto_frontier(aggregate)
    assert {(row["placement"], tuple(row["frequency_tuple"])) for row in frontier} == {
        ("P1", (2520,) * 4), ("P2", (1635,) * 4),
    }
    sequential, oracle = select_oracles(aggregate, [11.0, 12.0])
    assert sequential == "P1"
    assert oracle[0]["joint_placement"] == "P1"
    assert oracle[1]["joint_placement"] == "P2"
    assert oracle[1]["energy_saving_pct"] > 11.0


def test_decision_gate_preserves_small_negative_result() -> None:
    rows = [{"feasible": True, "tpot_slo_ms": 10.0, "energy_saving_pct": 0.8},
            {"feasible": True, "tpot_slo_ms": 11.0, "energy_saving_pct": 0.2}]
    assert classify_opportunity(rows)["decision"] == "C_NOT_PROMISING"
