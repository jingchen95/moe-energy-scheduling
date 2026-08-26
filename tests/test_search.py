import pytest

from src.search import (
    dvfs_sensitivity,
    enumerate_frequency_tuples,
    paired_comparison,
    pareto_frontier,
    select_joint_candidates,
    select_min_energy,
    slo_feasible,
    summarize_repeatability,
)


def test_dvfs_sensitivity_deltas():
    reference = {
        "ttft_p99_ms": 100, "tpot_p99_ms": 20, "e2e_p99_ms": 500,
        "total_gpu_energy_j": 1000, "gpu2_energy_j": 260,
        "throughput_requests_s": 2.0,
    }
    changed = {
        "ttft_p99_ms": 110, "tpot_p99_ms": 25, "e2e_p99_ms": 550,
        "total_gpu_energy_j": 950, "gpu2_energy_j": 210,
        "throughput_requests_s": 1.8,
    }
    result = dvfs_sensitivity(reference, changed, 2)
    assert result["delta_tpot_p99_ms"] == 5
    assert result["delta_total_energy_j"] == -50
    assert result["delta_gpu_energy_j"] == -50
    assert result["delta_throughput_requests_s"] == pytest.approx(-0.2)


def configuration(clocks, tpot, energy, ttft=100, label="x"):
    return {"frequency_tuple": clocks, "target_clocks_mhz": clocks,
            "tpot_p99_ms": tpot, "ttft_p99_ms": ttft, "e2e_p99_ms": 1000,
            "total_gpu_energy_j": energy, "joules_per_request": energy / 50,
            "throughput_requests_s": 0.5, "failed_requests": 0, "label": label}


def test_frequency_tuple_enumeration_is_complete_and_seeded():
    natural = enumerate_frequency_tuples()
    seeded = enumerate_frequency_tuples(seed=7)
    assert len(natural) == 81
    assert len(set(natural)) == 81
    assert set(natural) == set(seeded)
    assert natural != seeded
    assert seeded == enumerate_frequency_tuples(seed=7)


def test_slo_and_minimum_energy_selection():
    rows = [configuration((2520,) * 4, 11, 100),
            configuration((1635,) * 4, 12.5, 80),
            configuration((1005,) * 4, 16, 50)]
    assert slo_feasible(rows[0], ttft_limit_ms=500, tpot_limit_ms=12)
    assert not slo_feasible(rows[1], ttft_limit_ms=500, tpot_limit_ms=12)
    assert select_min_energy(rows, ttft_limit_ms=500,
                             tpot_limit_ms=13) is rows[1]


def test_pareto_frontier_removes_dominated_points():
    rows = [configuration((2520,) * 4, 10, 100, label="a"),
            configuration((1635,) * 4, 11, 90, label="b"),
            configuration((1005,) * 4, 12, 95, label="dominated")]
    assert [row["label"] for row in pareto_frontier(rows)] == ["a", "b"]


def test_joint_candidate_selection_is_deterministic_and_bounded():
    rows = []
    for index, clocks in enumerate(enumerate_frequency_tuples()):
        mean_clock = sum(clocks) / 4
        rows.append(configuration(clocks, 10 + (2520 - mean_clock) / 400,
                                  1000 + mean_clock, label=str(index)))
    first = select_joint_candidates(rows, tpot_limits_ms=(12, 13, 15),
                                    ttft_limit_ms=500, max_candidates=8)
    second = select_joint_candidates(rows, tpot_limits_ms=(12, 13, 15),
                                     ttft_limit_ms=500, max_candidates=8)
    assert 5 <= len(first) <= 8
    assert [row["frequency_tuple"] for row in first] == [
        row["frequency_tuple"] for row in second
    ]


def test_paired_comparison_requires_same_tuple():
    left = configuration((2520, 1635, 1635, 1005), 13, 100)
    right = configuration((2520, 1635, 1635, 1005), 12, 95)
    result = paired_comparison(left, right)
    assert result["delta_tpot_p99_ms"] == -1
    assert result["delta_total_gpu_energy_j"] == -5
    with pytest.raises(ValueError, match="identical"):
        paired_comparison(left, configuration((2520,) * 4, 12, 95))


def test_repeatability_summary_preserves_individual_feasibility():
    rows = [configuration((2520,) * 4, 12.5, 100, ttft=400),
            configuration((2520,) * 4, 13.1, 102, ttft=410),
            configuration((2520,) * 4, 12.7, 98, ttft=390)]
    result = summarize_repeatability(rows, ttft_limit_ms=500, tpot_limit_ms=13)
    assert result["trials"] == 3
    assert result["feasible_trials"] == 2
    assert not result["robustly_slo_feasible"]
    assert result["total_gpu_energy_j_mean"] == 100
