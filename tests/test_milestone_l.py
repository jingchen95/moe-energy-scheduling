from tools.analyze_rank_permutation_dvfs_alignment import H, M
from tools.prepare_milestone_l import (
    CONFIGURATIONS,
    construct_aligned_placement,
    experiment_configurations,
    validate_intact_group_permutation,
)
from tools.analyze_milestone_l import (
    aggregate_runs,
    classify_result,
    compare_aggregates,
)


def _mapping() -> list[list[int]]:
    return [[group for group in range(4) for _ in range(2)] for _ in range(3)]


def _counts() -> list[list[int]]:
    return [[1, 2, 3, 4, 5, 6, 7, 8] for _ in range(3)]


def test_aligned_placement_preserves_intact_groups_and_cardinality():
    original = _mapping()
    placement = construct_aligned_placement(
        original, _counts(), (M, H, H, H), name="aligned"
    )
    observed = validate_intact_group_permutation(
        original, placement["logical_to_rank"], experts_per_group=2
    )
    assert observed == placement["metadata"]["group_at_gpu_per_layer"]
    assert all(sorted(row) == [0, 1, 2, 3] for row in observed)
    assert all(
        row.count(gpu) == 2
        for row in placement["logical_to_rank"] for gpu in range(4)
    )


def test_mhhh_places_lightest_intact_group_on_gpu0():
    placement = construct_aligned_placement(
        _mapping(), _counts(), (M, H, H, H), name="mhhh"
    )
    assert all(
        row[0] == 0 for row in placement["metadata"]["group_at_gpu_per_layer"]
    )


def test_mmhh_places_two_lightest_groups_on_slow_gpus():
    placement = construct_aligned_placement(
        _mapping(), _counts(), (M, M, H, H), name="mmhh"
    )
    assert all(
        set(row[:2]) == {0, 1}
        for row in placement["metadata"]["group_at_gpu_per_layer"]
    )
    assert all(
        set(row[2:]) == {2, 3}
        for row in placement["metadata"]["group_at_gpu_per_layer"]
    )


def test_experiment_matrix_is_exact_and_reproducible():
    first = experiment_configurations()
    second = experiment_configurations()
    assert first == second
    assert len(first) == len(CONFIGURATIONS) == 7
    assert {row["config_id"] for row in first} == {
        f"C{index}" for index in range(1, 8)
    }
    assert {row["task_index"] for row in first} == set(range(7))
    assert all(
        {run["repetition"] for run in row["runs"]} == {0, 1, 2}
        for row in first
    )
    assert sum(len(row["runs"]) for row in first) == 21


def _run(config_id: str, repetition: int, energy: float, tpot: float) -> dict:
    return {
        "config_id": config_id,
        "placement_id": "p",
        "placement_name": "p",
        "placement_sha256": "hash",
        "frequency_tuple": "1635|2520|2520|2520",
        "trace_sha256": "trace",
        "repetition": repetition,
        "completed_requests": 300,
        "failed_requests": 0,
        "ttft_p99_ms": 100.0 + repetition,
        "tpot_p99_ms": tpot,
        "tpot_p50_ms": tpot - 1,
        "tpot_mean_ms": tpot - 0.5,
        "e2e_p99_ms": 1000.0,
        "total_gpu_energy_j": energy,
        "joules_per_request": energy / 300,
        "throughput_requests_s": 15.0,
        "client_send_delay_p99_ms": 1.0,
        **{
            f"gpu{gpu}_{name}": value
            for gpu in range(4)
            for name, value in (
                ("energy_j", energy / 4), ("avg_power_w", 200.0),
                ("avg_sm_util", 80.0), ("actual_clock_mhz", 2000.0),
            )
        },
    }


def test_aggregate_and_comparison_report_all_three_values():
    rows = []
    for config_index in range(1, 8):
        config_id = f"C{config_index}"
        for repetition in range(3):
            rows.append(_run(config_id, repetition, 100.0 + config_index, 10.0 + config_index))
    aggregate = aggregate_runs(rows)
    assert len(aggregate) == 7
    c4 = next(row for row in aggregate if row["config_id"] == "C4")
    assert len(c4["tpot_p99_ms_values"].split("|")) == 3
    comparisons = compare_aggregates(aggregate)
    assert len(comparisons) == 4
    mhhh = next(row for row in comparisons if row["comparison"] == "A_MHHH")
    assert mhhh["original_config"] == "C4"
    assert mhhh["aligned_config"] == "C5"


def test_decision_preserves_no_effect_and_non_neutral_control():
    rows = []
    for name in ("A_MHHH", "B_MMHH", "C_HHHH_align_MHHH"):
        row = {"comparison": name}
        for metric in ("ttft_p99_ms", "tpot_p99_ms", "total_gpu_energy_j"):
            row[f"aligned_vs_original_{metric}_pct"] = 0.4
            row[f"{metric}_benefit_pct"] = -0.4
        rows.append(row)
    assert classify_result(rows)["decision"] == "C_NO_MEASURED_EFFECT"
    rows[-1]["aligned_vs_original_tpot_p99_ms_pct"] = 6.0
    assert classify_result(rows)["decision"] == "D_HHHH_CONTROL_NOT_NEUTRAL"
