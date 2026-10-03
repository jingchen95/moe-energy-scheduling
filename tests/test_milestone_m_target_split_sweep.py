import random
from pathlib import Path

from tools.analyze_milestone_m_target_split_sweep import pareto_points
from tools.prepare_milestone_m_target_split_sweep import (
    ORIENTATIONS,
    RANDOMIZATION_SEED,
    REPETITIONS,
    REQUEST_RATE,
)


def test_target_split_matrix_is_fixed_and_randomized() -> None:
    assert ORIENTATIONS == {
        "MH": {"ffn_clocks_mhz": [1635, 2070], "slow_rank": 0},
        "HM": {"ffn_clocks_mhz": [2070, 1635], "slow_rank": 1},
    }
    assert REQUEST_RATE == 8.0
    assert REPETITIONS == (1, 2, 3)
    order = list(range(10))
    random.Random(RANDOMIZATION_SEED).shuffle(order)
    assert order == [5, 2, 6, 7, 0, 1, 3, 9, 8, 4]
    assert len(order) * len(REPETITIONS) == 30


def test_target_split_launcher_preserves_experiment_boundaries() -> None:
    script = Path("scripts/milestone_m_target_split_sweep.sbatch").read_text()
    assert "repository commit drift" in script
    assert "--override-arrival-rate" in script
    assert "conditioning" in script
    assert 'afd_stop_process_group "${attention_pid}"' in script
    assert 'afd_stop_process_group "${ffn_pid}"' in script
    assert "frequency/H" not in script
    assert "routing-profile" not in script


def test_combined_frontier_keeps_only_non_dominated_points() -> None:
    rows = [
        {"point": "symmetric", "total_gpu_energy_j_mean": 100, "tpot_p99_ms_mean": 10},
        {"point": "heterogeneous", "total_gpu_energy_j_mean": 90, "tpot_p99_ms_mean": 11},
        {"point": "dominated", "total_gpu_energy_j_mean": 105, "tpot_p99_ms_mean": 12},
    ]
    assert pareto_points(rows, "tpot_p99_ms") == {"symmetric", "heterogeneous"}
