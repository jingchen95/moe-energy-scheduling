import random
from pathlib import Path

from tools.analyze_milestone_m_symmetric_ffn_sweep import pareto_frequencies
from tools.prepare_milestone_m_symmetric_ffn_sweep import (
    RANDOMIZATION_SEED,
    REPETITIONS,
    REQUEST_RATE,
    SYMMETRIC_FREQUENCIES_MHZ,
)


def test_symmetric_sweep_definition_is_frozen() -> None:
    assert SYMMETRIC_FREQUENCIES_MHZ == (2520, 2220, 2070, 1635, 1005)
    assert REPETITIONS == (1, 2, 3)
    assert REQUEST_RATE == 8.0
    order = list(range(5))
    random.Random(RANDOMIZATION_SEED).shuffle(order)
    assert order == [3, 2, 4, 0, 1]


def test_launcher_conditions_each_frequency_and_keeps_placement_fixed() -> None:
    script = Path("scripts/milestone_m_symmetric_ffn_sweep.sbatch").read_text()
    assert "--override-arrival-rate" in script
    assert "conditioning" in script
    assert "--placement-name P_balanced" in script
    assert "src.gpu_control set-all" in script
    assert "expected_runs" in script
    assert "repository commit drift" in script
    assert "git diff --quiet" in script
    assert '"${config_dir}/server/ffn.log"' in script
    assert '"${study}/server/attention.log"' not in script
    assert 'afd_stop_process_group "${attention_pid}"' in script
    assert 'afd_stop_process_group "${ffn_pid}"' in script
    assert "P_MH" not in script
    assert "P_HM" not in script


def test_pareto_frequency_selection() -> None:
    rows = [
        {"frequency_mhz": 2520, "total_gpu_energy_j_mean": 120, "tpot_p99_ms_mean": 10},
        {"frequency_mhz": 2070, "total_gpu_energy_j_mean": 100, "tpot_p99_ms_mean": 11},
        {"frequency_mhz": 1635, "total_gpu_energy_j_mean": 105, "tpot_p99_ms_mean": 12},
        {"frequency_mhz": 1005, "total_gpu_energy_j_mean": 90, "tpot_p99_ms_mean": 18},
    ]
    assert pareto_frequencies(rows, "tpot_p99_ms") == {2520, 2070, 1005}
