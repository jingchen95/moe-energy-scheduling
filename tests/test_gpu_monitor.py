import pytest

from src.gpu_monitor import GPUSample, energy_delta_j, integrate_power_j, summarize_energy_window


def sample(gpu, mono, energy, power=100):
    return GPUSample(1000 + mono, mono, gpu, power, energy, 50, 20, 1500, 810, 55, 350)


def test_energy_delta_counter_units_and_monotonicity():
    assert energy_delta_j(1000, 2500) == 1.5
    with pytest.raises(ValueError, match="decreased"):
        energy_delta_j(2, 1)


def test_trapezoidal_power_integration():
    assert integrate_power_j([sample(0, 0, None, 100), sample(0, 2, None, 200)]) == 300


def test_counter_is_primary_and_summary_is_per_gpu():
    starts = [sample(gpu, 0, 1000) for gpu in range(4)]
    ends = [sample(gpu, 2, 3000 + gpu * 1000) for gpu in range(4)]
    diagnostics = [sample(gpu, 1, 2000, 100 + gpu) for gpu in range(4)]
    result = summarize_energy_window(starts, ends, diagnostics)
    assert result["energy_source"] == "nvml_total_energy_counter"
    assert result["gpu0_energy_j"] == 2
    assert result["gpu3_energy_j"] == 5
    assert result["total_gpu_energy_j"] == 14
    assert result["gpu2_avg_power_w"] == 102
