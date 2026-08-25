import pytest

from src.search import dvfs_sensitivity


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
