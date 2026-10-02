import json
from pathlib import Path

from tools.analyze_milestone_m_rate_sweep import _request_trend
from tools.analyze_milestone_m_stationary_repeat import classify_fixed_dvfs
from tools.prepare_milestone_m_stationary import (
    RATE_SWEEP_RANDOMIZATION_SEED,
    RATE_SWEEP_RATES,
    RATE_SWEEP_REPETITIONS,
)
from tools.prepare_milestone_m_stationary_repeat import (
    RANDOMIZATION_SEED as REPEAT_RANDOMIZATION_SEED,
    REPETITIONS as REPEAT_REPETITIONS,
    SELECTED_REQUEST_RATE,
)


def test_rate_sweep_definition_is_fixed() -> None:
    assert RATE_SWEEP_RATES == (2.0, 4.0, 6.0, 8.0)
    assert RATE_SWEEP_REPETITIONS == (1, 2, 3)
    assert RATE_SWEEP_RANDOMIZATION_SEED == 20261002
    assert len(RATE_SWEEP_RATES) * len(RATE_SWEEP_REPETITIONS) == 12
    script = Path("scripts/milestone_m_rate_sweep.sbatch").read_text()
    assert "--override-arrival-rate" in script


def test_request_trend_detects_accumulating_ttft(tmp_path: Path) -> None:
    rows = []
    for index in range(300):
        rows.append({
            "status": "ok",
            "scheduled_send_s": index / 4,
            "actual_send_s": index / 4,
            "ttft_ms": 100 + index,
            "tpot_ms": 20 + index / 100,
            "e2e_ms": 1000,
        })
    path = tmp_path / "requests.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    result = _request_trend(path)

    assert result["late_minus_early_ttft_ms"] == 200
    assert result["ttft_slope_ms_per_request"] == 1
    assert result["late_minus_early_tpot_ms"] > 1.9
    assert result["completion_span_throughput_requests_s"] == 4


def test_stationary_repeat_is_frozen_at_eight_requests_per_second() -> None:
    assert SELECTED_REQUEST_RATE == 8.0
    assert REPEAT_REPETITIONS == (1, 2, 3, 4, 5)
    assert REPEAT_RANDOMIZATION_SEED == 20261003
    script = Path("scripts/milestone_m_stationary_repeat.sbatch").read_text()
    assert "--override-arrival-rate" in script
    assert "conditioning" in script
    assert "expected_runs" in script


def test_fixed_dvfs_classification_requires_both_orientations() -> None:
    supported = {
        name: {
            "total_gpu_energy_j_relative_change": -0.07,
            "tpot_p99_ms_relative_change": 0.02,
            "throughput_requests_s_relative_change": -0.01,
        }
        for name in ("fixed_dvfs_mh", "fixed_dvfs_hm")
    }
    assert classify_fixed_dvfs(supported)[0] == "supported_at_tested_pair"
    supported["fixed_dvfs_hm"]["tpot_p99_ms_relative_change"] = 0.05
    assert classify_fixed_dvfs(supported)[0] == "mixed"
