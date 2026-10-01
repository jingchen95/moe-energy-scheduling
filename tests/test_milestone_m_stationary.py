import json
from pathlib import Path

from tools.analyze_milestone_m_rate_sweep import _request_trend
from tools.prepare_milestone_m_stationary import (
    RATE_SWEEP_RANDOMIZATION_SEED,
    RATE_SWEEP_RATES,
    RATE_SWEEP_REPETITIONS,
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
            "ttft_ms": 100 + index,
            "tpot_ms": 20 + index / 100,
        })
    path = tmp_path / "requests.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    result = _request_trend(path)

    assert result["late_minus_early_ttft_ms"] == 200
    assert result["ttft_slope_ms_per_request"] == 1
    assert result["late_minus_early_tpot_ms"] > 1.9
