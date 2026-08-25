import pytest

from src.metrics import latency_metrics, percentile, summarize_run


def test_latency_and_tpot_convention():
    ttft, tpot, e2e = latency_metrics(10.0, 10.2, 11.0, 5)
    assert ttft == pytest.approx(200)
    assert tpot == pytest.approx(200)
    assert e2e == pytest.approx(1000)


@pytest.mark.parametrize("tokens", [0, 1])
def test_tpot_is_undefined_for_zero_or_one_token(tokens):
    ttft, tpot, _ = latency_metrics(1, 1.1, 1.2, tokens)
    assert ttft == pytest.approx(100)
    assert tpot is None


def test_percentile_uses_linear_interpolation():
    assert percentile([0, 10, 20, 30], 50) == 15
    assert percentile([0, 10, 20, 30], 95) == pytest.approx(28.5)
    assert percentile([], 99) is None


def test_run_summary():
    requests = [
        {"status": "ok", "ttft_ms": 10, "tpot_ms": 5, "e2e_ms": 100,
         "send_delay_ms": 1, "prompt_tokens": 100, "output_tokens": 10},
        {"status": "ok", "ttft_ms": 20, "tpot_ms": None, "e2e_ms": 50,
         "send_delay_ms": 3, "prompt_tokens": 120, "output_tokens": 1},
        {"status": "client_error", "send_delay_ms": 2},
    ]
    energy = {"total_gpu_energy_j": 44, "gpu0_energy_j": 11}
    result = summarize_run(run_id="x", workload="code", request_rate=2,
                           request_results=requests, measurement_duration_s=2,
                           energy_summary=energy)
    assert result["completed_requests"] == 2
    assert result["failed_requests"] == 1
    assert result["output_tokens_total"] == 11
    assert result["throughput_requests_s"] == 1
    assert result["joules_per_request"] == 22
    assert result["joules_per_output_token"] == 4
