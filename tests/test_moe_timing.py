import pytest

from src.moe_timing import aggregate_timing_records


def test_timing_aggregation_splits_mixed_rows_and_uses_max_rank():
    result = aggregate_timing_records({
        0: [
            {"layer": "l0", "phase": "prefill", "token_rows": 4,
             "decode_rows": 0, "elapsed_ms": 2.0},
            {"layer": "l0", "phase": "mixed", "token_rows": 4,
             "decode_rows": 1, "elapsed_ms": 4.0},
        ],
        1: [{"layer": "l0", "phase": "decode", "token_rows": 2,
             "decode_rows": 2, "elapsed_ms": 5.0}],
    })
    assert result["record_count"] == 3
    assert result["max_rank_total_ms"] == pytest.approx(6.0)
    assert result["max_rank_prefill_ms"] == pytest.approx(5.0)
    assert result["max_rank_decode_ms"] == pytest.approx(5.0)
    assert result["per_layer"]["l0"]["total_ms"] == pytest.approx(11.0)
