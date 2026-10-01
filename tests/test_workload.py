import json

import pytest

from src.workload import (
    TraceRequest,
    load_trace,
    scheduled_monotonic_times,
    validate_trace,
    with_constant_arrival_rate,
)


def request(identifier="r0", offset=0.0):
    return TraceRequest(identifier, "math", "What is 2 + 2?", 8, 128, offset)


def test_load_trace_preserves_order_and_fields(tmp_path):
    path = tmp_path / "trace.jsonl"
    rows = [request("r0", 0.0), request("r1", 0.25)]
    path.write_text("".join(json.dumps(row.to_dict()) + "\n" for row in rows))
    assert load_trace(path) == rows


def test_trace_rejects_reordering_and_duplicates():
    with pytest.raises(ValueError, match="ordered"):
        validate_trace([request("a", 1), request("b", 0)])
    with pytest.raises(ValueError, match="duplicate"):
        validate_trace([request("a", 0), request("a", 1)])


def test_arrival_schedule_uses_one_fixed_epoch():
    rows = [request("a", 0), request("b", 0.125), request("c", 2.0)]
    assert scheduled_monotonic_times(rows, 10.0) == [10.0, 10.125, 12.0]


def test_constant_rate_override_preserves_content_and_replaces_offsets():
    rows = [request("a", 0), request("b", 0.125), request("c", 2.0)]

    result = with_constant_arrival_rate(rows, 4.0)

    assert [row.request_id for row in result] == ["a", "b", "c"]
    assert [row.prompt for row in result] == [row.prompt for row in rows]
    assert [row.arrival_offset_s for row in result] == [0.0, 0.25, 0.5]
    assert [row.arrival_offset_s for row in rows] == [0, 0.125, 2.0]


@pytest.mark.parametrize("rate", [0, -1, float("inf"), float("nan")])
def test_constant_rate_override_rejects_invalid_rate(rate):
    with pytest.raises(ValueError, match="finite and positive"):
        with_constant_arrival_rate([request()], rate)
