import json

import pytest

from src.workload import TraceRequest, load_trace, scheduled_monotonic_times, validate_trace


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
