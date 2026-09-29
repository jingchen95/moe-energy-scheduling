import csv
import json

import pytest

from tools.validate_afd_smoke import validate_gpu_samples, validate_outputs


def _response(request_id: str, content: str) -> dict:
    return {
        "request": {"request_id": request_id},
        "response": {
            "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2},
        },
    }


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_deterministic_smoke_outputs(tmp_path):
    rows = [_response(f"r{i}", f"answer {i}") for i in range(3)]
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    _write_jsonl(first, rows)
    _write_jsonl(second, rows)
    assert len(validate_outputs(first, second)) == 3
    rows[1]["response"]["choices"][0]["message"]["content"] = "different"
    _write_jsonl(second, rows)
    with pytest.raises(ValueError, match="differ"):
        validate_outputs(first, second)


def test_gpu_samples_require_all_four_gpus(tmp_path):
    path = tmp_path / "samples.csv"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("timestamp_unix_s", "gpu", "memory_used_mib",
                        "sm_util_percent", "sm_clock_mhz", "power_w"),
        )
        writer.writeheader()
        for gpu in range(4):
            writer.writerow({
                "timestamp_unix_s": 1.0,
                "gpu": gpu,
                "memory_used_mib": 2048,
                "sm_util_percent": 50,
                "sm_clock_mhz": 2520,
                "power_w": 200,
            })
    assert set(validate_gpu_samples(path)) == {"0", "1", "2", "3"}
