import csv
import json
from pathlib import Path

import pytest

from tools.validate_afd_smoke import validate_gpu_samples, validate_outputs


REPO_ROOT = Path(__file__).resolve().parents[1]


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


def test_afd_readiness_matches_official_attention_api_contract():
    common = (REPO_ROOT / "scripts" / "afd_common.sh").read_text()
    launcher = (
        REPO_ROOT / "scripts" / "milestone_m_afd_validation.sbatch"
    ).read_text()
    assert "afd_wait_attention_api()" in common
    assert 'curl --fail --silent "${attention_models_url}"' in common
    assert "http://127.0.0.1:18100/v1/models" in launcher
    assert "http://127.0.0.1:18101/v1/models" not in launcher
    assert "AFD FFN EngineCore started; workers run connector loop." in launcher
