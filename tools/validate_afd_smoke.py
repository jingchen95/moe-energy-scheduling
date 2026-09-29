#!/usr/bin/env python3
"""Validate deterministic outputs and four-GPU use for the AFD smoke run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _semantic_response(row: dict[str, Any]) -> dict[str, Any]:
    response = row["response"]
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise ValueError("smoke response must contain exactly one choice")
    choice = choices[0]
    message = choice.get("message") or {}
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("smoke response contains no generated text")
    usage = response.get("usage") or {}
    return {
        "request_id": row["request"]["request_id"],
        "content": content,
        "finish_reason": choice.get("finish_reason"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
    }


def validate_outputs(first: Path, second: Path) -> list[dict[str, Any]]:
    first_rows = [_semantic_response(row) for row in _load_jsonl(first)]
    second_rows = [_semantic_response(row) for row in _load_jsonl(second)]
    if len(first_rows) != 3 or len(second_rows) != 3:
        raise ValueError("expected three requests in each deterministic pass")
    if first_rows != second_rows:
        raise ValueError("greedy AFD smoke outputs differ between identical passes")
    return first_rows


def validate_gpu_samples(path: Path) -> dict[str, dict[str, float]]:
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    per_gpu: dict[str, dict[str, float]] = {}
    for gpu in range(4):
        selected = [row for row in rows if int(row["gpu"]) == gpu]
        if not selected:
            raise ValueError(f"GPU {gpu} has no monitor samples")
        max_memory = max(float(row["memory_used_mib"]) for row in selected)
        max_util = max(float(row["sm_util_percent"]) for row in selected)
        if max_memory < 1024:
            raise ValueError(f"GPU {gpu} was not used by the AFD smoke ({max_memory} MiB)")
        per_gpu[str(gpu)] = {
            "max_memory_used_mib": max_memory,
            "max_sm_util_percent": max_util,
        }
    return per_gpu


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument("--gpu-samples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    responses = validate_outputs(args.first, args.second)
    record = {
        "status": "pass",
        "deterministic": True,
        "request_count_per_pass": len(responses),
        "response_sha256": hashlib.sha256(
            json.dumps(responses, sort_keys=True).encode()
        ).hexdigest(),
        "responses": responses,
        "gpu_usage": validate_gpu_samples(args.gpu_samples),
    }
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
