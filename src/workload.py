"""Load and schedule a deterministic JSONL request trace."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class TraceRequest:
    request_id: str
    category: str
    prompt: str
    prompt_tokens: int
    max_output_tokens: int
    arrival_offset_s: float
    seed: int = 0

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TraceRequest":
        required = {
            "request_id",
            "category",
            "prompt",
            "prompt_tokens",
            "max_output_tokens",
            "arrival_offset_s",
        }
        missing = required.difference(value)
        if missing:
            raise ValueError(f"trace row missing fields: {sorted(missing)}")
        request = cls(
            request_id=str(value["request_id"]),
            category=str(value["category"]),
            prompt=str(value["prompt"]),
            prompt_tokens=int(value["prompt_tokens"]),
            max_output_tokens=int(value["max_output_tokens"]),
            arrival_offset_s=float(value["arrival_offset_s"]),
            seed=int(value.get("seed", 0)),
        )
        request.validate()
        return request

    def validate(self) -> None:
        if not self.request_id:
            raise ValueError("request_id must be non-empty")
        if not self.category:
            raise ValueError(f"request {self.request_id}: category must be non-empty")
        if not self.prompt.strip():
            raise ValueError(f"request {self.request_id}: prompt must be non-empty")
        if self.prompt_tokens <= 0:
            raise ValueError(f"request {self.request_id}: prompt_tokens must be positive")
        if self.max_output_tokens <= 0:
            raise ValueError(
                f"request {self.request_id}: max_output_tokens must be positive"
            )
        if not math.isfinite(self.arrival_offset_s) or self.arrival_offset_s < 0:
            raise ValueError(f"request {self.request_id}: invalid arrival_offset_s")
        if self.seed < 0:
            raise ValueError(f"request {self.request_id}: seed must be non-negative")


def validate_trace(requests: Iterable[TraceRequest]) -> list[TraceRequest]:
    rows = list(requests)
    if not rows:
        raise ValueError("trace has no requests")

    seen: set[str] = set()
    previous_offset = -1.0
    for request in rows:
        request.validate()
        if request.request_id in seen:
            raise ValueError(f"duplicate request_id {request.request_id!r}")
        if request.arrival_offset_s < previous_offset:
            raise ValueError("trace must be ordered by nondecreasing arrival_offset_s")
        seen.add(request.request_id)
        previous_offset = request.arrival_offset_s
    return rows


def load_trace(path: Path) -> list[TraceRequest]:
    rows: list[TraceRequest] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(TraceRequest.from_dict(value))
    return validate_trace(rows)


def scheduled_monotonic_times(
    requests: Iterable[TraceRequest], start_monotonic_s: float
) -> list[float]:
    """Translate fixed trace offsets to deadlines from one replay epoch."""
    return [start_monotonic_s + row.arrival_offset_s for row in requests]
