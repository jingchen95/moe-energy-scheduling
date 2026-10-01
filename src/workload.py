"""Canonical deterministic semantic workload traces."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable


CATEGORIES = frozenset({"general", "code", "math"})


@dataclass(frozen=True)
class TraceRequest:
    request_id: str
    category: str
    prompt: str
    prompt_tokens: int
    max_output_tokens: int
    arrival_offset_s: float
    dataset_source: str | None = None
    original_sample_id: str | None = None
    seed: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.request_id:
            raise ValueError("request_id must be non-empty")
        if self.category not in CATEGORIES:
            raise ValueError(f"unsupported category {self.category!r}")
        if not self.prompt.strip():
            raise ValueError(f"request {self.request_id}: prompt must be non-empty")
        if self.prompt_tokens <= 0:
            raise ValueError(f"request {self.request_id}: prompt_tokens must be positive")
        if self.max_output_tokens <= 0:
            raise ValueError(f"request {self.request_id}: max_output_tokens must be positive")
        if not math.isfinite(self.arrival_offset_s) or self.arrival_offset_s < 0:
            raise ValueError(f"request {self.request_id}: invalid arrival_offset_s")
        if self.seed < 0:
            raise ValueError(f"request {self.request_id}: seed must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TraceRequest":
        required = {
            "request_id", "category", "prompt", "prompt_tokens",
            "max_output_tokens", "arrival_offset_s",
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
            dataset_source=(None if value.get("dataset_source") is None
                            else str(value["dataset_source"])),
            original_sample_id=(None if value.get("original_sample_id") is None
                                else str(value["original_sample_id"])),
            seed=int(value.get("seed", 0)),
            metadata=dict(value.get("metadata") or {}),
        )
        request.validate()
        return request


def load_trace(path: Path) -> list[TraceRequest]:
    requests: list[TraceRequest] = []
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
            requests.append(TraceRequest.from_dict(value))
    validate_trace(requests)
    return requests


def validate_trace(requests: Iterable[TraceRequest]) -> None:
    rows = list(requests)
    if not rows:
        raise ValueError("trace has no requests")
    seen: set[str] = set()
    previous_offset = -1.0
    categories: set[str] = set()
    for request in rows:
        request.validate()
        if request.request_id in seen:
            raise ValueError(f"duplicate request_id {request.request_id!r}")
        if request.arrival_offset_s < previous_offset:
            raise ValueError("trace must be ordered by nondecreasing arrival_offset_s")
        seen.add(request.request_id)
        categories.add(request.category)
        previous_offset = request.arrival_offset_s
    if len(categories) != 1:
        raise ValueError(f"one trace must contain exactly one category, found {sorted(categories)}")


def write_trace(path: Path, requests: Iterable[TraceRequest]) -> None:
    rows = list(requests)
    validate_trace(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        for request in rows:
            stream.write(json.dumps(request.to_dict(), sort_keys=True) + "\n")


def scheduled_monotonic_times(
    requests: Iterable[TraceRequest], start_monotonic_s: float,
) -> list[float]:
    """Return immutable open-loop deadlines derived from one replay epoch."""
    return [start_monotonic_s + request.arrival_offset_s for request in requests]


def with_constant_arrival_rate(
    requests: Iterable[TraceRequest], request_rate: float,
) -> list[TraceRequest]:
    """Preserve request content/order while replacing the open-loop schedule."""
    if not math.isfinite(request_rate) or request_rate <= 0:
        raise ValueError("request_rate must be finite and positive")
    rows = list(requests)
    validate_trace(rows)
    scheduled = [
        replace(request, arrival_offset_s=index / request_rate)
        for index, request in enumerate(rows)
    ]
    validate_trace(scheduled)
    return scheduled
