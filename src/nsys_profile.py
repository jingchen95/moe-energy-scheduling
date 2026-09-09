"""Capture a short, fixed vLLM Nsight Systems profiling window."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiohttp

from src.benchmark import (
    BASE_URL,
    MODEL,
    REQUEST_TIMEOUT_SECONDS,
    SETTLE_SECONDS,
    TRACE_PATH,
    TRACE_SHA256,
    WARMUP_REQUESTS,
    _sha256,
    replay_trace,
    warm_up,
)
from src.metrics import active_request_time_ms
from src.workload import load_trace


PROFILE_REQUESTS = 3


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


async def _profile_control(action: str) -> dict[str, Any]:
    if action not in {"start", "stop"}:
        raise ValueError(f"unsupported profiler action: {action}")
    timeout = aiohttp.ClientTimeout(total=300)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(f"{BASE_URL}/{action}_profile") as response:
            body = await response.text()
            if response.status != 200:
                raise RuntimeError(
                    f"{action}_profile failed: HTTP {response.status}: {body}"
                )
            return {"http_status": response.status, "body": body}


async def run_profile(output_dir: Path) -> Path:
    """Warm up normally, then profile three requests from the frozen trace."""
    trace_sha256 = _sha256(TRACE_PATH)
    if trace_sha256 != TRACE_SHA256:
        raise RuntimeError(f"unexpected trace SHA-256: {trace_sha256}")

    all_requests = load_trace(TRACE_PATH)
    requests = all_requests[:PROFILE_REQUESTS]
    if len(requests) != PROFILE_REQUESTS:
        raise RuntimeError(f"trace has fewer than {PROFILE_REQUESTS} requests")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(exist_ok=False)
    shutil.copyfile(TRACE_PATH, output_dir / "trace.jsonl")
    with (output_dir / "profile_trace.jsonl").open(
        "x", encoding="utf-8"
    ) as stream:
        with TRACE_PATH.open(encoding="utf-8") as source:
            for index, line in enumerate(source):
                if index == PROFILE_REQUESTS:
                    break
                stream.write(line)

    _write_json(
        output_dir / "profile_config.json",
        {
            "created_at": _utc_now(),
            "trace": str(TRACE_PATH),
            "trace_sha256": trace_sha256,
            "profile_request_ids": [request.request_id for request in requests],
            "profile_request_count": len(requests),
            "profile_arrival_offsets_s": [
                request.arrival_offset_s for request in requests
            ],
            "base_url": BASE_URL,
            "model": MODEL,
            "warmup_requests": WARMUP_REQUESTS,
            "settle_seconds": SETTLE_SECONDS,
            "request_timeout_seconds": REQUEST_TIMEOUT_SECONDS,
            "purpose": "diagnostic CUDA Graph kernel-time attribution",
            "run_scope": os.environ.get("MOE_PROFILE_RUN_SCOPE", "unknown"),
            "not_for_primary_latency": True,
        },
    )

    print("Warming up eight requests...", flush=True)
    await warm_up(all_requests)
    await asyncio.sleep(SETTLE_SECONDS)

    start_result = await _profile_control("start")
    _write_json(output_dir / "profile_control.json", {"start": start_result})
    print("Profiling three fixed requests...", flush=True)
    measurement_started_at = _utc_now()
    replay_epoch_s = time.monotonic()
    try:
        rows = await replay_trace(requests, replay_epoch_s=replay_epoch_s)
        # Preserve responses even if profiler shutdown/export subsequently fails.
        with (output_dir / "requests.jsonl").open("x", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
    finally:
        stop_result = await _profile_control("stop")
    measurement_completed_at = _utc_now()

    _write_json(
        output_dir / "profile_control.json",
        {"start": start_result, "stop": stop_result},
    )
    failed_count = sum(row["status"] != "ok" for row in rows)
    metrics = {
        "status": "ok" if failed_count == 0 else "invalid_failed_requests",
        "measurement_started_at": measurement_started_at,
        "measurement_completed_at": measurement_completed_at,
        "submitted_requests": len(rows),
        "completed_requests": len(rows) - failed_count,
        "failed_requests": failed_count,
        "run_scope": os.environ.get("MOE_PROFILE_RUN_SCOPE", "unknown"),
        "active_request_wall_time_ms_diagnostic": active_request_time_ms(rows),
        "warning": (
            "Profiler-run latency is diagnostic only; obtain primary E2E, TTFT, "
            "and TPOT from an otherwise identical unprofiled run."
        ),
    }
    _write_json(output_dir / "metrics.json", metrics)
    print(json.dumps({"output_dir": str(output_dir), "metrics": metrics}))

    if failed_count:
        raise RuntimeError(f"{failed_count} profiled requests failed")
    return output_dir


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m src.nsys_profile OUTPUT_DIR")
    asyncio.run(run_profile(Path(sys.argv[1])))


if __name__ == "__main__":
    main()
