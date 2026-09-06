"""Replay one fixed trace against vLLM and report the FusedMoE time share."""

from __future__ import annotations

import asyncio
import hashlib
import json
import shutil
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiohttp

from src.metrics import active_request_time_ms, latency_metrics
from src.moe_timing import fused_moe_time_share_percent, load_timing_files
from src.workload import TraceRequest, load_trace, scheduled_monotonic_times


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
# 固定基准参数；正常运行时只需要传入一个新的输出目录。
TRACE_PATH = REPOSITORY_ROOT / "data/traces/gsm8k-50-r0.5.jsonl"
TRACE_SHA256 = "721638a6250b929de05c2c355c8bc9b20258db7b28fedb94acb5572136571cbc"
BASE_URL = "http://127.0.0.1:8000"
MODEL = "qwen3-30b-a3b"
EXPECTED_RANKS = 4
WARMUP_REQUESTS = 8
SETTLE_SECONDS = 5.0
REQUEST_TIMEOUT_SECONDS = 900.0


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _request_result(
    request: TraceRequest,
    *,
    scheduled_s: float,
    replay_epoch_s: float,
    sent_s: float,
    completed_s: float,
    first_token_s: float | None,
    prompt_tokens: int | None,
    output_tokens: int | None,
    output_text: str,
    status: str,
    error: str | None,
) -> dict[str, Any]:
    ttft_ms, tpot_ms, e2e_ms = latency_metrics(
        sent_s, first_token_s, completed_s, output_tokens
    )
    return {
        "request_id": request.request_id,
        "category": request.category,
        "scheduled_send_s": scheduled_s - replay_epoch_s,
        "actual_send_s": sent_s - replay_epoch_s,
        "send_delay_ms": (sent_s - scheduled_s) * 1000.0,
        "ttft_ms": ttft_ms,
        "tpot_ms": tpot_ms,
        "tpot_valid": tpot_ms is not None,
        "tpot_convention": "(completion-first_token)/(output_tokens-1)",
        "e2e_ms": e2e_ms,
        "prompt_tokens": request.prompt_tokens,
        "server_prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "output_text": output_text,
        "status": status,
        "error": error,
    }


async def _send_request(
    session: aiohttp.ClientSession,
    *,
    request: TraceRequest,
    scheduled_s: float,
    replay_epoch_s: float,
) -> dict[str, Any]:
    delay_s = scheduled_s - time.monotonic()
    if delay_s > 0:
        await asyncio.sleep(delay_s)

    sent_s = time.monotonic()
    first_token_s: float | None = None
    prompt_tokens: int | None = None
    output_tokens: int | None = None
    output_chunks: list[str] = []
    try:
        payload = {
            "model": MODEL,
            "messages": [{"role": "user", "content": request.prompt}],
            "max_tokens": request.max_output_tokens,
            "temperature": 0.0,
            "seed": request.seed,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        async with session.post(
            f"{BASE_URL}/v1/chat/completions",
            json=payload,
            headers={"X-Request-Id": request.request_id},
        ) as response:
            if response.status != 200:
                detail = (await response.text())[:500]
                return _request_result(
                    request,
                    scheduled_s=scheduled_s,
                    replay_epoch_s=replay_epoch_s,
                    sent_s=sent_s,
                    completed_s=time.monotonic(),
                    first_token_s=None,
                    prompt_tokens=None,
                    output_tokens=None,
                    output_text="",
                    status=f"http_{response.status}",
                    error=detail,
                )

            async for raw_line in response.content:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    event = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if event.get("error") is not None:
                    raise RuntimeError(f"server stream error: {event['error']}")

                usage = event.get("usage")
                if isinstance(usage, dict):
                    if usage.get("prompt_tokens") is not None:
                        prompt_tokens = int(usage["prompt_tokens"])
                    if usage.get("completion_tokens") is not None:
                        output_tokens = int(usage["completion_tokens"])

                for choice in event.get("choices") or []:
                    delta = choice.get("delta") or {}
                    text = (delta.get("reasoning_content") or "") + (
                        delta.get("content") or ""
                    )
                    if text and first_token_s is None:
                        first_token_s = time.monotonic()
                    if text:
                        output_chunks.append(text)

        status = "ok"
        error = None
        if first_token_s is None:
            status = "invalid_response"
            error = "response contained no generated text"
        elif output_tokens is None or output_tokens <= 1:
            status = "invalid_response"
            error = "response did not contain a usable completion token count"

        return _request_result(
            request,
            scheduled_s=scheduled_s,
            replay_epoch_s=replay_epoch_s,
            sent_s=sent_s,
            completed_s=time.monotonic(),
            first_token_s=first_token_s,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            output_text="".join(output_chunks),
            status=status,
            error=error,
        )
    except Exception as exc:
        return _request_result(
            request,
            scheduled_s=scheduled_s,
            replay_epoch_s=replay_epoch_s,
            sent_s=sent_s,
            completed_s=time.monotonic(),
            first_token_s=first_token_s,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            output_text="".join(output_chunks),
            status="client_error",
            error=repr(exc),
        )


async def replay_trace(
    requests: list[TraceRequest],
    *,
    replay_epoch_s: float | None = None,
) -> list[dict[str, Any]]:
    """Replay all requests open-loop from their immutable trace offsets."""
    epoch = time.monotonic() if replay_epoch_s is None else replay_epoch_s
    deadlines = scheduled_monotonic_times(requests, epoch)
    timeout = aiohttp.ClientTimeout(
        total=REQUEST_TIMEOUT_SECONDS, connect=10, sock_connect=10
    )

    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = [
            asyncio.create_task(
                _send_request(
                    session,
                    request=request,
                    scheduled_s=deadline,
                    replay_epoch_s=epoch,
                )
            )
            for request, deadline in zip(requests, deadlines)
        ]
        return list(await asyncio.gather(*tasks))


async def collective_rpc(method: str, *, args: list[str] | None = None) -> Any:
    timeout = aiohttp.ClientTimeout(total=300)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(
            f"{BASE_URL}/collective_rpc",
            json={"method": method, "args": args or [], "timeout": 300},
        ) as response:
            detail = await response.text()
            if response.status != 200:
                raise RuntimeError(
                    f"collective RPC {method} failed: HTTP {response.status}: {detail}"
                )
            return json.loads(detail) if detail else None


async def warm_up(
    requests: list[TraceRequest],
) -> None:
    warmups = [
        replace(
            requests[index % len(requests)],
            request_id=f"warmup-{index}",
            arrival_offset_s=index * 0.01,
        )
        for index in range(WARMUP_REQUESTS)
    ]
    results = await replay_trace(warmups)
    failed = [row for row in results if row["status"] != "ok"]
    if failed:
        raise RuntimeError(f"{len(failed)} warm-up requests failed")


async def run_benchmark(output_dir: Path) -> Path:
    trace_sha256 = _sha256(TRACE_PATH)
    if trace_sha256 != TRACE_SHA256:
        raise RuntimeError(f"unexpected trace SHA-256: {trace_sha256}")

    requests = load_trace(TRACE_PATH)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(exist_ok=False)
    shutil.copyfile(TRACE_PATH, output_dir / "trace.jsonl")

    run_config = {
        "created_at": _utc_now(),
        "trace": str(TRACE_PATH),
        "trace_sha256": trace_sha256,
        "request_count": len(requests),
        "base_url": BASE_URL,
        "model": MODEL,
        "expected_ranks": EXPECTED_RANKS,
        "warmup_requests": WARMUP_REQUESTS,
        "settle_seconds": SETTLE_SECONDS,
        "request_timeout_seconds": REQUEST_TIMEOUT_SECONDS,
        "decoding": {"temperature": 0.0, "seed": "from trace"},
    }
    _write_json(output_dir / "benchmark_config.json", run_config)

    await warm_up(requests)
    await asyncio.sleep(SETTLE_SECONDS)

    reset_result = await collective_rpc("reset_moe_timing")
    measurement_started_at = _utc_now()
    replay_epoch_s = time.monotonic()
    rows = await replay_trace(
        requests,
        replay_epoch_s=replay_epoch_s,
    )
    measurement_completed_at = _utc_now()

    with (output_dir / "requests.jsonl").open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")

    dump_result = await collective_rpc(
        "dump_moe_timing",
        args=[str(output_dir.resolve())],
    )
    _write_json(
        output_dir / "moe_timing_rpc.json",
        {"reset": reset_result, "dump": dump_result},
    )

    timing_paths = [
        output_dir / f"moe_timing_rank{rank}.json"
        for rank in range(EXPECTED_RANKS)
    ]
    timing = load_timing_files(timing_paths, EXPECTED_RANKS)
    active_ms = active_request_time_ms(rows)
    failed_count = sum(row["status"] != "ok" for row in rows)
    share = (
        fused_moe_time_share_percent(timing, active_ms)
        if failed_count == 0
        else None
    )
    metrics = {
        "status": "ok" if failed_count == 0 else "invalid_failed_requests",
        "measurement_started_at": measurement_started_at,
        "measurement_completed_at": measurement_completed_at,
        "submitted_requests": len(rows),
        "completed_requests": len(rows) - failed_count,
        "failed_requests": failed_count,
        "active_request_wall_time_ms": active_ms,
        "fused_moe_max_rank_time_ms": timing["max_rank_total_ms"],
        "fused_moe_sum_rank_time_ms_diagnostic": timing["sum_rank_total_ms"],
        "fused_moe_over_active_request_wall_time_percent": share,
        "moe_timing_record_count": timing["record_count"],
        "moe_timing_per_rank": timing["per_rank"],
    }
    _write_json(output_dir / "metrics.json", metrics)
    print(json.dumps({"output_dir": str(output_dir), "metrics": metrics}))

    if failed_count:
        raise RuntimeError(
            f"{failed_count} measured requests failed; timing share is invalid"
        )
    return output_dir


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m src.benchmark OUTPUT_DIR")
    asyncio.run(run_benchmark(Path(sys.argv[1])))


if __name__ == "__main__":
    main()
