"""Replay one fixed trace against vLLM and report the FusedMoE time share."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import shutil
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiohttp

from src.metrics import active_request_time_ms, latency_metrics
from src.moe_timing import fused_moe_time_share_percent, load_timing_files
from src.workload import TraceRequest, load_trace, scheduled_monotonic_times


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
    url: str,
    model: str,
    request: TraceRequest,
    scheduled_s: float,
    replay_epoch_s: float,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    delay_s = scheduled_s - time.monotonic()
    if delay_s > 0:
        await asyncio.sleep(delay_s)

    async with semaphore:
        sent_s = time.monotonic()
        first_token_s: float | None = None
        prompt_tokens: int | None = None
        output_tokens: int | None = None
        output_chunks: list[str] = []
        try:
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": request.prompt}],
                "max_tokens": request.max_output_tokens,
                "temperature": 0.0,
                "seed": request.seed,
                "stream": True,
                "stream_options": {"include_usage": True},
            }
            async with session.post(
                url,
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
    base_url: str,
    model: str,
    max_concurrency: int,
    request_timeout_seconds: float,
    replay_epoch_s: float | None = None,
) -> list[dict[str, Any]]:
    """Replay all requests open-loop from their immutable trace offsets."""
    epoch = time.monotonic() if replay_epoch_s is None else replay_epoch_s
    deadlines = scheduled_monotonic_times(requests, epoch)
    timeout = aiohttp.ClientTimeout(
        total=request_timeout_seconds, connect=10, sock_connect=10
    )
    connector = aiohttp.TCPConnector(limit=max_concurrency)
    semaphore = asyncio.Semaphore(max_concurrency)
    url = f"{base_url.rstrip('/')}/v1/chat/completions"

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        tasks = [
            asyncio.create_task(
                _send_request(
                    session,
                    url=url,
                    model=model,
                    request=request,
                    scheduled_s=deadline,
                    replay_epoch_s=epoch,
                    semaphore=semaphore,
                )
            )
            for request, deadline in zip(requests, deadlines)
        ]
        return list(await asyncio.gather(*tasks))


async def check_health(base_url: str) -> None:
    timeout = aiohttp.ClientTimeout(total=10)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(f"{base_url.rstrip('/')}/health") as response:
            if response.status != 200:
                raise RuntimeError(
                    f"vLLM health check returned HTTP {response.status}"
                )


async def collective_rpc(
    base_url: str, method: str, *, args: list[str] | None = None
) -> Any:
    timeout = aiohttp.ClientTimeout(total=300)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(
            f"{base_url.rstrip('/')}/collective_rpc",
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
    *,
    base_url: str,
    model: str,
    count: int,
    request_timeout_seconds: float,
) -> None:
    if count == 0:
        return
    warmups = [
        replace(
            requests[index % len(requests)],
            request_id=f"warmup-{index}",
            arrival_offset_s=index * 0.01,
        )
        for index in range(count)
    ]
    results = await replay_trace(
        warmups,
        base_url=base_url,
        model=model,
        max_concurrency=min(count, 4),
        request_timeout_seconds=request_timeout_seconds,
    )
    failed = [row for row in results if row["status"] != "ok"]
    if failed:
        raise RuntimeError(f"{len(failed)} warm-up requests failed")


async def run_benchmark(args: argparse.Namespace) -> Path:
    requests = load_trace(args.trace)
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(exist_ok=False)
    shutil.copyfile(args.trace, args.output_dir / "trace.jsonl")

    run_config = {
        "created_at": _utc_now(),
        "trace": str(args.trace.resolve()),
        "trace_sha256": _sha256(args.trace),
        "request_count": len(requests),
        "base_url": args.base_url,
        "model": args.model,
        "expected_ranks": args.expected_ranks,
        "warmup_requests": args.warmup_requests,
        "settle_seconds": args.settle_seconds,
        "max_concurrency": args.max_concurrency,
        "request_timeout_seconds": args.request_timeout_seconds,
        "decoding": {"temperature": 0.0, "seed": "from trace"},
    }
    _write_json(args.output_dir / "benchmark_config.json", run_config)

    await check_health(args.base_url)
    await warm_up(
        requests,
        base_url=args.base_url,
        model=args.model,
        count=args.warmup_requests,
        request_timeout_seconds=args.request_timeout_seconds,
    )
    if args.settle_seconds:
        await asyncio.sleep(args.settle_seconds)

    reset_result = await collective_rpc(args.base_url, "reset_moe_timing")
    measurement_started_at = _utc_now()
    replay_epoch_s = time.monotonic()
    rows = await replay_trace(
        requests,
        base_url=args.base_url,
        model=args.model,
        max_concurrency=args.max_concurrency,
        request_timeout_seconds=args.request_timeout_seconds,
        replay_epoch_s=replay_epoch_s,
    )
    measurement_completed_at = _utc_now()

    with (args.output_dir / "requests.jsonl").open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")

    dump_result = await collective_rpc(
        args.base_url,
        "dump_moe_timing",
        args=[str(args.output_dir.resolve())],
    )
    _write_json(
        args.output_dir / "moe_timing_rpc.json",
        {"reset": reset_result, "dump": dump_result},
    )

    timing_paths = [
        args.output_dir / f"moe_timing_rank{rank}.json"
        for rank in range(args.expected_ranks)
    ]
    timing = load_timing_files(timing_paths, args.expected_ranks)
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
    _write_json(args.output_dir / "metrics.json", metrics)
    print(json.dumps({"output_dir": str(args.output_dir), "metrics": metrics}))

    if failed_count:
        raise RuntimeError(
            f"{failed_count} measured requests failed; timing share is invalid"
        )
    return args.output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="qwen3-30b-a3b")
    parser.add_argument("--expected-ranks", type=int, default=4)
    parser.add_argument("--warmup-requests", type=int, default=8)
    parser.add_argument("--settle-seconds", type=float, default=5.0)
    parser.add_argument("--max-concurrency", type=int, default=256)
    parser.add_argument("--request-timeout-seconds", type=float, default=900.0)
    args = parser.parse_args()

    if args.expected_ranks <= 0:
        parser.error("--expected-ranks must be positive")
    if args.warmup_requests < 0:
        parser.error("--warmup-requests must be non-negative")
    if args.settle_seconds < 0:
        parser.error("--settle-seconds must be non-negative")
    if args.max_concurrency <= 0:
        parser.error("--max-concurrency must be positive")
    if args.request_timeout_seconds <= 0:
        parser.error("--request-timeout-seconds must be positive")
    return args


def main() -> None:
    asyncio.run(run_benchmark(parse_args()))


if __name__ == "__main__":
    main()
