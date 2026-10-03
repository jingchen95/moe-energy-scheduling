"""Open-loop semantic trace replay with exact-window four-GPU measurement."""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import subprocess
import time
import uuid
import hashlib
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiohttp
import yaml

from src.gpu_monitor import FixedPeriodMonitor, NVMLReader, summarize_energy_window
from src.metrics import latency_metrics, summarize_run
from src.workload import (
    TraceRequest,
    load_trace,
    scheduled_monotonic_times,
    with_constant_arrival_rate,
    write_trace,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], text=True, check=False,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    return result.stdout.strip() or "unknown"


def _vllm_version() -> str:
    try:
        import vllm
        return str(vllm.__version__)
    except ImportError:
        return "unavailable"


def create_run_directory(root: Path, category: str) -> tuple[str, Path]:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    run_id = f"c-{category}-{stamp}-{uuid.uuid4().hex[:8]}"
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_id, run_dir


async def _consume_stream(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    request: TraceRequest,
    scheduled_monotonic_s: float,
    replay_epoch_monotonic_s: float,
    tokenizer: Any,
    semaphore: asyncio.Semaphore,
    store_output_token_ids: bool = False,
) -> dict[str, Any]:
    delay = scheduled_monotonic_s - time.monotonic()
    if delay > 0:
        await asyncio.sleep(delay)
    try:
        async with semaphore:
            actual_send_monotonic_s = time.monotonic()
            actual_send_unix_s = time.time()
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": request.prompt}],
                "max_tokens": request.max_output_tokens,
                "temperature": 0.0,
                "seed": request.seed,
                "stream": True,
                "stream_options": {"include_usage": True},
            }
            first_token_monotonic_s: float | None = None
            chunks: list[str] = []
            streamed_text_event_count = 0
            server_prompt_tokens: int | None = None
            server_output_tokens: int | None = None
            async with session.post(
                url, json=payload, headers={"X-Request-Id": request.request_id},
            ) as response:
                if response.status != 200:
                    detail = (await response.text())[:500]
                    complete = time.monotonic()
                    _, _, e2e_ms = latency_metrics(
                        actual_send_monotonic_s, None, complete, 0,
                    )
                    return _request_record(
                        request, scheduled_monotonic_s, replay_epoch_monotonic_s,
                        actual_send_monotonic_s, actual_send_unix_s, complete,
                        None, None, e2e_ms, 0, f"http_{response.status}", detail,
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
                            server_prompt_tokens = int(usage["prompt_tokens"])
                        if usage.get("completion_tokens") is not None:
                            server_output_tokens = int(usage["completion_tokens"])
                    for choice in event.get("choices") or []:
                        delta = choice.get("delta") or {}
                        text = (delta.get("reasoning_content") or "") + (delta.get("content") or "")
                        if text:
                            streamed_text_event_count += 1
                            if first_token_monotonic_s is None:
                                first_token_monotonic_s = time.monotonic()
                            chunks.append(text)
            complete_monotonic_s = time.monotonic()
            if server_output_tokens is None:
                server_output_tokens = len(tokenizer.encode(
                    "".join(chunks), add_special_tokens=False,
                )) if chunks else 0
                token_source = "retokenized_text_fallback"
            else:
                token_source = "server_usage"
            ttft_ms, tpot_ms, e2e_ms = latency_metrics(
                actual_send_monotonic_s, first_token_monotonic_s,
                complete_monotonic_s, server_output_tokens,
            )
            row = _request_record(
                request, scheduled_monotonic_s, replay_epoch_monotonic_s,
                actual_send_monotonic_s, actual_send_unix_s, complete_monotonic_s,
                ttft_ms, tpot_ms, e2e_ms, server_output_tokens, "ok", None,
            )
            row["output_token_source"] = token_source
            row["server_prompt_tokens"] = server_prompt_tokens
            row["streamed_text_event_count"] = streamed_text_event_count
            output_bytes = "".join(chunks).encode("utf-8")
            row["output_text_sha256"] = hashlib.sha256(output_bytes).hexdigest()
            if store_output_token_ids:
                # Defer diagnostic tokenization until after the GPU energy window.
                row["_output_text_for_tokens"] = "".join(chunks)
            elif request.metadata.get("store_output_token_ids"):
                row["output_token_ids"] = tokenizer.encode(
                    "".join(chunks), add_special_tokens=False,
                )
            return row
    except Exception as exc:
        complete = time.monotonic()
        actual = locals().get("actual_send_monotonic_s", complete)
        actual_unix = locals().get("actual_send_unix_s", time.time())
        _, _, e2e_ms = latency_metrics(actual, None, complete, 0)
        return _request_record(
            request, scheduled_monotonic_s, replay_epoch_monotonic_s,
            actual, actual_unix, complete, None, None, e2e_ms, 0,
            "client_error", repr(exc),
        )


def _request_record(
    request: TraceRequest,
    scheduled_monotonic_s: float,
    replay_epoch_monotonic_s: float,
    actual_send_monotonic_s: float,
    actual_send_unix_s: float,
    complete_monotonic_s: float,
    ttft_ms: float | None,
    tpot_ms: float | None,
    e2e_ms: float,
    output_tokens: int,
    status: str,
    error: str | None,
) -> dict[str, Any]:
    return {
        "request_id": request.request_id,
        "category": request.category,
        "scheduled_send_s": scheduled_monotonic_s - replay_epoch_monotonic_s,
        "actual_send_s": actual_send_monotonic_s - replay_epoch_monotonic_s,
        "actual_send_unix_s": actual_send_unix_s,
        "completion_monotonic_s": complete_monotonic_s,
        "send_delay_ms": (actual_send_monotonic_s - scheduled_monotonic_s) * 1000.0,
        "ttft_ms": ttft_ms,
        "tpot_ms": tpot_ms,
        "tpot_valid": tpot_ms is not None,
        "tpot_convention": "(completion-first_token)/(output_tokens-1)",
        "e2e_ms": e2e_ms,
        "prompt_tokens": request.prompt_tokens,
        "output_tokens": output_tokens,
        "status": status,
        "error": error,
    }


async def replay_trace(
    requests: list[TraceRequest],
    *,
    base_url: str,
    model: str,
    tokenizer: Any,
    max_concurrency: int,
    request_timeout_s: float,
    epoch_monotonic_s: float | None = None,
    store_output_token_ids: bool = False,
) -> list[dict[str, Any]]:
    epoch = time.monotonic() if epoch_monotonic_s is None else epoch_monotonic_s
    deadlines = scheduled_monotonic_times(requests, epoch)
    timeout = aiohttp.ClientTimeout(
        total=request_timeout_s, connect=10, sock_connect=10,
    )
    connector = aiohttp.TCPConnector(limit=max_concurrency)
    semaphore = asyncio.Semaphore(max_concurrency)
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        tasks = [asyncio.create_task(_consume_stream(
            session, url, model, request, deadline, epoch, tokenizer, semaphore,
            store_output_token_ids,
        )) for request, deadline in zip(requests, deadlines)]
        return list(await asyncio.gather(*tasks))


async def check_health(base_url: str) -> None:
    timeout = aiohttp.ClientTimeout(total=10)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(f"{base_url.rstrip('/')}/health") as response:
            if response.status != 200:
                raise RuntimeError(f"vLLM health check returned HTTP {response.status}")


async def collective_rpc(
    base_url: str, method: str, *, args: list[str] | None = None,
) -> list[Any]:
    timeout = aiohttp.ClientTimeout(total=300)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(
            f"{base_url.rstrip('/')}/collective_rpc",
            json={"method": method, "args": args or [], "timeout": 300},
        ) as response:
            detail = await response.text()
            if response.status != 200:
                raise RuntimeError(f"collective RPC {method} failed: HTTP {response.status}: {detail}")
            value = json.loads(detail) if detail else {"results": []}
            return list(value.get("results") or [])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


async def warm_up(
    requests: list[TraceRequest], *, base_url: str, model: str,
    tokenizer: Any, count: int, request_timeout_s: float,
) -> None:
    if count <= 0:
        return
    warmups = []
    for index in range(count):
        source = requests[index % len(requests)]
        values = asdict(source)
        values["request_id"] = f"warmup-{index}"
        values["arrival_offset_s"] = float(index) * 0.01
        warmups.append(TraceRequest(**values))
    rows = await replay_trace(
        warmups, base_url=base_url, model=model, tokenizer=tokenizer,
        max_concurrency=min(count, 4), request_timeout_s=request_timeout_s,
    )
    failed = [row for row in rows if row["status"] != "ok"]
    if failed:
        raise RuntimeError(f"{len(failed)} warm-up requests failed: {failed[0]['error']}")


async def run_benchmark(args: argparse.Namespace) -> Path:
    if args.expert_profile and args.afd_profile_control_dir:
        raise ValueError("legacy EPLB and AFD routing profiling are mutually exclusive")
    if args.afd_profile_control_dir and args.afd_timing_control_dir:
        raise ValueError("AFD routing and timing instrumentation must use separate runs")
    requests = load_trace(args.trace)
    if args.override_arrival_rate:
        requests = with_constant_arrival_rate(requests, args.request_rate)
    category = requests[0].category
    run_id, run_dir = create_run_directory(args.output_root, category)
    (run_dir / "stdout-stderr.log").touch(exist_ok=False)
    shutil.copyfile(args.trace, run_dir / "trace.jsonl")
    write_trace(run_dir / "resolved_trace.jsonl", requests)
    if args.placement_file:
        shutil.copyfile(args.placement_file, run_dir / "expert_placement.json")
    if args.vllm_patch_manifest:
        shutil.copyfile(args.vllm_patch_manifest, run_dir / "vllm_patch_manifest.json")
    commit = _git_commit()
    (run_dir / "git_commit.txt").write_text(commit + "\n", encoding="utf-8")
    config = {
        "run_id": run_id,
        "created_at": _utc_now(),
        "trace": str(args.trace.resolve()),
        "base_url": args.base_url,
        "model": args.model,
        "vllm_version": _vllm_version(),
        "tokenizer": args.tokenizer,
        "request_rate": args.request_rate,
        "arrival_schedule_source": (
            "request_rate_override" if args.override_arrival_rate else "trace"
        ),
        "run_label": args.run_label,
        "placement_name": args.placement_name,
        "warmup_requests": args.warmup_requests,
        "settle_before_s": args.settle_before_s,
        "settle_after_s": args.settle_after_s,
        "monitor_interval_s": args.monitor_interval_s,
        "max_concurrency": args.max_concurrency,
        "request_timeout_s": args.request_timeout_s,
        "decoding": {"temperature": 0.0, "seed": 0,
                     "max_output_tokens": sorted({row.max_output_tokens for row in requests})},
        "expert_profile": args.expert_profile,
        "afd_profile_control_dir": (
            str(args.afd_profile_control_dir.resolve())
            if args.afd_profile_control_dir else None
        ),
        "afd_timing_control_dir": (
            str(args.afd_timing_control_dir.resolve())
            if args.afd_timing_control_dir else None
        ),
        "placement_file": str(args.placement_file.resolve()) if args.placement_file else None,
        "placement_sha256": _sha256(args.placement_file) if args.placement_file else None,
        "trace_sha256": _sha256(args.trace),
        "resolved_trace_sha256": _sha256(run_dir / "resolved_trace.jsonl"),
        "target_clocks_mhz": args.target_clocks,
        "store_output_token_ids": args.store_output_token_ids,
    }
    (run_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=True), encoding="utf-8",
    )
    (run_dir / "server_metadata.json").write_text(json.dumps({
        "base_url": args.base_url,
        "model": args.model,
        "server_pid_file": str(args.server_pid_file) if args.server_pid_file else None,
        "server_log": str(args.server_log) if args.server_log else None,
        "vllm_patch_manifest": str(args.vllm_patch_manifest) if args.vllm_patch_manifest else None,
        "vllm_patch_manifest_sha256": (
            _sha256(args.vllm_patch_manifest) if args.vllm_patch_manifest else None
        ),
    }, indent=2) + "\n", encoding="utf-8")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=args.local_files_only)
    await check_health(args.base_url)
    await warm_up(
        requests, base_url=args.base_url, model=args.model, tokenizer=tokenizer,
        count=args.warmup_requests, request_timeout_s=args.request_timeout_s,
    )
    if args.settle_before_s:
        await asyncio.sleep(args.settle_before_s)
    if args.expert_profile:
        await collective_rpc(args.base_url, "reset_expert_profile")
    afd_profile_generation = run_id
    if args.afd_profile_control_dir:
        from src.afd_routing_profile import write_profile_control

        write_profile_control(
            args.afd_profile_control_dir, afd_profile_generation, True
        )
    if args.afd_timing_control_dir:
        from src.afd_routing_profile import write_profile_control

        write_profile_control(
            args.afd_timing_control_dir, afd_profile_generation, True
        )

    with NVMLReader(range(4)) as reader:
        monitor = FixedPeriodMonitor(reader, run_dir / "gpu_samples.csv", args.monitor_interval_s)
        monitor.start()
        start_samples = reader.sample_all()
        replay_epoch = time.monotonic()
        try:
            rows = await replay_trace(
                requests, base_url=args.base_url, model=args.model, tokenizer=tokenizer,
                max_concurrency=args.max_concurrency,
                request_timeout_s=args.request_timeout_s,
                epoch_monotonic_s=replay_epoch,
                store_output_token_ids=args.store_output_token_ids,
            )
        finally:
            end_samples = reader.sample_all()
            samples = monitor.stop()

    if args.afd_profile_control_dir:
        from src.afd_routing_profile import (
            finalize_rank_profiles,
            write_profile_control,
        )

        write_profile_control(
            args.afd_profile_control_dir, afd_profile_generation, False
        )
        # Both DP/FFN ranks must observe the inactive transition. Four
        # concurrent flush requests exercise both ranks; their routes are not
        # counted because the transition dumps before layer 0 capture.
        await warm_up(
            requests,
            base_url=args.base_url,
            model=args.model,
            tokenizer=tokenizer,
            count=4,
            request_timeout_s=args.request_timeout_s,
        )
        profile_metadata = finalize_rank_profiles(
            args.afd_profile_control_dir,
            afd_profile_generation,
            run_dir,
            timeout_s=120.0,
        )
        (run_dir / "routing_profile_finalize.json").write_text(
            json.dumps(profile_metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if args.afd_timing_control_dir:
        from src.afd_ffn_timing import finalize_timing_profiles
        from src.afd_routing_profile import write_profile_control

        write_profile_control(
            args.afd_timing_control_dir, afd_profile_generation, False
        )
        await warm_up(
            requests,
            base_url=args.base_url,
            model=args.model,
            tokenizer=tokenizer,
            count=4,
            request_timeout_s=args.request_timeout_s,
        )
        timing_metadata = finalize_timing_profiles(
            args.afd_timing_control_dir,
            afd_profile_generation,
            run_dir,
            timeout_s=120.0,
        )
        (run_dir / "ffn_timing_finalize.json").write_text(
            json.dumps(timing_metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    for row in rows:
        output_text = row.pop("_output_text_for_tokens", None)
        if output_text is not None:
            row["output_token_ids"] = tokenizer.encode(
                output_text, add_special_tokens=False,
            )

    energy = summarize_energy_window(start_samples, end_samples, samples)
    energy["gpu_monitor_missed_deadlines"] = monitor.missed_deadlines
    (run_dir / "energy.json").write_text(
        json.dumps(energy, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    with (run_dir / "requests.jsonl").open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    duration_s = (float(energy["measurement_end_monotonic_s"])
                  - float(energy["measurement_start_monotonic_s"]))
    metrics = summarize_run(
        run_id=run_id, workload=category, request_rate=args.request_rate,
        request_results=rows, measurement_duration_s=duration_s,
        energy_summary=energy,
    )
    metrics["run_label"] = args.run_label
    metrics["placement_name"] = args.placement_name
    metrics["target_clocks_mhz"] = args.target_clocks
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    if args.expert_profile:
        profile_results = await collective_rpc(
            args.base_url, "dump_expert_profile", args=[str(run_dir.resolve())],
        )
        (run_dir / "expert_profile_rpc.json").write_text(
            json.dumps(profile_results, indent=2, sort_keys=True) + "\n", encoding="utf-8",
        )
    if args.settle_after_s:
        await asyncio.sleep(args.settle_after_s)
    print(json.dumps({"run_dir": str(run_dir), "metrics": metrics}, sort_keys=True))
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="qwen3-30b-a3b")
    parser.add_argument("--tokenizer", default="Qwen/Qwen3-30B-A3B")
    parser.add_argument("--request-rate", type=float, required=True)
    parser.add_argument(
        "--override-arrival-rate", action="store_true",
        help="replace trace offsets with index/request-rate while preserving content order",
    )
    parser.add_argument("--output-root", type=Path, default=Path("results/raw"))
    parser.add_argument("--warmup-requests", type=int, default=8)
    parser.add_argument("--settle-before-s", type=float, default=5.0)
    parser.add_argument("--settle-after-s", type=float, default=5.0)
    parser.add_argument("--monitor-interval-s", type=float, default=0.2)
    parser.add_argument("--max-concurrency", type=int, default=256)
    parser.add_argument("--request-timeout-s", type=float, default=900.0)
    parser.add_argument("--server-pid-file", type=Path)
    parser.add_argument("--server-log", type=Path)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--expert-profile", action="store_true")
    parser.add_argument("--afd-profile-control-dir", type=Path)
    parser.add_argument("--afd-timing-control-dir", type=Path)
    parser.add_argument("--placement-file", type=Path)
    parser.add_argument("--vllm-patch-manifest", type=Path)
    parser.add_argument("--target-clocks", type=int, nargs=4)
    parser.add_argument("--store-output-token-ids", action="store_true")
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--placement-name", required=True)
    return parser.parse_args()


def main() -> None:
    asyncio.run(run_benchmark(parse_args()))


if __name__ == "__main__":
    main()
