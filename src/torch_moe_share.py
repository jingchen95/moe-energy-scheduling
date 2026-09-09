"""Estimate MoE kernel-time share from four complete PyTorch Chrome traces.

Chrome trace `ts`/`dur` fields are in microseconds, regardless of displayTimeUnit.
CPU operators, runtime API calls, graph totals, memcpy and memset are excluded.
"""

from __future__ import annotations

from collections import defaultdict
import gzip
import json
import math
from pathlib import Path
import sys
from typing import Any

from src.nsys_moe_share import estimate_moe_share


def load_gpu_trace(path: Path) -> dict[str, Any]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        document = json.load(stream)
    events = document.get("traceEvents")
    if not isinstance(events, list):
        raise ValueError(f"missing traceEvents list: {path}")

    graph_calls = {
        event.get("args", {}).get("correlation")
        for event in events
        if event.get("cat") in {"cuda_runtime", "cuda_driver"}
        and "GraphLaunch" in event.get("name", "")
        and event.get("ph") == "X"
    } - {None}
    totals: dict[str, float] = defaultdict(float)
    devices: set[int] = set()
    kernel_count = 0
    graph_kernel_count = 0
    for event in events:
        if event.get("cat") != "kernel" or event.get("ph") != "X":
            continue
        name = event.get("name")
        args = event.get("args", {})
        device = args.get("device")
        duration = event.get("dur")
        if not isinstance(name, str) or not name:
            raise ValueError(f"kernel without a name: {path}")
        if isinstance(device, bool) or not isinstance(device, int) or device < 0:
            raise ValueError(f"kernel without an integer CUDA device: {path}")
        if (isinstance(duration, bool) or not isinstance(duration, (int, float))
                or not math.isfinite(duration) or duration < 0):
            raise ValueError(f"invalid kernel duration {duration!r}: {path}")
        devices.add(device)
        totals[name] += duration
        kernel_count += 1
        # Accept both CUPTI graph metadata and correlation to a graph launch.
        graph_metadata = any(
            isinstance(args.get(key), int) and args[key] > 0
            for key in ("graph id", "graph node id")
        )
        if args.get("correlation") in graph_calls or graph_metadata:
            graph_kernel_count += 1

    if len(devices) != 1 or not kernel_count:
        raise ValueError(f"expected one GPU with kernels per worker trace: {path}: {devices}")
    if not graph_kernel_count:
        raise ValueError(
            f"no verifiable CUDA Graph replay kernels: {path}; "
            "do not treat a CPU-only or prefill-only trace as full model timing"
        )
    return {
        "path": str(path), "device": next(iter(devices)),
        "kernel_count": kernel_count, "graph_kernel_count": graph_kernel_count,
        "rows": [{"name": name, "total_time": total} for name, total in totals.items()],
    }


def summarize_traces(paths: list[Path], expected_devices: int = 4) -> dict[str, Any]:
    if len(paths) != expected_devices or len({p.resolve() for p in paths}) != len(paths):
        raise ValueError(f"expected exactly {expected_devices} distinct worker traces")
    traces = [load_gpu_trace(path) for path in paths]
    devices = {trace["device"] for trace in traces}
    if devices != set(range(expected_devices)):
        raise ValueError(f"expected CUDA devices 0..{expected_devices - 1}, got {devices}")
    # Aggregate by name before classification; never add CPU parent totals.
    totals: dict[str, float] = defaultdict(float)
    per_device = []
    for trace in traces:
        for row in trace["rows"]:
            totals[row["name"]] += row["total_time"]
        per_device.append({
            key: value for key, value in trace.items() if key != "rows"
        } | {"total_cuda_kernel_time_us": sum(r["total_time"] for r in trace["rows"])})
    result = estimate_moe_share([
        [{"name": name, "total_time": total} for name, total in totals.items()]
    ])
    if result["moe_compute_kernel_time"] <= 0:
        raise ValueError("no recognized MoE compute kernels; inspect raw traces, not a 0% estimate")
    result.update({
        "profiler": "torch", "time_unit": "us",
        "status": "ok_heuristic_attribution",
        "input_files": [str(path) for path in paths],
        "per_device": sorted(per_device, key=lambda item: item["device"]),
        "total_cuda_kernel_time_ms": result["total_cuda_kernel_time"] / 1000,
        "classified_moe_kernel_time_ms": result["classified_moe_kernel_time"] / 1000,
    })
    result["limitations"].append(
        "Kernel-name attribution is approximate, not a rigorous lower bound or E2E fraction."
    )
    return result


def analyze_run(run_dir: Path) -> dict[str, Any]:
    benchmark = run_dir / "benchmark"
    metrics = json.loads((benchmark / "metrics.json").read_text())
    controls = json.loads((benchmark / "profile_control.json").read_text())
    config = json.loads((benchmark / "profile_config.json").read_text())
    rows = [json.loads(line) for line in (benchmark / "requests.jsonl").read_text().splitlines()]
    if (metrics.get("status") != "ok" or metrics.get("failed_requests") != 0
            or metrics.get("completed_requests") != 3 or len(rows) != 3
            or any(row.get("status") != "ok" for row in rows)):
        raise ValueError("benchmark incomplete/failed; refusing to publish a MoE percentage")
    if (config.get("profiler") != "torch"
            or [row.get("request_id") for row in rows] != config.get("profile_request_ids")):
        raise ValueError("profiler/response identity mismatch")
    for action in ("start", "stop"):
        if controls.get(action, {}).get("http_status") != 200:
            raise ValueError(f"profiler {action} did not complete successfully")
    paths = sorted((run_dir / "profile").glob("*.pt.trace.json"))
    paths += sorted((run_dir / "profile").glob("*.pt.trace.json.gz"))
    result = summarize_traces(paths)
    result["run_scope"] = metrics.get("run_scope", "unknown")
    return result


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m src.torch_moe_share RUN_DIR")
    run_dir = Path(sys.argv[1])
    result = analyze_run(run_dir)
    with (run_dir / "moe-kernel-share.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
