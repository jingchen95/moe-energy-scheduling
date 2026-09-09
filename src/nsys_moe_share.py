"""Compute a conservative MoE kernel-time share from nsys CSV summaries."""

from __future__ import annotations

import csv
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Iterable


# These deliberately prefer under-counting to assigning generic GEMM kernels to MoE.
MOE_COMPUTE_PATTERN = re.compile(
    r"(fused.?moe|moe.?align|moe.?sum|moe.?topk|topk.?softmax|"
    r"grouped.?gemm|grouped.?mm|expert)",
    re.IGNORECASE,
)
MOE_COMM_PATTERN = re.compile(
    r"(all.?gather|reduce.?scatter)",
    re.IGNORECASE,
)


def _number(value: str) -> float:
    return float(value.strip().replace(",", ""))


def load_kernel_summary(path: Path) -> list[dict[str, Any]]:
    """Read `nsys stats --format csv --report cuda_gpu_kern_sum` output."""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    header_index = None
    for index, line in enumerate(lines):
        if "Total Time" in line and "Name" in line:
            header_index = index
            break
    if header_index is None:
        raise ValueError(f"no CUDA kernel-summary CSV header in {path}")

    rows: list[dict[str, Any]] = []
    reader = csv.DictReader(lines[header_index:])
    if not reader.fieldnames:
        raise ValueError(f"empty CUDA kernel-summary CSV in {path}")
    total_column = next(
        (name for name in reader.fieldnames if name.startswith("Total Time")), None
    )
    name_column = next(
        (name for name in reader.fieldnames if name.strip() == "Name"), None
    )
    if total_column is None or name_column is None:
        raise ValueError(f"missing Total Time or Name column in {path}")

    for row in reader:
        name = (row.get(name_column) or "").strip()
        raw_total = row.get(total_column)
        if not name or raw_total is None:
            continue
        try:
            total_time = _number(raw_total)
        except ValueError:
            continue
        if not math.isfinite(total_time) or total_time < 0:
            raise ValueError(f"invalid kernel time in {path}: {raw_total!r}")
        rows.append({"name": name, "total_time": total_time})
    if not rows:
        raise ValueError(f"no CUDA kernel rows in {path}")
    return rows


def estimate_moe_share(
    summaries: Iterable[Iterable[dict[str, Any]]],
) -> dict[str, Any]:
    all_rows = [row for summary in summaries for row in summary]
    total = sum(float(row["total_time"]) for row in all_rows)
    if total <= 0:
        raise ValueError("total CUDA kernel time must be positive")

    compute_rows = [
        row for row in all_rows if MOE_COMPUTE_PATTERN.search(str(row["name"]))
    ]
    communication_rows = [
        row
        for row in all_rows
        if not MOE_COMPUTE_PATTERN.search(str(row["name"]))
        and MOE_COMM_PATTERN.search(str(row["name"]))
    ]
    compute = sum(float(row["total_time"]) for row in compute_rows)
    communication = sum(
        float(row["total_time"]) for row in communication_rows
    )
    moe = compute + communication
    unmatched = sorted(
        (
            row
            for row in all_rows
            if row not in compute_rows and row not in communication_rows
        ),
        key=lambda row: float(row["total_time"]),
        reverse=True,
    )
    return {
        "schema_version": 1,
        "metric": "conservative_moe_cuda_kernel_time_share_percent",
        "run_scope": os.environ.get("MOE_PROFILE_RUN_SCOPE", "unknown"),
        "scope": "aggregate kernel time across all captured ranks/devices",
        "total_cuda_kernel_time": total,
        "moe_compute_kernel_time": compute,
        "moe_allgather_reducescatter_kernel_time": communication,
        "classified_moe_kernel_time": moe,
        "conservative_moe_cuda_kernel_time_share_percent": moe / total * 100.0,
        "classification": {
            "moe_compute_regex": MOE_COMPUTE_PATTERN.pattern,
            "moe_communication_regex": MOE_COMM_PATTERN.pattern,
            "compute_kernel_names": sorted({row["name"] for row in compute_rows}),
            "communication_kernel_names": sorted(
                {row["name"] for row in communication_rows}
            ),
            "largest_unmatched_kernels": unmatched[:30],
        },
        "limitations": [
            "This is a kernel-time share, not an end-to-end wall-time share.",
            "Generic GEMM and fused kernels are left unmatched, so the result can undercount MoE.",
            "Summed kernel time can double-count overlapping streams; use it only as a coarse diagnostic.",
            "AllGather and ReduceScatter are attributed to EP because this run fixes TP=1 and the AgRs all-to-all backend.",
        ],
    }


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit(
            "usage: python -m src.nsys_moe_share OUTPUT_JSON NSYS_STATS_CSV..."
        )
    output_path = Path(sys.argv[1])
    input_paths = [Path(value) for value in sys.argv[2:]]
    result = estimate_moe_share(load_kernel_summary(path) for path in input_paths)
    result["input_files"] = [str(path) for path in input_paths]
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
