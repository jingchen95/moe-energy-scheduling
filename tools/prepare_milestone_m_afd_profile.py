#!/usr/bin/env python3
"""Prepare the frozen 2A2F routing-profile study after the smoke gate."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess

from src.workload import load_trace, write_trace


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-trace", type=Path, required=True)
    parser.add_argument("--smoke-study", type=Path, required=True)
    parser.add_argument("--profile-patch", type=Path, required=True)
    parser.add_argument("--plugin-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)

    smoke_summary = json.loads(
        (args.smoke_study / "validation-summary.json").read_text(encoding="utf-8")
    )
    if smoke_summary.get("status") != "pass":
        raise ValueError("single custom-placement smoke gate did not pass")
    if smoke_summary.get("selected_configurations") != ["balanced"]:
        raise ValueError("expected one repaired-launcher balanced smoke")
    if (args.smoke_study / "exit-status.txt").read_text().strip() != "0":
        raise ValueError("single custom-placement smoke did not exit zero")

    source_rows = load_trace(args.source_trace)
    if len(source_rows) != 300:
        raise ValueError(f"expected 300 requests, found {len(source_rows)}")
    expected_offsets = [index / 16.0 for index in range(len(source_rows))]
    actual_offsets = [row.arrival_offset_s for row in source_rows]
    if actual_offsets != expected_offsets:
        raise ValueError("source trace is not the frozen 16 request/s schedule")
    if any(row.max_output_tokens != 128 for row in source_rows):
        raise ValueError("source trace does not use 128 output tokens")
    profile_rows = [replace(row, seed=0) for row in source_rows]
    trace_path = output / "trace.jsonl"
    write_trace(trace_path, profile_rows)

    if not args.profile_patch.is_file():
        raise FileNotFoundError(args.profile_patch)
    plugin_commit = subprocess.check_output(
        ["git", "-C", str(args.plugin_root), "rev-parse", "HEAD"], text=True
    ).strip()
    manifest = {
        "schema_version": 1,
        "study": "milestone_m_afd_routing_profile",
        "model": "Qwen/Qwen3-30B-A3B",
        "vllm_version": "0.26.0",
        "afd_plugin_commit": plugin_commit,
        "topology": {"attention_ranks": 2, "ffn_ranks": 2, "tp": 1, "dp": 2, "ep": 2},
        "physical_roles": {"gpu0": "attention0", "gpu1": "attention1", "gpu2": "ffn0", "gpu3": "ffn1"},
        "clocks_mhz": [2520, 2520, 2520, 2520],
        "placement": "default_linear",
        "eplb": False,
        "request_rate": 16.0,
        "requests": 300,
        "decoding": {"temperature": 0.0, "seed": 0, "max_output_tokens": 128},
        "source_trace": str(args.source_trace.resolve()),
        "source_trace_sha256": sha256(args.source_trace),
        "trace": str(trace_path),
        "trace_sha256": sha256(trace_path),
        "smoke_gate": str(args.smoke_study.resolve()),
        "profile_patch": str(args.profile_patch.resolve()),
        "profile_patch_sha256": sha256(args.profile_patch),
        "performance_or_energy_result": False,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
