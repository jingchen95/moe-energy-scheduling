#!/usr/bin/env python3
"""Prepare immutable heterogeneous-smoke and primary AFD measurement plans."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import subprocess
from pathlib import Path
from typing import Any

from src.afd_static_placement import load_placement


TRACE_SHA256 = "6cf46c02b3a88ace790c6f9b59dbb7b5f44a9872a5813584a3f3a232b39b1020"
PLACEMENT_SHA256 = {
    "P_balanced": "59c33ea86a95a1c252418b4e612bd1479465941c4b0a726207a8ee362ed6710c",
    "P_MH": "8824e967f45b322788fc0faefcc9d66e4201aed40826f5d853cd732f5b6f2f80",
    "P_HM": "355c1d7da943fcb457feff7be8372e8e2cb9e853e0732797fe6dc4f100d76596",
}
PLACEMENT_FILES = {
    name: Path("data/placements/milestone_m") / f"{name}.json"
    for name in PLACEMENT_SHA256
}
CONFIGURATIONS = (
    ("C0", "P_balanced", (2520, 2520)),
    ("C1", "P_MH", (2520, 2520)),
    ("C2", "P_balanced", (1635, 2520)),
    ("C3", "P_MH", (1635, 2520)),
    ("C4", "P_balanced", (2520, 1635)),
    ("C5", "P_HM", (2520, 1635)),
)
RANDOMIZATION_SEED = 20260929


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_mapping_sha256(mapping: tuple[tuple[int, ...], ...]) -> str:
    payload = json.dumps([list(row) for row in mapping], separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _git_commit(root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()


def frozen_placement(root: Path, name: str) -> dict[str, Any]:
    relative = PLACEMENT_FILES[name]
    path = (root / relative).resolve(strict=True)
    mapping, observed_sha, observed_name = load_placement(str(path))
    if observed_sha != PLACEMENT_SHA256[name]:
        raise ValueError(f"{name} hash drift: {observed_sha}")
    if observed_name != name:
        raise ValueError(f"{name} document name drift: {observed_name}")
    return {
        "name": name,
        "path": str(relative),
        "sha256": observed_sha,
        "canonical_mapping_sha256": canonical_mapping_sha256(mapping),
    }


def matrix_plan(root: Path, trace: Path) -> dict[str, Any]:
    trace_sha = sha256_file(trace)
    if trace_sha != TRACE_SHA256:
        raise ValueError(f"frozen trace hash drift: {trace_sha}")
    order = list(range(len(CONFIGURATIONS)))
    random.Random(RANDOMIZATION_SEED).shuffle(order)
    rows = []
    for task_order, config_index in enumerate(order):
        config_id, placement_name, ffn_clocks = CONFIGURATIONS[config_index]
        rows.append({
            "order": task_order,
            "config_id": config_id,
            "placement": frozen_placement(root, placement_name),
            "attention_clocks_mhz": [2520, 2520],
            "ffn_clocks_mhz": list(ffn_clocks),
            "target_clocks_mhz": [2520, 2520, *ffn_clocks],
            "repetitions": [1, 2, 3],
        })
    return {
        "schema_version": 1,
        "study": "milestone_m_afd_fixed_frequency_primary",
        "git_commit": _git_commit(root),
        "model": "Qwen/Qwen3-30B-A3B",
        "vllm_version": "0.26.0",
        "plugin_commit": "4c2541a53ec8ad507508073eac81ac9feecc6c9a",
        "topology": "2A2F",
        "attention_devices": [0, 1],
        "ffn_devices": [2, 3],
        "execution_mode": "FULL_DECODE_ONLY",
        "trace": "trace.jsonl",
        "trace_sha256": trace_sha,
        "request_rate": 16.0,
        "requests": 300,
        "max_output_tokens": 128,
        "temperature": 0.0,
        "seed": 0,
        "randomization_seed": RANDOMIZATION_SEED,
        "configuration_order": [row["config_id"] for row in rows],
        "configurations": rows,
        "expected_runs": 18,
        "routing_profiling": False,
        "custom_kernel_timing": False,
    }


def smoke_plan(root: Path, baseline_run: Path) -> dict[str, Any]:
    for name in ("responses-pass-1.jsonl", "responses-pass-2.jsonl"):
        if not (baseline_run / name).is_file():
            raise FileNotFoundError(baseline_run / name)
    return {
        "schema_version": 1,
        "study": "milestone_m_afd_heterogeneous_smoke",
        "git_commit": _git_commit(root),
        "model": "Qwen/Qwen3-30B-A3B",
        "vllm_version": "0.26.0",
        "plugin_commit": "4c2541a53ec8ad507508073eac81ac9feecc6c9a",
        "topology": "2A2F",
        "execution_mode": "FULL_DECODE_ONLY",
        "baseline_run": str(baseline_run.resolve()),
        "placement": frozen_placement(root, "P_MH"),
        "target_clocks_mhz": [2520, 2520, 1635, 2520],
        "routing_profiling": False,
        "custom_kernel_timing": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    smoke = subparsers.add_parser("smoke")
    smoke.add_argument("--output-dir", type=Path, required=True)
    smoke.add_argument("--baseline-run", type=Path, required=True)
    matrix = subparsers.add_parser("matrix")
    matrix.add_argument("--output-dir", type=Path, required=True)
    matrix.add_argument("--trace", type=Path, required=True)
    args = parser.parse_args()

    root = Path.cwd().resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    if args.command == "smoke":
        manifest = smoke_plan(root, args.baseline_run.resolve(strict=True))
    else:
        trace = args.trace.resolve(strict=True)
        manifest = matrix_plan(root, trace)
        shutil.copyfile(trace, output / "trace.jsonl")
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(output), **manifest}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
