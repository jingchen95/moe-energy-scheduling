#!/usr/bin/env python3
"""Prepare the measured five-placement by 81-tuple ground-truth matrix."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from src.expert_profile import validate_phase_counts
from src.placement import load_placement, placement_loads
from src.search import DEFAULT_FREQUENCIES_MHZ, enumerate_frequency_tuples
from tools.prepare_milestone_e import load_runs


TTFT_LIMIT_MS = 500.0
TPOT_LIMITS_MS = (12.0, 13.0, 15.0)
EXPECTED_REQUESTS = 50
EXPECTED_OUTPUT_TOKENS = 6400


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _runtime_fingerprint(config: dict[str, Any]) -> dict[str, Any]:
    decoding = config.get("decoding") or {}
    return {
        "model": config.get("model"),
        "vllm_version": config.get("vllm_version"),
        "tokenizer": config.get("tokenizer"),
        "request_rate": config.get("request_rate"),
        "decoding": {
            "temperature": decoding.get("temperature"),
            "seed": decoding.get("seed"),
            "max_output_tokens": decoding.get("max_output_tokens"),
        },
        "max_concurrency": config.get("max_concurrency"),
        "request_timeout_s": config.get("request_timeout_s"),
        "monitor_interval_s": config.get("monitor_interval_s"),
        "warmup_requests": config.get("warmup_requests"),
        "settle_before_s": config.get("settle_before_s"),
        "settle_after_s": config.get("settle_after_s"),
    }


def _load_phase_counts(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        required = {"expert_counts", "prefill_expert_counts", "decode_expert_counts"}
        if not required.issubset(archive.files):
            raise ValueError(f"phase profile must contain {sorted(required)}: {path}")
        total = np.asarray(archive["expert_counts"], dtype=np.int64)
        prefill = np.asarray(archive["prefill_expert_counts"], dtype=np.int64)
        decode = np.asarray(archive["decode_expert_counts"], dtype=np.int64)
    prefill, decode = validate_phase_counts(prefill, decode, total)
    return total, prefill, decode


def _diagnostics(
    total: np.ndarray, prefill: np.ndarray, decode: np.ndarray, placement: Any,
) -> dict[str, Any]:
    aggregate = placement_loads(total, placement).astype(np.int64)
    prefill_load = placement_loads(prefill, placement).astype(np.int64)
    decode_load = placement_loads(decode, placement).astype(np.int64)

    def ratio(load: np.ndarray) -> float:
        mean = float(load.mean())
        return float(load.max() / mean) if mean else 0.0

    return {
        "aggregate_expert_load": aggregate.tolist(),
        "prefill_expert_load": prefill_load.tolist(),
        "decode_expert_load": decode_load.tolist(),
        "aggregate_max_over_mean": ratio(aggregate),
        "prefill_max_over_mean": ratio(prefill_load),
        "decode_max_over_mean": ratio(decode_load),
    }


def _validate_p1_reuse(
    controller: Path, trace: Path, placement_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = load_runs(controller)
    expected = set(enumerate_frequency_tuples(DEFAULT_FREQUENCIES_MHZ))
    tuples = {tuple(row["frequency_tuple"]) for row in rows}
    if len(rows) != 81 or tuples != expected:
        raise ValueError(f"P1 reuse requires exactly 81 unique tuples, found {len(rows)}/{len(tuples)}")
    trace_hash = sha256(trace)
    placement_hash = sha256(placement_path)
    fingerprints: list[dict[str, Any]] = []
    for row in rows:
        run_dir = Path(row["run_dir"]).resolve()
        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        config_path = run_dir / "resolved_config.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise ValueError(f"invalid resolved config: {config_path}")
        if sha256(run_dir / "trace.jsonl") != trace_hash or config.get("trace_sha256") != trace_hash:
            raise ValueError(f"P1 trace mismatch: {run_dir}")
        if config.get("placement_name") != "upstream_default_eplb":
            raise ValueError(f"P1 placement label mismatch: {run_dir}")
        if config.get("placement_sha256") != placement_hash:
            raise ValueError(f"P1 placement hash mismatch: {run_dir}")
        if config.get("vllm_version") != "0.15.1" or config.get("model") != "qwen3-30b-a3b":
            raise ValueError(f"P1 runtime mismatch: {run_dir}")
        if float(config.get("request_rate")) != 0.5:
            raise ValueError(f"P1 request rate mismatch: {run_dir}")
        decoding = config.get("decoding") or {}
        if (float(decoding.get("temperature")) != 0.0
                or int(decoding.get("seed")) != 0
                or decoding.get("max_output_tokens") != [128]):
            raise ValueError(f"P1 decoding mismatch: {run_dir}")
        if (int(metrics.get("completed_requests", 0)) != EXPECTED_REQUESTS
                or int(metrics.get("failed_requests", 1)) != 0
                or int(metrics.get("output_tokens_total", 0)) != EXPECTED_OUTPUT_TOKENS):
            raise ValueError(f"P1 incomplete run: {run_dir}")
        for gpu, target in enumerate(row["frequency_tuple"]):
            actual = metrics.get(f"gpu{gpu}_avg_sm_clock")
            if actual is None or abs(float(actual) - int(target)) > 25:
                raise ValueError(f"P1 clock mismatch in {run_dir}: GPU{gpu} {actual} vs {target}")
        fingerprints.append(_runtime_fingerprint(config))
    if any(value != fingerprints[0] for value in fingerprints[1:]):
        raise ValueError("P1 runs do not share one runtime configuration")
    return rows, fingerprints[0]


def ground_truth_plan(args: argparse.Namespace) -> None:
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    trace = args.trace.resolve()
    phase_counts = args.phase_counts.resolve()
    model_config = args.model_config.resolve()
    for path in (trace, phase_counts, model_config, args.p0, args.p1, args.p2, args.p3, args.p4):
        if not path.exists():
            raise FileNotFoundError(path)

    p1_rows, runtime_fingerprint = _validate_p1_reuse(
        args.p1_controller.resolve(), trace, args.p1.resolve(),
    )
    total, prefill, decode = _load_phase_counts(phase_counts)
    placement_specs = [
        ("P0", "default_linear", args.p0.resolve(), False, 0),
        ("P1", "upstream_eplb", args.p1.resolve(), True, None),
        ("P2", "controlled_medium", args.p2.resolve(), False, 1),
        ("P3", "controlled_high", args.p3.resolve(), False, 2),
        ("P4", "controlled_very_high", args.p4.resolve(), False, 3),
    ]
    placements: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {}
    for policy, name, path, reused, task_index in placement_specs:
        placement = load_placement(path)
        if placement.num_layers != total.shape[0] or placement.num_experts != total.shape[1]:
            raise ValueError(f"{name} shape does not match phase profile")
        diag = _diagnostics(total, prefill, decode, placement)
        diagnostics[policy] = {"placement": name, **diag}
        placements.append({
            "policy": policy,
            "name": name,
            "placement_file": str(path),
            "placement_sha256": sha256(path),
            "reused": reused,
            "task_index": task_index,
            "diagnostics": diag,
        })

    tuples = [list(values) for values in enumerate_frequency_tuples(DEFAULT_FREQUENCIES_MHZ)]
    plan = {
        "schema_version": 1,
        "study": "ground_truth_placement_dvfs",
        "model": "Qwen/Qwen3-30B-A3B",
        "vllm_version": "0.15.1",
        "node": "uranus",
        "parallel": {"tensor": 1, "data": 4, "expert": 4},
        "all2all_backend": "allgather_reducescatter",
        "trace": str(trace),
        "trace_sha256": sha256(trace),
        "phase_counts": str(phase_counts),
        "phase_counts_sha256": sha256(phase_counts),
        "model_config": str(model_config),
        "model_config_sha256": sha256(model_config),
        "runtime_fingerprint": runtime_fingerprint,
        "frequencies_mhz": list(DEFAULT_FREQUENCIES_MHZ),
        "frequency_tuple_count": len(tuples),
        "placement_count": len(placements),
        "expected_matrix_rows": len(placements) * len(tuples),
        "ttft_limit_ms": TTFT_LIMIT_MS,
        "tpot_limits_ms": list(TPOT_LIMITS_MS),
        "placements": placements,
        "new_placement_count": 4,
        "new_array_task_count": 4,
        "frequency_tuples": tuples,
        "p1_controller": str(args.p1_controller.resolve()),
        "p1_reused_run_count": len(p1_rows),
        "selection": {
            "sequential_high_primary": "minimum TPOT p99 among HHHH rows with TTFT p99 <= 500 ms",
            "sequential_dvfs": "minimum measured energy within selected placement subject to TTFT/TPOT SLO",
            "joint": "minimum measured energy across all placements and tuples subject to TTFT/TPOT SLO",
            "performance_models": False,
        },
    }
    write_json(output / "matrix-plan.json", plan)
    write_json(output / "placement-diagnostics.json", diagnostics)
    (output / "trace.sha256").write_text(f"{sha256(trace)}  {trace}\n", encoding="utf-8")
    (output / "p1-reuse-summary.json").write_text(
        json.dumps({"controller": str(args.p1_controller.resolve()), "rows": len(p1_rows),
                    "trace_sha256": sha256(trace), "placement_sha256": sha256(args.p1.resolve()),
                    "runtime_fingerprint": runtime_fingerprint}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(output), "new_array_tasks": 4,
                      "reused_p1_rows": len(p1_rows), "total_matrix_rows": len(placements) * 81}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--phase-counts", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, default=Path("configs/model_qwen3_30b_a3b_profile.yaml"))
    parser.add_argument("--p1-controller", type=Path, required=True)
    parser.add_argument("--p0", type=Path, required=True)
    parser.add_argument("--p1", type=Path, required=True)
    parser.add_argument("--p2", type=Path, required=True)
    parser.add_argument("--p3", type=Path, required=True)
    parser.add_argument("--p4", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    ground_truth_plan(args)


if __name__ == "__main__":
    main()
