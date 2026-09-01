#!/usr/bin/env python3
"""Prepare deterministic calibration and repeatability plans for Milestone I."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Iterable

from src.metrics import percentile
from src.workload import TraceRequest, load_trace, write_trace


CALIBRATION_RATES = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0)
CALIBRATION_ORDER_SEED = 20260902
PLACEMENT_ORDER_SEED = 20260903
RUN_ORDER_SEED = 20260904
PROMPT_SEED = 20260901
HIGH = 2520
MEDIUM = 1635
LOW = 1005


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def targeted_frequency_tuples() -> list[tuple[int, int, int, int]]:
    rows = [(HIGH,) * 4, (MEDIUM,) * 4, (LOW,) * 4]
    for low_rank in range(4):
        rows.append(tuple(LOW if gpu == low_rank else MEDIUM for gpu in range(4)))
    for medium_rank in range(4):
        rows.append(tuple(MEDIUM if gpu == medium_rank else LOW for gpu in range(4)))
    if len(rows) != 11 or len(set(rows)) != 11:
        raise AssertionError("targeted frequency set must contain 11 unique tuples")
    return rows


def _select_length_controlled(
    primary: list[tuple[int, str, str, int]],
    extension: list[tuple[int, str, str, int]],
    *, count: int, seed: int,
) -> list[tuple[int, str, str, int]]:
    if len(primary) > count:
        raise ValueError("primary bucket unexpectedly exceeds requested count")
    needed = count - len(primary)
    if len(extension) < needed:
        raise ValueError(f"need {needed} extension samples, found {len(extension)}")
    rng = random.Random(seed)
    primary = list(primary)
    extension = list(extension)
    rng.shuffle(primary)
    rng.shuffle(extension)
    selected = primary + extension[:needed]
    rng.shuffle(selected)
    return selected


def build_base_trace(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer
    from tools.build_semantic_traces import _prompt, _read_json, _token_count

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    primary: list[tuple[int, str, str, int]] = []
    extension: list[tuple[int, str, str, int]] = []
    for index, row in enumerate(_read_json(args.source)):
        prompt, source_id = _prompt("math", row)
        tokens = _token_count(tokenizer, prompt)
        item = (index, source_id, prompt, tokens)
        if 128 <= tokens <= 256:
            primary.append(item)
        elif 120 <= tokens <= 127:
            extension.append(item)
    selected = _select_length_controlled(
        primary, extension, count=args.count, seed=args.seed,
    )
    requests = [TraceRequest(
        request_id=f"math-i-{position:04d}", category="math", prompt=prompt,
        prompt_tokens=tokens, max_output_tokens=128,
        arrival_offset_s=float(position), dataset_source=args.source.name,
        original_sample_id=str(source_id or index), seed=args.seed,
        metadata={"source_row": index, "token_count_includes_chat_template": True,
                  "milestone_i_bucket": "primary" if tokens >= 128 else "extension"},
    ) for position, (index, source_id, prompt, tokens) in enumerate(selected)]
    write_trace(args.output, requests)
    values = [row.prompt_tokens for row in requests]
    summary = {
        "trace": str(args.output.resolve()), "trace_sha256": sha256(args.output),
        "source": str(args.source.resolve()), "seed": args.seed,
        "number_of_prompts": len(requests), "unique_source_rows": len({
            row.metadata["source_row"] for row in requests
        }),
        "primary_bucket_available": len(primary),
        "primary_bucket_selected": sum(value >= 128 for value in values),
        "extension_bucket_available": len(extension),
        "extension_bucket_selected": sum(value < 128 for value in values),
        "input_token_min": min(values), "input_token_max": max(values),
        "input_token_mean": sum(values) / len(values),
        "input_token_p50": percentile(values, 50),
        "input_token_p90": percentile(values, 90),
        "max_output_tokens": 128,
    }
    write_json(args.output.with_suffix(".summary.json"), summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


def _rescaled(rows: Iterable[TraceRequest], rate: float) -> list[TraceRequest]:
    return [replace(row, arrival_offset_s=index / rate) for index, row in enumerate(rows)]


def _load_h_plan(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("study") != "ground_truth_placement_dvfs":
        raise ValueError("expected the accepted Milestone H matrix plan")
    return value


def _runtime_placement_name(spec: dict[str, Any]) -> str:
    if spec.get("runtime_placement_name"):
        return str(spec["runtime_placement_name"])
    return "upstream_default_eplb" if spec.get("policy") == "P1" else str(spec["name"])


def prepare_calibration(args: argparse.Namespace) -> None:
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    base = load_trace(args.base_trace)
    if len(base) != 300:
        raise ValueError(f"base trace must have 300 requests, found {len(base)}")
    h_plan = _load_h_plan(args.h_plan)
    p1 = dict(next(row for row in h_plan["placements"] if row["policy"] == "P1"))
    p1["runtime_placement_name"] = _runtime_placement_name(p1)
    rates = list(args.rates or CALIBRATION_RATES)
    if any(rate <= 0 for rate in rates) or len(set(rates)) != len(rates):
        raise ValueError("calibration rates must be unique and positive")
    random.Random(args.rate_order_seed).shuffle(rates)
    runs = []
    for order, rate in enumerate(rates):
        trace = output / "traces" / f"gsm8k-100-r{rate:g}.jsonl"
        write_trace(trace, _rescaled(base[:args.requests_per_rate], rate))
        runs.append({"order": order, "request_rate": rate, "trace": str(trace),
                     "trace_sha256": sha256(trace)})
    plan = {
        "schema_version": 1, "study": "higher_load_calibration",
        "model": "Qwen/Qwen3-30B-A3B", "vllm_version": "0.15.1",
        "node": "uranus", "parallel": {"tensor": 1, "data": 4, "expert": 4},
        "model_config": h_plan["model_config"], "phase_counts": h_plan["phase_counts"],
        "base_trace": str(args.base_trace.resolve()),
        "base_trace_sha256": sha256(args.base_trace),
        "placement": p1, "target_clocks_mhz": [HIGH] * 4,
        "requests_per_rate": args.requests_per_rate, "offered_rates": sorted(rates),
        "rate_order_seed": args.rate_order_seed, "runs": runs,
        "selection_target_fraction": 0.65,
        "selection_boundary": "P1/HHHH calibration only; no other placement data",
    }
    write_json(output / "calibration-plan.json", plan)
    print(json.dumps({"output_dir": str(output), "rates": rates}, indent=2))


def prepare_experiment(args: argparse.Namespace) -> None:
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    if args.request_rate <= 0:
        raise ValueError("request rate must be positive")
    base = load_trace(args.base_trace)
    if len(base) != 300:
        raise ValueError(f"base trace must have 300 requests, found {len(base)}")
    trace = output / f"gsm8k-300-r{args.request_rate:g}.jsonl"
    write_trace(trace, _rescaled(base, args.request_rate))
    h_plan = _load_h_plan(args.h_plan)
    specs = [row for row in h_plan["placements"] if row["policy"] in {"P1", "P2", "P3", "P4"}]
    by_policy = {row["policy"]: row for row in specs}
    placement_order = list(by_policy)
    random.Random(PLACEMENT_ORDER_SEED).shuffle(placement_order)
    tuples = targeted_frequency_tuples()
    placements = []
    for task_index, policy in enumerate(placement_order):
        spec = dict(by_policy[policy])
        spec["runtime_placement_name"] = _runtime_placement_name(spec)
        run_rows = [{"frequency_tuple": list(clocks), "repetition": repetition}
                    for clocks in tuples for repetition in range(3)]
        random.Random(RUN_ORDER_SEED + task_index).shuffle(run_rows)
        for order, row in enumerate(run_rows):
            row["order"] = order
        spec.update({"task_index": task_index, "runs": run_rows})
        placements.append(spec)
    plan = {
        "schema_version": 1, "study": "higher_load_repeatability",
        "model": "Qwen/Qwen3-30B-A3B", "vllm_version": "0.15.1",
        "node": "uranus", "parallel": {"tensor": 1, "data": 4, "expert": 4},
        "all2all_backend": "allgather_reducescatter",
        "model_config": h_plan["model_config"], "phase_counts": h_plan["phase_counts"],
        "trace": str(trace), "trace_sha256": sha256(trace),
        "base_trace": str(args.base_trace.resolve()), "base_trace_sha256": sha256(args.base_trace),
        "request_rate": args.request_rate, "requests": 300,
        "frequencies_mhz": {"H": HIGH, "M": MEDIUM, "L": LOW},
        "frequency_tuples": [list(row) for row in tuples], "frequency_tuple_count": 11,
        "repetitions": 3, "placement_count": 4, "expected_runs": 132,
        "placement_order_seed": PLACEMENT_ORDER_SEED,
        "within_placement_order_seed": RUN_ORDER_SEED,
        "placement_task_order": placement_order, "placements": placements,
        "ttft_limit_ms": 500.0, "reference_tpot_slos_ms": [12.0, 13.0, 15.0],
        "exploratory_slo_step_ms": 0.1, "performance_models": False,
        "calibration_analysis": str(args.calibration_analysis.resolve()),
        "calibration_analysis_sha256": sha256(args.calibration_analysis),
    }
    write_json(output / "experiment-plan.json", plan)
    print(json.dumps({"output_dir": str(output), "request_rate": args.request_rate,
                      "task_order": placement_order, "runs": 132}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build-base-trace")
    build.add_argument("--source", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--tokenizer", default="Qwen/Qwen3-30B-A3B")
    build.add_argument("--count", type=int, default=300)
    build.add_argument("--seed", type=int, default=PROMPT_SEED)
    build.set_defaults(func=build_base_trace)
    calibration = sub.add_parser("prepare-calibration")
    calibration.add_argument("--base-trace", type=Path, required=True)
    calibration.add_argument("--h-plan", type=Path, required=True)
    calibration.add_argument("--output-dir", type=Path, required=True)
    calibration.add_argument("--rates", type=float, nargs="+")
    calibration.add_argument("--rate-order-seed", type=int, default=CALIBRATION_ORDER_SEED)
    calibration.add_argument("--requests-per-rate", type=int, default=100,
                             choices=(100, 300))
    calibration.set_defaults(func=prepare_calibration)
    experiment = sub.add_parser("prepare-experiment")
    experiment.add_argument("--base-trace", type=Path, required=True)
    experiment.add_argument("--h-plan", type=Path, required=True)
    experiment.add_argument("--calibration-analysis", type=Path, required=True)
    experiment.add_argument("--request-rate", type=float, required=True)
    experiment.add_argument("--output-dir", type=Path, required=True)
    experiment.set_defaults(func=prepare_experiment)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
