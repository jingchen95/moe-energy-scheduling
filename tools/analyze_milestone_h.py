#!/usr/bin/env python3
"""Analyze the measured five-placement ground-truth matrix without models."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import yaml

from src.search import slo_feasible
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


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore",
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _load_plan(path: Path) -> dict[str, Any]:
    plan = json.loads(path.read_text(encoding="utf-8"))
    if plan.get("study") != "ground_truth_placement_dvfs":
        raise ValueError("not a ground-truth placement-DVFS plan")
    if int(plan.get("placement_count", 0)) != 5 or int(plan.get("frequency_tuple_count", 0)) != 81:
        raise ValueError("plan must contain five placements and 81 frequency tuples")
    return plan


def _load_resolved_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"invalid resolved config: {path}")
    return value


def _validate_run(
    row: dict[str, Any], spec: dict[str, Any], plan: dict[str, Any],
) -> dict[str, Any]:
    run_dir = Path(row["run_dir"]).resolve()
    metrics_path = run_dir / "metrics.json"
    config_path = run_dir / "resolved_config.yaml"
    if not metrics_path.exists() or not config_path.exists():
        raise ValueError(f"missing measured artifacts under {run_dir}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    config = _load_resolved_config(config_path)
    trace_hash = str(plan["trace_sha256"])
    if sha256(run_dir / "trace.jsonl") != trace_hash or config.get("trace_sha256") != trace_hash:
        raise ValueError(f"trace hash mismatch under {run_dir}")
    if config.get("model") != "qwen3-30b-a3b" or config.get("vllm_version") != "0.15.1":
        raise ValueError(f"runtime model/vLLM mismatch under {run_dir}")
    if config.get("placement_name") != spec["name"]:
        raise ValueError(f"placement name mismatch under {run_dir}")
    if config.get("placement_sha256") != spec["placement_sha256"]:
        raise ValueError(f"placement hash mismatch under {run_dir}")
    if float(config.get("request_rate")) != 0.5:
        raise ValueError(f"request rate mismatch under {run_dir}")
    decoding = config.get("decoding") or {}
    if (float(decoding.get("temperature")) != 0.0 or int(decoding.get("seed")) != 0
            or decoding.get("max_output_tokens") != [128]):
        raise ValueError(f"decoding mismatch under {run_dir}")
    if (int(metrics.get("completed_requests", 0)) != EXPECTED_REQUESTS
            or int(metrics.get("failed_requests", 1)) != 0
            or int(metrics.get("output_tokens_total", 0)) != EXPECTED_OUTPUT_TOKENS):
        raise ValueError(f"incomplete measured run under {run_dir}")
    if metrics.get("energy_source") != "nvml_total_energy_counter":
        raise ValueError(f"NVML cumulative energy unavailable under {run_dir}")
    clocks = tuple(int(value) for value in row["frequency_tuple"])
    for gpu, target in enumerate(clocks):
        actual = metrics.get(f"gpu{gpu}_avg_sm_clock")
        if actual is None or abs(float(actual) - target) > 25:
            raise ValueError(f"actual clock mismatch under {run_dir}: GPU{gpu} {actual}/{target}")
    value: dict[str, Any] = {
        "run_id": metrics.get("run_id"),
        "run_dir": str(run_dir),
        "placement": spec["policy"],
        "placement_name": spec["name"],
        "placement_file": spec["placement_file"],
        "placement_sha256": spec["placement_sha256"],
        "frequency_tuple": json.dumps(list(clocks)),
        "trace_sha256": trace_hash,
        "config_sha256": sha256(config_path),
        "ttft_p99_ms": metrics.get("ttft_p99_ms"),
        "tpot_p99_ms": metrics.get("tpot_p99_ms"),
        "e2e_p99_ms": metrics.get("e2e_p99_ms"),
        "total_gpu_energy_j": metrics.get("total_gpu_energy_j"),
        "joules_per_request": metrics.get("joules_per_request"),
        "throughput_requests_s": metrics.get("throughput_requests_s"),
        "failed_requests": metrics.get("failed_requests"),
        "completed_requests": metrics.get("completed_requests"),
        "input_tokens_total": metrics.get("input_tokens_total"),
        "output_tokens_total": metrics.get("output_tokens_total"),
    }
    for gpu, target in enumerate(clocks):
        value[f"gpu{gpu}_target_clock_mhz"] = target
        value[f"gpu{gpu}_actual_clock_mhz"] = metrics.get(f"gpu{gpu}_avg_sm_clock")
        value[f"gpu{gpu}_energy_j"] = metrics.get(f"gpu{gpu}_energy_j")
        value[f"gpu{gpu}_avg_power_w"] = metrics.get(f"gpu{gpu}_avg_power_w")
        value[f"gpu{gpu}_avg_sm_util"] = metrics.get(f"gpu{gpu}_avg_sm_util")
        value[f"gpu{gpu}_avg_memory_util"] = metrics.get(f"gpu{gpu}_avg_memory_util")
    for phase in ("aggregate", "prefill", "decode"):
        loads = spec["diagnostics"][f"{phase}_expert_load"]
        for gpu, load in enumerate(loads):
            value[f"{phase}_expert_load_g{gpu}"] = load
        value[f"{phase}_max_over_mean"] = spec["diagnostics"][f"{phase}_max_over_mean"]
    for limit in TPOT_LIMITS_MS:
        value[f"slo{int(limit)}_feasible"] = slo_feasible(
            value, ttft_limit_ms=TTFT_LIMIT_MS, tpot_limit_ms=limit,
        )
    return value


def _successful_controller(study: Path, task_index: int) -> Path:
    paths = sorted((study / "measured").glob(f"task-{task_index:02d}-attempt-*/runs.tsv"))
    complete = [path for path in paths if path.exists()]
    if len(complete) != 1:
        raise ValueError(f"expected one completed controller for task {task_index}, found {len(complete)}")
    return complete[0].parent


def load_matrix(plan_dir: Path) -> list[dict[str, Any]]:
    plan = _load_plan(plan_dir / "matrix-plan.json")
    specs = {row["policy"]: row for row in plan["placements"]}
    rows: list[dict[str, Any]] = []
    p1_controller = Path(plan["p1_controller"])
    p1_rows = load_runs(p1_controller)
    if len(p1_rows) != 81:
        raise ValueError(f"P1 reuse has {len(p1_rows)} rows, expected 81")
    for entry in p1_rows:
        rows.append(_validate_run(entry, specs["P1"], plan))
    for spec in plan["placements"]:
        if spec["reused"]:
            continue
        controller = _successful_controller(plan_dir, int(spec["task_index"]))
        entries = load_runs(controller)
        if len(entries) != 81:
            raise ValueError(f"{spec['policy']} has {len(entries)} rows, expected 81")
        for entry in entries:
            rows.append(_validate_run(entry, spec, plan))
    expected = {(policy, tuple(values)) for policy in specs
                for values in plan["frequency_tuples"]}
    observed = {(row["placement"], tuple(json.loads(row["frequency_tuple"]))) for row in rows}
    if observed != expected or len(rows) != 405:
        raise ValueError(f"matrix is incomplete or duplicated: {len(rows)} rows")
    rows.sort(key=lambda row: (row["placement"], json.loads(row["frequency_tuple"])))
    return rows


def select_oracles(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    high = [row for row in rows if tuple(json.loads(row["frequency_tuple"])) == (2520,) * 4
            and float(row["ttft_p99_ms"]) <= TTFT_LIMIT_MS]
    if not high:
        raise ValueError("no high-frequency placement satisfies TTFT SLO")
    sequential_placement = min(high, key=lambda row: (
        float(row["tpot_p99_ms"]), float(row["ttft_p99_ms"]),
        float(row["total_gpu_energy_j"]), row["placement"],
    ))["placement"]
    summary: list[dict[str, Any]] = []
    for limit in TPOT_LIMITS_MS:
        sequential_candidates = [row for row in rows
                                 if row["placement"] == sequential_placement
                                 and slo_feasible(row, ttft_limit_ms=TTFT_LIMIT_MS,
                                                   tpot_limit_ms=limit)]
        joint_candidates = [row for row in rows
                            if slo_feasible(row, ttft_limit_ms=TTFT_LIMIT_MS,
                                            tpot_limit_ms=limit)]
        if not sequential_candidates or not joint_candidates:
            raise ValueError(f"no feasible point at TPOT <= {limit:g} ms")
        seq = min(sequential_candidates, key=lambda row: (
            float(row["total_gpu_energy_j"]), float(row["tpot_p99_ms"]),
            row["placement"], row["frequency_tuple"],
        ))
        joint = min(joint_candidates, key=lambda row: (
            float(row["total_gpu_energy_j"]), float(row["tpot_p99_ms"]),
            row["placement"], row["frequency_tuple"],
        ))
        seq_energy = float(seq["total_gpu_energy_j"])
        joint_energy = float(joint["total_gpu_energy_j"])
        summary.append({
            "tpot_slo_ms": limit,
            "ttft_slo_ms": TTFT_LIMIT_MS,
            "sequential_placement": seq["placement"],
            "sequential_placement_name": seq["placement_name"],
            "sequential_frequency_tuple": seq["frequency_tuple"],
            "sequential_ttft_p99_ms": seq["ttft_p99_ms"],
            "sequential_tpot_p99_ms": seq["tpot_p99_ms"],
            "sequential_energy_j": seq_energy,
            "joint_placement": joint["placement"],
            "joint_placement_name": joint["placement_name"],
            "joint_frequency_tuple": joint["frequency_tuple"],
            "joint_ttft_p99_ms": joint["ttft_p99_ms"],
            "joint_tpot_p99_ms": joint["tpot_p99_ms"],
            "joint_energy_j": joint_energy,
            "energy_saving_pct": (seq_energy - joint_energy) / seq_energy * 100.0,
            "joint_strictly_lower_energy": joint_energy < seq_energy,
            "sequential_run_dir": seq["run_dir"],
            "joint_run_dir": joint["run_dir"],
        })
    return {"placement": sequential_placement,
            "placement_name": next(row["placement_name"] for row in rows
                                    if row["placement"] == sequential_placement)}, summary


def write_report(
    path: Path, plan: dict[str, Any], rows: list[dict[str, Any]],
    sequential: dict[str, Any], summary: list[dict[str, Any]],
) -> None:
    lines = [
        "# Ground-truth Placement × DVFS Motivation Sweep",
        "",
        "## Research question",
        "",
        "Does exhaustive measured Joint placement plus per-GPU DVFS find a lower-energy SLO-feasible point than the measured Placement→DVFS Sequential Oracle? No performance or service model is fit or used.",
        "",
        "## Method and validation",
        "",
        "Qwen/Qwen3-30B-A3B, vLLM 0.15.1, TP=1/DP=4/EP=4, four L40S GPUs on Uranus, `allgather_reducescatter`, fixed GSM8K 50-request trace at 0.5 requests/s, and H/M/L = 2520/1635/1005 MHz. The matrix contains five placements × 81 tuples = 405 measured/reused rows. P1 is reused only after exact trace, mapping, runtime, completion, NVML-energy, and clock checks. Selection uses only measured TTFT p99, TPOT p99, and total GPU energy.",
        "",
        "## Sequential Oracle",
        "",
        f"At `(H,H,H,H)`, the performance-selected placement was **{sequential['placement']} ({sequential['placement_name']})**, using TPOT p99 as the primary metric and TTFT p99 <= 500 ms as a hard constraint.",
        "",
        "## Placement diagnostics",
        "",
        "These loads are recorded for interpretation only and are not used to select points.",
        "",
        "| placement | aggregate GPU loads | prefill GPU loads | decode GPU loads | aggregate max/mean |",
        "|---|---|---|---|---:|",
    ]
    for spec in plan["placements"]:
        diag = spec["diagnostics"]
        lines.append(f"| {spec['policy']} {spec['name']} | {diag['aggregate_expert_load']} | {diag['prefill_expert_load']} | {diag['decode_expert_load']} | {diag['aggregate_max_over_mean']:.3f} |")
    lines.extend([
        "",
        "## Sequential versus Joint Oracle",
        "",
        "| SLO | Sequential placement | Sequential DVFS | Sequential energy (J) | Joint placement | Joint DVFS | Joint energy (J) | Saving |",
        "|---|---|---|---:|---|---|---:|---:|",
    ])
    for row in summary:
        lines.append(f"| TPOT <= {row['tpot_slo_ms']:g} ms, TTFT <= 500 ms | {row['sequential_placement']} | {row['sequential_frequency_tuple']} | {row['sequential_energy_j']:.3f} | {row['joint_placement']} | {row['joint_frequency_tuple']} | {row['joint_energy_j']:.3f} | {row['energy_saving_pct']:.4f}% |")
    lines.extend([
        "",
        "## Data and limitations",
        "",
        "The complete normalized measured table is `results/processed/milestone_h_ground_truth_matrix.csv`; the oracle table is `results/processed/milestone_h_oracle_summary.csv`. Every row retains actual clocks, trace/config hashes, energy, throughput, failures, and diagnostic placement loads. Raw run directories are not committed. This isolates placement × DVFS with four active GPUs and exactly 32 experts per rank; it does not optimize active GPU count, replication, migration, or communication.",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    plan = _load_plan(args.plan_dir / "matrix-plan.json")
    rows = load_matrix(args.plan_dir)
    sequential, summary = select_oracles(rows)
    fields = ["run_id", "run_dir", "placement", "placement_name", "placement_file",
              "placement_sha256", "frequency_tuple", "ttft_p99_ms", "tpot_p99_ms",
              "e2e_p99_ms", "total_gpu_energy_j", "joules_per_request",
              "throughput_requests_s", "failed_requests", "completed_requests",
              "input_tokens_total", "output_tokens_total", "trace_sha256", "config_sha256"]
    for gpu in range(4):
        fields.extend([f"gpu{gpu}_target_clock_mhz", f"gpu{gpu}_actual_clock_mhz",
                       f"gpu{gpu}_energy_j", f"gpu{gpu}_avg_power_w",
                       f"gpu{gpu}_avg_sm_util", f"gpu{gpu}_avg_memory_util"])
    for phase in ("aggregate", "prefill", "decode"):
        fields.extend([f"{phase}_expert_load_g{gpu}" for gpu in range(4)])
        fields.append(f"{phase}_max_over_mean")
    fields.extend([f"slo{int(limit)}_feasible" for limit in TPOT_LIMITS_MS])
    write_csv(args.output_dir / "milestone_h_ground_truth_matrix.csv", rows, fields)
    write_csv(args.output_dir / "milestone_h_oracle_summary.csv", summary, list(summary[0]))
    payload = {
        "study": "ground_truth_placement_dvfs",
        "plan_dir": str(args.plan_dir.resolve()),
        "matrix_rows": len(rows),
        "trace_sha256": plan["trace_sha256"],
        "model_config_sha256": plan["model_config_sha256"],
        "sequential_oracle": sequential,
        "oracle_summary": summary,
        "performance_models_used": False,
        "matrix_validated": True,
    }
    (args.output_dir / "milestone_h_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    write_report(args.report, plan, rows, sequential, summary)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/processed"))
    parser.add_argument("--report", type=Path, default=Path("docs/milestone_h_ground_truth.md"))
    args = parser.parse_args()
    print(json.dumps(analyze(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
