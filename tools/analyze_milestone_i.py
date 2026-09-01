#!/usr/bin/env python3
"""Validate and analyze the Milestone I calibration and repeated measurements."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any, Iterable

import yaml


TTFT_LIMIT_MS = 500.0
REFERENCE_SLOS_MS = (12.0, 13.0, 15.0)
EXPECTED_OUTPUT_TOKENS_CALIBRATION = 100 * 128
EXPECTED_OUTPUT_TOKENS_EXPERIMENT = 300 * 128


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore",
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def _complete_controller(root: Path, pattern: str, expected_runs: int) -> Path:
    complete = []
    for controller in sorted(root.glob(pattern)):
        result_path = controller / "result.json"
        manifest = controller / "runs.tsv"
        if not result_path.is_file() or not manifest.is_file():
            continue
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
            rows = [row for row in _read_tsv(manifest) if row.get("status") == "complete"]
        except (OSError, json.JSONDecodeError, csv.Error):
            continue
        if result.get("status") == "complete" and len(rows) == expected_runs:
            complete.append(controller)
    if len(complete) != 1:
        raise ValueError(f"expected one complete controller for {pattern}, found {complete}")
    return complete[0]


def _config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"invalid config: {path}")
    return value


def _validate_common(
    run_dir: Path, *, expected_trace_hash: str, request_rate: float,
    expected_requests: int, expected_output_tokens: int,
    placement_name: str, placement_hash: str, clocks: tuple[int, int, int, int],
) -> tuple[dict[str, Any], dict[str, Any]]:
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    config_path = run_dir / "resolved_config.yaml"
    config = _config(config_path)
    if sha256(run_dir / "trace.jsonl") != expected_trace_hash:
        raise ValueError(f"trace artifact mismatch: {run_dir}")
    if config.get("trace_sha256") != expected_trace_hash:
        raise ValueError(f"trace config mismatch: {run_dir}")
    if config.get("model") != "qwen3-30b-a3b" or config.get("vllm_version") != "0.15.1":
        raise ValueError(f"runtime mismatch: {run_dir}")
    if config.get("placement_name") != placement_name:
        raise ValueError(f"placement label mismatch: {run_dir}")
    if config.get("placement_sha256") != placement_hash:
        raise ValueError(f"placement hash mismatch: {run_dir}")
    if not math.isclose(float(config.get("request_rate")), request_rate):
        raise ValueError(f"request-rate mismatch: {run_dir}")
    decoding = config.get("decoding") or {}
    if (float(decoding.get("temperature")) != 0.0 or int(decoding.get("seed")) != 0
            or decoding.get("max_output_tokens") != [128]):
        raise ValueError(f"decoding mismatch: {run_dir}")
    if (int(metrics.get("completed_requests", 0)) != expected_requests
            or int(metrics.get("failed_requests", 1)) != 0
            or int(metrics.get("output_tokens_total", 0)) != expected_output_tokens):
        raise ValueError(f"request completion mismatch: {run_dir}")
    if metrics.get("energy_source") != "nvml_total_energy_counter":
        raise ValueError(f"cumulative NVML energy unavailable: {run_dir}")
    if float(metrics.get("total_gpu_energy_j", 0)) <= 0:
        raise ValueError(f"non-positive energy: {run_dir}")
    for gpu, target in enumerate(clocks):
        actual = metrics.get(f"gpu{gpu}_avg_sm_clock")
        if actual is None or abs(float(actual) - target) > 25:
            raise ValueError(f"clock mismatch GPU{gpu} {actual}/{target}: {run_dir}")
    return metrics, config


def _load_calibration_dir(plan_dir: Path) -> tuple[dict[str, Any], Path, list[dict[str, Any]]]:
    plan = json.loads((plan_dir / "calibration-plan.json").read_text(encoding="utf-8"))
    if plan.get("study") != "higher_load_calibration":
        raise ValueError(f"not a Milestone I calibration plan: {plan_dir}")
    controller = _complete_controller(
        plan_dir / "measured", "attempt-*", len(plan["runs"]),
    )
    expected = {float(row["request_rate"]): row for row in plan["runs"]}
    rows = []
    for entry in _read_tsv(controller / "runs.tsv"):
        if entry.get("status") != "complete":
            continue
        rate = float(entry["request_rate"])
        run_dir = Path(entry["run_dir"]).resolve()
        spec = expected[rate]
        placement = plan["placement"]
        expected_requests = int(plan["requests_per_rate"])
        metrics, config = _validate_common(
            run_dir, expected_trace_hash=spec["trace_sha256"], request_rate=rate,
            expected_requests=expected_requests,
            expected_output_tokens=expected_requests * 128,
            placement_name=placement["runtime_placement_name"],
            placement_hash=placement["placement_sha256"], clocks=(2520,) * 4,
        )
        rows.append({
            "request_rate": rate, "throughput_requests_s": metrics["throughput_requests_s"],
            "ttft_p50_ms": metrics["ttft_p50_ms"], "ttft_p99_ms": metrics["ttft_p99_ms"],
            "tpot_p50_ms": metrics["tpot_p50_ms"], "tpot_p99_ms": metrics["tpot_p99_ms"],
            "e2e_p99_ms": metrics["e2e_p99_ms"],
            "client_send_delay_p99_ms": metrics["client_send_delay_p99_ms"],
            "completed_requests": metrics["completed_requests"],
            "failed_requests": metrics["failed_requests"],
            "total_gpu_energy_j": metrics["total_gpu_energy_j"],
            "run_dir": str(run_dir), "trace_sha256": spec["trace_sha256"],
            "config_sha256": sha256(run_dir / "resolved_config.yaml"),
        })
    if len(rows) != len(expected) or {row["request_rate"] for row in rows} != set(expected):
        raise ValueError("calibration rate sweep is incomplete")
    return plan, controller, rows


def analyze_calibration(args: argparse.Namespace) -> dict[str, Any]:
    plan_dirs = [args.plan_dir, *(args.additional_plan_dir or [])]
    loaded = [_load_calibration_dir(path) for path in plan_dirs]
    plans = [item[0] for item in loaded]
    controllers = [item[1] for item in loaded]
    rows = [row for item in loaded for row in item[2]]
    if len({float(row["request_rate"]) for row in rows}) != len(rows):
        raise ValueError("combined calibration contains duplicate offered rates")
    if len({plan["base_trace_sha256"] for plan in plans}) != 1:
        raise ValueError("combined calibration plans do not share one base trace")
    if len({plan["placement"]["placement_sha256"] for plan in plans}) != 1:
        raise ValueError("combined calibration plans do not share P1 placement")
    rows.sort(key=lambda row: row["request_rate"])
    complete = [row for row in rows if row["failed_requests"] == 0]
    client_valid = [row for row in complete
                    if float(row["client_send_delay_p99_ms"]) <= 50.0]
    if not client_valid:
        raise ValueError("no stable calibration point")
    capacity_lower_bound = max(float(row["throughput_requests_s"]) for row in complete)
    client_valid_capacity = max(float(row["throughput_requests_s"]) for row in client_valid)
    highest = max(rows, key=lambda row: row["request_rate"])
    high_load_rows = sorted(complete, key=lambda row: row["request_rate"])[-3:]
    plateau = (len(high_load_rows) == 3
               and min(float(row["throughput_requests_s"]) for row in high_load_rows)
               >= 0.95 * capacity_lower_bound)
    overloaded = float(highest["request_rate"]) >= 1.20 * capacity_lower_bound
    bracketed = plateau and overloaded
    target = 0.65 * capacity_lower_bound
    recommended = min(client_valid, key=lambda row: (
        abs(float(row["request_rate"]) - target), float(row["request_rate"]),
    ))
    baseline = rows[0]
    queue_visible = (float(recommended["ttft_p99_ms"]) >= 1.10 * float(baseline["ttft_p99_ms"])
                     or float(recommended["tpot_p99_ms"]) >= 1.05 * float(baseline["tpot_p99_ms"]))
    payload = {
        "study": "higher_load_calibration", "controllers": [str(row) for row in controllers],
        "plan_dirs": [str(row) for row in plan_dirs],
        "rows": rows, "saturation_throughput_requests_s": capacity_lower_bound,
        "saturation_estimate_is_lower_bound": not bracketed,
        "client_valid_capacity_throughput_requests_s": client_valid_capacity,
        "saturation_bracketed": bracketed, "plateau_detected": plateau,
        "highest_rate_over_saturation": float(highest["request_rate"]) / capacity_lower_bound,
        "target_fraction": 0.65,
        "target_offered_load_requests_s": target,
        "recommended_request_rate": float(recommended["request_rate"]),
        "recommended_queueing_visible": queue_visible,
        "selection_uses_only_p1_hhhh": True,
    }
    write_json(args.output, payload)
    write_csv(args.output.with_suffix(".csv"), rows, list(rows[0]))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def aggregate_repetitions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, tuple[int, ...]], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["placement"], tuple(row["frequency_tuple"]))
        groups.setdefault(key, []).append(row)
    result = []
    metric_names = (
        "total_gpu_energy_j", "joules_per_request", "ttft_p99_ms", "tpot_p99_ms",
        "tpot_p50_ms", "tpot_mean_ms", "throughput_requests_s",
    )
    for (placement, clocks), group in sorted(groups.items()):
        if len(group) != 3 or {int(row["repetition"]) for row in group} != {0, 1, 2}:
            raise ValueError(f"expected repetitions 0/1/2 for {placement}/{clocks}")
        value: dict[str, Any] = {
            "placement": placement, "placement_name": group[0]["placement_name"],
            "frequency_tuple": list(clocks), "repetitions": 3,
            "failed_requests_total": sum(int(row["failed_requests"]) for row in group),
            "trace_sha256": group[0]["trace_sha256"],
            "placement_sha256": group[0]["placement_sha256"],
        }
        for name in metric_names:
            values = [float(row[name]) for row in group]
            value[f"{name}_mean"] = statistics.mean(values)
            value[f"{name}_std"] = statistics.stdev(values)
        result.append(value)
    return result


def pareto_frontier(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    feasible = [row for row in rows if float(row["ttft_p99_ms_mean"]) <= TTFT_LIMIT_MS]
    frontier = []
    for row in feasible:
        energy = float(row["total_gpu_energy_j_mean"])
        tpot = float(row["tpot_p99_ms_mean"])
        dominated = any(
            float(other["total_gpu_energy_j_mean"]) <= energy
            and float(other["tpot_p99_ms_mean"]) <= tpot
            and (float(other["total_gpu_energy_j_mean"]) < energy
                 or float(other["tpot_p99_ms_mean"]) < tpot)
            for other in feasible if other is not row
        )
        if not dominated:
            frontier.append(row)
    return sorted(frontier, key=lambda row: float(row["tpot_p99_ms_mean"]))


def select_oracles(
    rows: list[dict[str, Any]], thresholds: Iterable[float],
) -> tuple[str, list[dict[str, Any]]]:
    high = [row for row in rows if tuple(row["frequency_tuple"]) == (2520,) * 4
            and float(row["ttft_p99_ms_mean"]) <= TTFT_LIMIT_MS]
    if not high:
        raise ValueError("no HHHH placement satisfies the TTFT guard")
    sequential = min(high, key=lambda row: (
        float(row["tpot_p99_ms_mean"]), float(row["ttft_p99_ms_mean"]), row["placement"],
    ))["placement"]
    output = []
    for threshold in thresholds:
        candidates = [row for row in rows if float(row["ttft_p99_ms_mean"]) <= TTFT_LIMIT_MS
                      and float(row["tpot_p99_ms_mean"]) <= threshold]
        seq = [row for row in candidates if row["placement"] == sequential]
        if not seq or not candidates:
            output.append({"tpot_slo_ms": threshold, "ttft_slo_ms": TTFT_LIMIT_MS,
                           "feasible": False, "reference_slo": threshold in REFERENCE_SLOS_MS})
            continue
        seq_best = min(seq, key=lambda row: float(row["total_gpu_energy_j_mean"]))
        joint_best = min(candidates, key=lambda row: float(row["total_gpu_energy_j_mean"]))
        seq_energy = float(seq_best["total_gpu_energy_j_mean"])
        joint_energy = float(joint_best["total_gpu_energy_j_mean"])
        output.append({
            "tpot_slo_ms": threshold, "ttft_slo_ms": TTFT_LIMIT_MS, "feasible": True,
            "reference_slo": threshold in REFERENCE_SLOS_MS,
            "sequential_placement": sequential,
            "sequential_frequency_tuple": seq_best["frequency_tuple"],
            "sequential_tpot_p99_ms": seq_best["tpot_p99_ms_mean"],
            "sequential_ttft_p99_ms": seq_best["ttft_p99_ms_mean"],
            "sequential_energy_j": seq_energy,
            "joint_placement": joint_best["placement"],
            "joint_frequency_tuple": joint_best["frequency_tuple"],
            "joint_tpot_p99_ms": joint_best["tpot_p99_ms_mean"],
            "joint_ttft_p99_ms": joint_best["ttft_p99_ms_mean"],
            "joint_energy_j": joint_energy,
            "energy_saving_pct": (seq_energy - joint_energy) / seq_energy * 100.0,
        })
    return sequential, output


def classify_opportunity(oracle_rows: list[dict[str, Any]]) -> dict[str, Any]:
    feasible = [row for row in oracle_rows if row.get("feasible")]
    max_saving = max((float(row["energy_saving_pct"]) for row in feasible), default=0.0)
    strong = [float(row["tpot_slo_ms"]) for row in feasible
              if float(row["energy_saving_pct"]) >= 5.0]
    strong_width = max(strong) - min(strong) if strong else 0.0
    if max_saving >= 5.0 and strong_width >= 1.0:
        decision = "A_STRONG_OPPORTUNITY"
    elif max_saving >= 1.0:
        decision = "B_WEAK_OPPORTUNITY"
    else:
        decision = "C_NOT_PROMISING"
    return {"decision": decision, "maximum_energy_saving_pct": max_saving,
            "slo_width_with_at_least_5pct_saving_ms": strong_width}


def _mean_request_tpot(run_dir: Path) -> float:
    values = []
    with (run_dir / "requests.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("status") == "ok" and row.get("tpot_ms") is not None:
                values.append(float(row["tpot_ms"]))
    if not values:
        raise ValueError(f"no request TPOT values: {run_dir}")
    return statistics.mean(values)


def _load_experiment_rows(plan_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    plan = json.loads((plan_dir / "experiment-plan.json").read_text(encoding="utf-8"))
    if plan.get("study") != "higher_load_repeatability":
        raise ValueError("not a Milestone I experiment plan")
    rows = []
    expected_keys = set()
    for spec in plan["placements"]:
        controller = _complete_controller(
            plan_dir / "measured", f"task-{int(spec['task_index']):02d}-attempt-*", 33,
        )
        expected_runs = {(tuple(row["frequency_tuple"]), int(row["repetition"]))
                         for row in spec["runs"]}
        expected_keys.update((spec["policy"], clocks, repetition)
                             for clocks, repetition in expected_runs)
        for entry in _read_tsv(controller / "runs.tsv"):
            if entry.get("status") != "complete":
                continue
            clocks = tuple(int(value) for value in entry["clocks"].split(","))
            repetition = int(entry["repetition"])
            run_dir = Path(entry["run_dir"]).resolve()
            metrics, config = _validate_common(
                run_dir, expected_trace_hash=plan["trace_sha256"],
                request_rate=float(plan["request_rate"]), expected_requests=300,
                expected_output_tokens=EXPECTED_OUTPUT_TOKENS_EXPERIMENT,
                placement_name=spec["runtime_placement_name"],
                placement_hash=spec["placement_sha256"], clocks=clocks,
            )
            row = {
                "run_id": metrics["run_id"], "run_dir": str(run_dir),
                "placement": spec["policy"], "placement_name": spec["name"],
                "placement_sha256": spec["placement_sha256"],
                "frequency_tuple": list(clocks), "repetition": repetition,
                "trace_sha256": plan["trace_sha256"],
                "config_sha256": sha256(run_dir / "resolved_config.yaml"),
                "ttft_p99_ms": metrics["ttft_p99_ms"], "tpot_p99_ms": metrics["tpot_p99_ms"],
                "tpot_p50_ms": metrics["tpot_p50_ms"], "tpot_mean_ms": _mean_request_tpot(run_dir),
                "e2e_p99_ms": metrics["e2e_p99_ms"],
                "total_gpu_energy_j": metrics["total_gpu_energy_j"],
                "joules_per_request": metrics["joules_per_request"],
                "throughput_requests_s": metrics["throughput_requests_s"],
                "failed_requests": metrics["failed_requests"],
                "completed_requests": metrics["completed_requests"],
                "client_send_delay_p99_ms": metrics["client_send_delay_p99_ms"],
            }
            for gpu, target in enumerate(clocks):
                row[f"gpu{gpu}_target_clock_mhz"] = target
                row[f"gpu{gpu}_actual_clock_mhz"] = metrics[f"gpu{gpu}_avg_sm_clock"]
                row[f"gpu{gpu}_energy_j"] = metrics[f"gpu{gpu}_energy_j"]
                row[f"gpu{gpu}_avg_power_w"] = metrics[f"gpu{gpu}_avg_power_w"]
            rows.append(row)
    observed = {(row["placement"], tuple(row["frequency_tuple"]), row["repetition"])
                for row in rows}
    if observed != expected_keys or len(rows) != 132:
        raise ValueError(f"experiment matrix mismatch: {len(rows)} rows")
    return plan, rows


def analyze_experiment(args: argparse.Namespace) -> dict[str, Any]:
    plan, raw = _load_experiment_rows(args.plan_dir)
    aggregate = aggregate_repetitions(raw)
    frontier = pareto_frontier(aggregate)
    feasible_tpots = [float(row["tpot_p99_ms_mean"]) for row in aggregate
                      if float(row["ttft_p99_ms_mean"]) <= TTFT_LIMIT_MS]
    start = math.floor(min(feasible_tpots) * 10) / 10
    stop = math.ceil(max(feasible_tpots) * 10) / 10
    thresholds = [round(start + index * 0.1, 1)
                  for index in range(int(round((stop - start) / 0.1)) + 1)]
    thresholds = sorted(set(thresholds).union(REFERENCE_SLOS_MS))
    sequential, oracle = select_oracles(aggregate, thresholds)
    decision = classify_opportunity(oracle)
    raw_fields = list(raw[0])
    aggregate_fields = list(aggregate[0])
    write_csv(args.output_dir / "milestone_i_runs.csv", raw, raw_fields)
    write_csv(args.output_dir / "milestone_i_aggregate.csv", aggregate, aggregate_fields)
    write_csv(args.output_dir / "milestone_i_pareto.csv", frontier, aggregate_fields)
    write_csv(args.output_dir / "milestone_i_oracle_curve.csv", oracle,
              sorted({key for row in oracle for key in row}))
    calibration = json.loads(Path(plan["calibration_analysis"]).read_text(encoding="utf-8"))
    payload = {
        "study": "higher_load_repeatability", "matrix_runs": len(raw),
        "aggregate_configurations": len(aggregate), "trace_sha256": plan["trace_sha256"],
        "request_rate": plan["request_rate"], "calibration": calibration,
        "sequential_placement": sequential, "pareto_points": frontier,
        "reference_slo_results": [row for row in oracle if row["tpot_slo_ms"] in REFERENCE_SLOS_MS],
        "decision": decision, "performance_models_used": False,
    }
    write_json(args.output_dir / "milestone_i_summary.json", payload)
    _write_report(args.report, plan, aggregate, frontier, oracle, payload)
    _write_figure(args.figure, aggregate, frontier)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def _write_report(path: Path, plan: dict[str, Any], aggregate: list[dict[str, Any]],
                  frontier: list[dict[str, Any]], oracle: list[dict[str, Any]],
                  payload: dict[str, Any]) -> None:
    calibration = payload["calibration"]
    lines = [
        "# Higher-load Placement × DVFS Repeatability Check", "", "## Method", "",
        f"P1/HHHH calibration estimated stable saturation throughput at **{calibration['saturation_throughput_requests_s']:.3f} req/s**. The preregistered 65% target selected **{plan['request_rate']:.3f} req/s**. The fixed trace contains 300 deterministic GSM8K requests and is identical across all 132 measured runs.",
        "", "Four preregistered placements (P1-P4), 11 targeted H/M/L tuples, and three independent measurement windows per configuration were measured. Selection uses measured serving behavior only; expert-load diagnostics and performance models are not used.",
        "", "## HHHH placement selection", "",
        "| placement | TTFT p99 mean ± std (ms) | TPOT p99 mean ± std (ms) | energy mean ± std (J) |",
        "|---|---:|---:|---:|",
    ]
    for row in sorted((x for x in aggregate if tuple(x["frequency_tuple"]) == (2520,) * 4),
                      key=lambda x: x["tpot_p99_ms_mean"]):
        lines.append(f"| {row['placement']} | {row['ttft_p99_ms_mean']:.3f} ± {row['ttft_p99_ms_std']:.3f} | {row['tpot_p99_ms_mean']:.3f} ± {row['tpot_p99_ms_std']:.3f} | {row['total_gpu_energy_j_mean']:.3f} ± {row['total_gpu_energy_j_std']:.3f} |")
    lines.extend(["", "## Measured Pareto frontier", "",
                  "| placement | frequencies (MHz) | TTFT p99 mean (ms) | TPOT p99 mean (ms) | energy mean (J) |",
                  "|---|---|---:|---:|---:|"])
    for row in frontier:
        lines.append(f"| {row['placement']} | {row['frequency_tuple']} | {row['ttft_p99_ms_mean']:.3f} | {row['tpot_p99_ms_mean']:.3f} | {row['total_gpu_energy_j_mean']:.3f} |")
    lines.extend(["", "## Preregistered reference SLOs", "",
                  "| TPOT SLO | feasible | Sequential | Joint | saving |",
                  "|---:|---|---|---|---:|"])
    for row in oracle:
        if row["tpot_slo_ms"] not in REFERENCE_SLOS_MS:
            continue
        if not row["feasible"]:
            lines.append(f"| {row['tpot_slo_ms']:.0f} ms | no | - | - | - |")
        else:
            lines.append(f"| {row['tpot_slo_ms']:.0f} ms | yes | {row['sequential_placement']} {row['sequential_frequency_tuple']} | {row['joint_placement']} {row['joint_frequency_tuple']} | {row['energy_saving_pct']:.3f}% |")
    decision = payload["decision"]
    lines.extend(["", "## Decision", "",
                  f"**{decision['decision']}**. The maximum saving anywhere on the complete 0.1 ms exploratory SLO grid is {decision['maximum_energy_saving_pct']:.3f}%. Preregistered 12/13/15 ms rows are reported above rather than replaced by a favorable post-hoc threshold.",
                  "", "## Limitations", "",
                  "One model, one GSM8K trace, one four-L40S node, four fixed placements, 11 targeted tuples, no replication/migration, and a 500 ms TTFT guard. P99 values are summarized across three windows but remain workload- and system-specific."])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_figure(path: Path, aggregate: list[dict[str, Any]],
                  frontier: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for placement in sorted({row["placement"] for row in aggregate}):
        rows = [row for row in aggregate if row["placement"] == placement]
        ax.scatter([row["tpot_p99_ms_mean"] for row in rows],
                   [row["joules_per_request_mean"] for row in rows], s=20, alpha=0.65,
                   label=placement)
    ax.plot([row["tpot_p99_ms_mean"] for row in frontier],
            [row["joules_per_request_mean"] for row in frontier], color="black",
            linewidth=1.2, label="measured frontier")
    ax.set_xlabel("Mean of repeated TPOT p99 (ms)")
    ax.set_ylabel("GPU joules / request")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    calibration = sub.add_parser("calibration")
    calibration.add_argument("--plan-dir", type=Path, required=True)
    calibration.add_argument("--additional-plan-dir", type=Path, action="append")
    calibration.add_argument("--output", type=Path,
                             default=Path("results/processed/milestone_i_calibration.json"))
    calibration.set_defaults(func=analyze_calibration)
    experiment = sub.add_parser("experiment")
    experiment.add_argument("--plan-dir", type=Path, required=True)
    experiment.add_argument("--output-dir", type=Path, default=Path("results/processed"))
    experiment.add_argument("--report", type=Path,
                            default=Path("docs/milestone_i_higher_load.md"))
    experiment.add_argument("--figure", type=Path,
                            default=Path("results/figures/milestone_i_pareto.pdf"))
    experiment.set_defaults(func=analyze_experiment)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
