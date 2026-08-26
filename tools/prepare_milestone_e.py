#!/usr/bin/env python3
"""Prepare deterministic Milestone E plans without launching GPU work."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any
from src.search import (
    DEFAULT_FREQUENCIES_MHZ,
    enumerate_frequency_tuples,
    select_joint_candidates,
    select_min_energy,
)


SLOS = (12.0, 13.0, 15.0)
TTFT_LIMIT_MS = 500.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_runs(controller: Path) -> list[dict[str, Any]]:
    manifest = controller / "runs.tsv"
    rows: list[dict[str, Any]] = []
    with manifest.open(newline="", encoding="utf-8") as stream:
        for entry in csv.DictReader(stream, delimiter="\t"):
            if entry.get("status", "complete") != "complete":
                continue
            run_dir = Path(entry["run_dir"])
            metrics_path = run_dir / "metrics.json"
            if not metrics_path.exists():
                continue
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            clocks = metrics.get("target_clocks_mhz")
            if clocks is None and entry.get("clocks"):
                clocks = [int(value) for value in entry["clocks"].split(",")]
            rows.append({**entry, **metrics, "frequency_tuple": tuple(clocks),
                         "run_dir": str(run_dir)})
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sequential_plan(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=False)
    tuples = enumerate_frequency_tuples(DEFAULT_FREQUENCIES_MHZ, seed=args.seed)
    rows = [{"order": index, "label": f"b4-{index:02d}-"
             + "-".join(str(value) for value in clocks),
             "frequency_tuple": list(clocks)} for index, clocks in enumerate(tuples)]
    write_json(args.output_dir / "sequential-plan.json", {
        "schema_version": 1, "seed": args.seed,
        "frequency_levels_mhz": list(DEFAULT_FREQUENCIES_MHZ),
        "configuration_count": len(rows), "configurations": rows,
    })
    with (args.output_dir / "sequential-plan.tsv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(("order", "label", "f0", "f1", "f2", "f3"))
        for row in rows:
            writer.writerow((row["order"], row["label"], *row["frequency_tuple"]))
    print(json.dumps({"output_dir": str(args.output_dir), "configurations": len(rows)}))


def _load_counts(path: Path) -> Any:
    import numpy as np
    from src.expert_profile import validate_counts
    with np.load(path, allow_pickle=False) as archive:
        return validate_counts(archive["expert_counts"])


def joint_plan(args: argparse.Namespace) -> None:
    import numpy as np
    from src.placement import (
        frequency_aware_placement,
        load_placement,
        placement_loads,
        save_placement,
    )
    rows = load_runs(args.sequential_controller)
    trace_path = args.sequential_controller / "trace.jsonl"
    if not trace_path.exists():
        raise SystemExit(f"Sequential controller has no immutable trace copy: {trace_path}")
    trace_hash = sha256(trace_path)
    tuples = {tuple(row["frequency_tuple"]) for row in rows}
    expected = set(enumerate_frequency_tuples(DEFAULT_FREQUENCIES_MHZ))
    if tuples != expected or len(rows) != 81:
        missing = sorted(expected - tuples)
        duplicates = len(rows) - len(tuples)
        raise SystemExit(
            f"Joint planning requires 81 unique successful Sequential runs; "
            f"found {len(rows)} rows/{len(tuples)} tuples, {len(missing)} missing, "
            f"{duplicates} duplicates"
        )
    for row in rows:
        run_dir = Path(row["run_dir"])
        if sha256(run_dir / "trace.jsonl") != trace_hash:
            raise SystemExit(f"trace hash mismatch in Sequential run {run_dir}")
        if int(row.get("output_tokens_total", 0)) != 50 * 128:
            raise SystemExit(f"unexpected output-token count in Sequential run {run_dir}")
        for gpu, target in enumerate(row["frequency_tuple"]):
            actual = row.get(f"gpu{gpu}_avg_sm_clock")
            if actual is None or abs(float(actual) - int(target)) > 25:
                raise SystemExit(
                    f"GPU{gpu} actual clock {actual} does not match {target} in {run_dir}"
                )
    counts = _load_counts(args.expert_counts)
    p1 = load_placement(args.p1_placement)
    candidates = select_joint_candidates(
        rows, tpot_limits_ms=SLOS, ttft_limit_ms=TTFT_LIMIT_MS,
        levels=DEFAULT_FREQUENCIES_MHZ, max_candidates=args.max_candidates,
    )
    if len(candidates) < min(5, args.max_candidates):
        raise SystemExit(f"candidate selection produced only {len(candidates)} configurations")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    planned = []
    for index, candidate in enumerate(candidates):
        clocks = tuple(candidate["frequency_tuple"])
        placement = frequency_aware_placement(counts, clocks)
        placement_path = args.output_dir / "placements" / (
            f"joint-{index:02d}-" + "-".join(str(value) for value in clocks) + ".json"
        )
        save_placement(placement_path, placement)
        source = candidate["sequential_row"]
        planned.append({
            "candidate_index": index,
            "label": f"joint-{index:02d}-" + "-".join(str(value) for value in clocks),
            "frequency_tuple": list(clocks), "reason": candidate["reason"],
            "placement_file": str(placement_path.resolve()),
            "placement_sha256": sha256(placement_path),
            "sequential_run_dir": source["run_dir"],
            "sequential_tpot_p99_ms": source["tpot_p99_ms"],
            "sequential_ttft_p99_ms": source["ttft_p99_ms"],
            "sequential_total_gpu_energy_j": source["total_gpu_energy_j"],
            "predicted_raw_load": placement.metadata["per_rank_raw_load"],
            "predicted_normalized_load": placement.metadata["per_rank_normalized_load"],
            "normalized_max_over_mean": placement.metadata["normalized_max_over_mean"],
        })

    sanity_clocks = (2520, 1635, 1635, 1005)
    sanity = frequency_aware_placement(counts, sanity_clocks)
    p1_raw = placement_loads(counts, p1).astype(float)
    speeds = np.asarray(sanity.metadata["speed_factors"])
    p1_normalized = p1_raw / speeds
    joint_raw = np.asarray(sanity.metadata["per_rank_raw_load"])
    joint_normalized = np.asarray(sanity.metadata["per_rank_normalized_load"])
    p1_pressure_ratio = float(p1_normalized.max() / p1_normalized.mean())
    joint_pressure_ratio = float(joint_normalized.max() / joint_normalized.mean())
    if not joint_raw[0] > joint_raw[3] or not joint_pressure_ratio < p1_pressure_ratio:
        raise SystemExit(
            "H/M/M/L sanity check failed: frequency-aware placement did not shift "
            "load toward the high-frequency rank and improve normalized balance"
        )
    write_json(args.output_dir / "sanity-hmml.json", {
        "frequency_tuple": sanity_clocks,
        "speed_factors": speeds.tolist(),
        "p1_raw_load": p1_raw.tolist(),
        "p1_normalized_load": p1_normalized.tolist(),
        "p1_normalized_max_over_mean": p1_pressure_ratio,
        "joint_raw_load": sanity.metadata["per_rank_raw_load"],
        "joint_normalized_load": sanity.metadata["per_rank_normalized_load"],
        "joint_normalized_max_over_mean": joint_pressure_ratio,
    })
    write_json(args.output_dir / "joint-plan.json", {
        "schema_version": 1, "sequential_controller": str(args.sequential_controller.resolve()),
        "expert_counts": str(args.expert_counts.resolve()),
        "expert_counts_sha256": sha256(args.expert_counts),
        "p1_placement": str(args.p1_placement.resolve()),
        "p1_placement_sha256": sha256(args.p1_placement),
        "trace": str(trace_path.resolve()), "trace_sha256": trace_hash,
        "ttft_limit_ms": TTFT_LIMIT_MS, "tpot_limits_ms": SLOS,
        "speed_proxy": "frequency_mhz / 2520",
        "candidate_count": len(planned), "candidates": planned,
    })
    print(json.dumps({"output_dir": str(args.output_dir), "candidates": len(planned)}))


def _joint_rows(study: Path) -> list[dict[str, Any]]:
    plan = json.loads((study / "joint-plan.json").read_text(encoding="utf-8"))
    by_index = {int(row["candidate_index"]): row for row in plan["candidates"]}
    result = []
    seen: set[int] = set()
    for result_path in sorted((study / "measured").glob("candidate-*-attempt-*/result.json")):
        value = json.loads(result_path.read_text(encoding="utf-8"))
        if value.get("status") != "complete":
            continue
        index = int(value["candidate_index"])
        if index in seen:
            raise ValueError(f"multiple successful attempts for Joint candidate {index}")
        seen.add(index)
        candidate = by_index[index]
        run_dir = Path(value["run_dir"])
        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        result.append({**candidate, **metrics,
                       "frequency_tuple": tuple(candidate["frequency_tuple"]),
                       "run_dir": str(run_dir)})
    return result


def repeat_plan(args: argparse.Namespace) -> None:
    sequential = load_runs(args.sequential_controller)
    joint = _joint_rows(args.joint_study)
    joint_plan_value = json.loads((args.joint_study / "joint-plan.json").read_text())
    if len(joint) != int(joint_plan_value["candidate_count"]):
        raise SystemExit(
            f"repeat planning requires all Joint candidates; found {len(joint)}/"
            f"{joint_plan_value['candidate_count']} successful"
        )
    best_b4 = select_min_energy(sequential, ttft_limit_ms=TTFT_LIMIT_MS,
                                tpot_limit_ms=13.0)
    best_joint = select_min_energy(joint, ttft_limit_ms=TTFT_LIMIT_MS,
                                   tpot_limit_ms=13.0)
    if best_b4 is None or best_joint is None:
        raise SystemExit("both B4 and Joint need a feasible 13 ms configuration")
    p1_path = Path(joint_plan_value["p1_placement"])
    configurations = [
        {"name": "b4-medium-slo", "placement_name": "upstream_default_eplb",
         "placement_file": str(p1_path),
         "frequency_tuple": list(best_b4["frequency_tuple"]), "repetitions": 3,
         "comparison_reference_run": None},
        {"name": "joint-medium-slo", "placement_name": "frequency_aware",
         "placement_file": best_joint["placement_file"],
         "frequency_tuple": list(best_joint["frequency_tuple"]), "repetitions": 3,
         "comparison_reference_run": best_joint["sequential_run_dir"]},
    ]
    if tuple(best_joint["frequency_tuple"]) != tuple(best_b4["frequency_tuple"]):
        configurations.append(
            {"name": "p1-at-joint-medium-slo", "placement_name": "upstream_default_eplb",
             "placement_file": str(p1_path),
             "frequency_tuple": list(best_joint["frequency_tuple"]), "repetitions": 3,
             "comparison_reference_run": None}
        )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_json(args.output_dir / "repeat-plan.json", {
        "schema_version": 1, "ttft_limit_ms": TTFT_LIMIT_MS,
        "tpot_limit_ms": 13.0, "configurations": configurations,
        "sequential_controller": str(args.sequential_controller.resolve()),
        "joint_study": str(args.joint_study.resolve()),
    })
    print(json.dumps({"output_dir": str(args.output_dir),
                      "configurations": len(configurations), "repetitions_each": 3}))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    sequential = subparsers.add_parser("sequential-plan")
    sequential.add_argument("--output-dir", type=Path, required=True)
    sequential.add_argument("--seed", type=int, default=20260826)
    sequential.set_defaults(func=sequential_plan)
    joint = subparsers.add_parser("joint-plan")
    joint.add_argument("--sequential-controller", type=Path, required=True)
    joint.add_argument("--expert-counts", type=Path, required=True)
    joint.add_argument("--p1-placement", type=Path, required=True)
    joint.add_argument("--output-dir", type=Path, required=True)
    joint.add_argument("--max-candidates", type=int, default=8)
    joint.set_defaults(func=joint_plan)
    repeat = subparsers.add_parser("repeat-plan")
    repeat.add_argument("--sequential-controller", type=Path, required=True)
    repeat.add_argument("--joint-study", type=Path, required=True)
    repeat.add_argument("--output-dir", type=Path, required=True)
    repeat.set_defaults(func=repeat_plan)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
