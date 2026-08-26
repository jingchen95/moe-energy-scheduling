#!/usr/bin/env python3
"""Generate Milestone E processed tables and diagnostic figures from raw studies."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any, Iterable

from src.search import (
    pareto_frontier,
    paired_comparison,
    select_min_energy,
    slo_feasible,
    summarize_repeatability,
)
from tools.prepare_milestone_e import SLOS, TTFT_LIMIT_MS, _joint_rows, load_runs


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore",
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def clocks(row: dict[str, Any]) -> tuple[int, ...]:
    return tuple(int(value) for value in row["frequency_tuple"])


def enriched_sequential(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        value = dict(row)
        for gpu, frequency in enumerate(clocks(row)):
            value[f"f{gpu}"] = frequency
        for limit in SLOS:
            value[f"slo{int(limit)}_feasible"] = slo_feasible(
                row, ttft_limit_ms=TTFT_LIMIT_MS, tpot_limit_ms=limit,
            )
        output.append(value)
    return output


def divergence_for(study: Path, candidate_index: int) -> dict[str, Any]:
    paths = list((study / "measured").glob(
        f"candidate-{candidate_index:02d}-attempt-*/output-divergence.json"
    ))
    if len(paths) > 1:
        raise ValueError(f"multiple successful divergence records for candidate {candidate_index}")
    return json.loads(paths[0].read_text(encoding="utf-8")) if paths else {}


def joint_table(sequential: list[dict[str, Any]], study: Path) -> list[dict[str, Any]]:
    by_tuple = {clocks(row): row for row in sequential}
    rows = []
    for joint in _joint_rows(study):
        sequential_row = by_tuple[clocks(joint)]
        comparison = paired_comparison(sequential_row, joint)
        divergence = divergence_for(study, int(joint["candidate_index"]))
        value = {**joint, **comparison,
                 "exact_output_match_fraction": divergence.get("exact_match_fraction"),
                 "minimum_common_prefix_tokens": divergence.get("minimum_observed_prefix"),
                 "median_common_prefix_tokens": divergence.get("median_observed_prefix")}
        for gpu, frequency in enumerate(clocks(joint)):
            value[f"f{gpu}"] = frequency
        for limit in SLOS:
            value[f"slo{int(limit)}_feasible"] = slo_feasible(
                joint, ttft_limit_ms=TTFT_LIMIT_MS, tpot_limit_ms=limit,
            )
        rows.append(value)
    return rows


def _choose_or_fixed(rows: list[dict[str, Any]], limit: float) -> dict[str, Any] | None:
    selected = select_min_energy(rows, ttft_limit_ms=TTFT_LIMIT_MS, tpot_limit_ms=limit)
    if selected is not None:
        return dict(selected)
    return dict(min(rows, key=lambda row: float(row["total_gpu_energy_j"]))) if rows else None


def policy_summary(
    d_rows: list[dict[str, Any]], sequential: list[dict[str, Any]],
    joint: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    p0_high = next((row for row in d_rows if row.get("label") == "default_linear-uniform-high"), None)
    p1_high = [row for row in sequential if clocks(row) == (2520,) * 4]
    uniform = [row for row in sequential if len(set(clocks(row))) == 1]
    result = []
    for limit in SLOS:
        b4 = _choose_or_fixed(sequential, limit)
        policies = {
            "B1": dict(p0_high) if p0_high else None,
            "B2": dict(p1_high[0]) if p1_high else None,
            "B3": _choose_or_fixed(uniform, limit),
            "B4": b4,
            "Joint": _choose_or_fixed(joint, limit),
        }
        b1_energy = float(policies["B1"]["total_gpu_energy_j"]) if policies["B1"] else None
        b4_energy = float(b4["total_gpu_energy_j"]) if b4 else None
        for policy, row in policies.items():
            if row is None:
                result.append({"tpot_slo_ms": limit, "ttft_slo_ms": TTFT_LIMIT_MS,
                               "policy": policy, "slo_feasible": False})
                continue
            energy = float(row["total_gpu_energy_j"])
            result.append({
                "tpot_slo_ms": limit, "ttft_slo_ms": TTFT_LIMIT_MS, "policy": policy,
                "placement": row.get("placement_name", row.get("placement")),
                "frequency_tuple": ",".join(map(str, clocks(row))),
                "total_gpu_energy_j": energy,
                "joules_per_request": row["joules_per_request"],
                "ttft_p99_ms": row["ttft_p99_ms"], "tpot_p99_ms": row["tpot_p99_ms"],
                "slo_feasible": slo_feasible(row, ttft_limit_ms=TTFT_LIMIT_MS,
                                               tpot_limit_ms=limit),
                "energy_saving_vs_b1_pct": ((b1_energy - energy) / b1_energy * 100
                                             if b1_energy else None),
                "energy_saving_vs_b4_pct": ((b4_energy - energy) / b4_energy * 100
                                             if b4_energy else None),
                "run_dir": row.get("run_dir"),
            })
    return result


def repeatability(repeat_study: Path | None) -> list[dict[str, Any]]:
    if repeat_study is None:
        return []
    plan = json.loads((repeat_study / "repeat-plan.json").read_text(encoding="utf-8"))
    output = []
    for index, configuration in enumerate(plan["configurations"]):
        paths = list((repeat_study / "measured").glob(
            f"configuration-{index:02d}-attempt-*/result.json"
        ))
        successful = [path for path in paths
                      if json.loads(path.read_text(encoding="utf-8")).get("status") == "complete"]
        if not successful:
            continue
        if len(successful) > 1:
            raise ValueError(f"multiple successful repeat attempts for configuration {index}")
        result = json.loads(successful[0].read_text(encoding="utf-8"))
        rows = [json.loads((Path(path) / "metrics.json").read_text(encoding="utf-8"))
                for path in result["run_dirs"]]
        summary = summarize_repeatability(rows, ttft_limit_ms=TTFT_LIMIT_MS,
                                          tpot_limit_ms=13.0)
        divergence_paths = sorted(successful[0].parent.glob("output-divergence-rep*.json"))
        if divergence_paths:
            divergence = [json.loads(path.read_text(encoding="utf-8"))
                          for path in divergence_paths]
            summary.update({
                "output_exact_match_fraction_mean": statistics.mean(
                    row["exact_match_fraction"] for row in divergence
                ),
                "minimum_common_prefix_tokens": min(
                    row["minimum_observed_prefix"] for row in divergence
                ),
                "median_common_prefix_tokens": statistics.median(
                    row["median_observed_prefix"] for row in divergence
                ),
            })
        output.append({"configuration": configuration["name"],
                       "placement": configuration["placement_name"],
                       "frequency_tuple": ",".join(map(str, configuration["frequency_tuple"])),
                       **summary})
    return output


def joint_extends_pareto(
    sequential: list[dict[str, Any]], joint: list[dict[str, Any]],
) -> bool:
    for candidate in joint:
        dominated = any(
            float(row["tpot_p99_ms"]) <= float(candidate["tpot_p99_ms"])
            and float(row["joules_per_request"]) <= float(candidate["joules_per_request"])
            for row in sequential
        )
        if not dominated:
            return True
    return False


def classify_result(
    repeats: list[dict[str, Any]], *, pareto_extended: bool,
    paired: list[dict[str, Any]],
) -> tuple[str, dict[str, Any]]:
    by_name = {row["configuration"]: row for row in repeats}
    b4 = by_name.get("b4-medium-slo")
    joint = by_name.get("joint-medium-slo")
    if b4 is None or joint is None or b4["trials"] < 3 or joint["trials"] < 3:
        return "INCONCLUSIVE", {"reason": "main three-trial comparison is incomplete"}
    b4_energy = float(b4["total_gpu_energy_j_mean"])
    joint_energy = float(joint["total_gpu_energy_j_mean"])
    saving = (b4_energy - joint_energy) / b4_energy * 100
    details = {"b4_energy_mean_j": b4_energy, "joint_energy_mean_j": joint_energy,
               "joint_energy_saving_vs_b4_pct": saving,
               "both_robustly_slo_feasible": bool(b4["robustly_slo_feasible"]
                                                   and joint["robustly_slo_feasible"]),
               "joint_extends_measured_pareto": pareto_extended}
    if not details["both_robustly_slo_feasible"]:
        return "INCONCLUSIVE", {**details, "reason": "selected B4 or Joint failed an SLO trial"}
    nonoverlap = (float(joint["total_gpu_energy_j_max"])
                  < float(b4["total_gpu_energy_j_min"]))
    if saving > 0 and nonoverlap:
        return "SUPPORTED", {**details, "reason": "Joint energy is lower in all repeated trials"}
    if saving > 0 or pareto_extended or any(
            float(row["delta_tpot_p99_ms"]) < 0 for row in paired):
        return "WEAKLY SUPPORTED", {
            **details, "reason": "paired/frontier evidence exists but the B4 energy gap is not robust"
        }
    return "NOT SUPPORTED", {
        **details, "reason": "exhaustive B4 matches or dominates all measured Joint candidates"
    }


def make_figures(
    sequential: list[dict[str, Any]], joint: list[dict[str, Any]],
    policies: list[dict[str, Any]], figure_dir: Path,
) -> None:
    import matplotlib.pyplot as plt
    figure_dir.mkdir(parents=True, exist_ok=True)
    medium = [row for row in policies if row["tpot_slo_ms"] == 13.0]
    figure, axis = plt.subplots(figsize=(6.2, 3.8))
    labels = [row["policy"] for row in medium]
    values = [row.get("joules_per_request", 0) for row in medium]
    colors = ["#287271" if row.get("slo_feasible") else "#b7b7b7" for row in medium]
    bars = axis.bar(labels, values, color=colors)
    for bar, row in zip(bars, medium):
        if not row.get("slo_feasible"):
            axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), "infeasible",
                      ha="center", va="bottom", fontsize=8)
    axis.set_ylabel("GPU energy (J/request)")
    axis.set_title("TTFT <= 500 ms, TPOT <= 13 ms")
    figure.tight_layout()
    figure.savefig(figure_dir / "milestone_e_main.pdf")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6.2, 4.0))
    axis.scatter([row["tpot_p99_ms"] for row in sequential],
                 [row["joules_per_request"] for row in sequential],
                 s=18, alpha=0.45, label="B4 Sequential (81)")
    if joint:
        axis.scatter([row["tpot_p99_ms"] for row in joint],
                     [row["joules_per_request"] for row in joint],
                     marker="D", s=38, label="Joint candidates")
    frontier = pareto_frontier(sequential, y_key="joules_per_request")
    axis.plot([row["tpot_p99_ms"] for row in frontier],
              [row["joules_per_request"] for row in frontier], color="#1b6ca8", linewidth=1)
    axis.axvline(13.0, color="#9b2226", linestyle="--", linewidth=1, label="13 ms SLO")
    axis.set_xlabel("p99 TPOT (ms)")
    axis.set_ylabel("GPU energy (J/request)")
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(figure_dir / "milestone_e_pareto.pdf")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--milestone-d-controller", type=Path, required=True)
    parser.add_argument("--sequential-controller", type=Path, required=True)
    parser.add_argument("--joint-study", type=Path, required=True)
    parser.add_argument("--repeat-study", type=Path)
    parser.add_argument("--processed-dir", type=Path, default=Path("results/processed"))
    parser.add_argument("--figure-dir", type=Path, default=Path("results/figures"))
    args = parser.parse_args()

    sequential = enriched_sequential(load_runs(args.sequential_controller))
    if len(sequential) != 81:
        raise SystemExit(f"expected 81 successful Sequential runs, found {len(sequential)}")
    joint = joint_table(sequential, args.joint_study)
    d_rows = load_runs(args.milestone_d_controller)
    policies = policy_summary(d_rows, sequential, joint)
    repeats = repeatability(args.repeat_study)
    repeats_by_name = {row["configuration"]: row for row in repeats}
    for row in policies:
        if row["tpot_slo_ms"] != 13.0:
            row["robustly_slo_feasible"] = None
        elif row["policy"] == "B4" and "b4-medium-slo" in repeats_by_name:
            row["robustly_slo_feasible"] = repeats_by_name[
                "b4-medium-slo"
            ]["robustly_slo_feasible"]
        elif row["policy"] == "Joint" and "joint-medium-slo" in repeats_by_name:
            row["robustly_slo_feasible"] = repeats_by_name[
                "joint-medium-slo"
            ]["robustly_slo_feasible"]
        else:
            row["robustly_slo_feasible"] = None

    common = ["run_id", "run_label", "run_dir", "f0", "f1", "f2", "f3",
              "ttft_p99_ms", "tpot_p99_ms", "e2e_p99_ms", "total_gpu_energy_j",
              "joules_per_request", "joules_per_output_token"]
    gpu_fields = [f"gpu{gpu}_{suffix}" for gpu in range(4)
                  for suffix in ("energy_j", "avg_power_w", "avg_sm_util",
                                 "avg_memory_util", "avg_sm_clock")]
    write_csv(args.processed_dir / "milestone_e_sequential_all.csv", sequential,
              common + gpu_fields + [f"slo{int(limit)}_feasible" for limit in SLOS])
    write_csv(args.processed_dir / "milestone_e_joint_candidates.csv", joint,
              ["candidate_index", "label", "reason", *common,
               "delta_ttft_p99_ms", "delta_tpot_p99_ms", "delta_e2e_p99_ms",
               "delta_total_gpu_energy_j", "delta_joules_per_request",
               "exact_output_match_fraction", "minimum_common_prefix_tokens",
               "median_common_prefix_tokens", "predicted_raw_load",
               "predicted_normalized_load", "normalized_max_over_mean"])
    write_csv(args.processed_dir / "milestone_e_policy_summary.csv", policies,
              ["tpot_slo_ms", "ttft_slo_ms", "policy", "placement", "frequency_tuple",
               "total_gpu_energy_j", "joules_per_request", "ttft_p99_ms", "tpot_p99_ms",
               "slo_feasible", "robustly_slo_feasible", "energy_saving_vs_b1_pct",
               "energy_saving_vs_b4_pct", "run_dir"])
    repeat_fields = ["configuration", "placement", "frequency_tuple", "trials",
                     "feasible_trials", "robustly_slo_feasible"]
    for metric in ("ttft_p99_ms", "tpot_p99_ms", "total_gpu_energy_j", "joules_per_request"):
        repeat_fields.extend(f"{metric}_{suffix}" for suffix in ("mean", "stdev", "min", "max"))
    repeat_fields.extend(("output_exact_match_fraction_mean",
                          "minimum_common_prefix_tokens", "median_common_prefix_tokens"))
    write_csv(args.processed_dir / "milestone_e_repeatability.csv", repeats, repeat_fields)
    pareto_extended = joint_extends_pareto(sequential, joint)
    classification, classification_details = classify_result(
        repeats, pareto_extended=pareto_extended, paired=joint,
    )
    summary = {"sequential_successful": len(sequential), "joint_measured": len(joint),
               "common_slos": {"ttft_p99_ms": TTFT_LIMIT_MS, "tpot_p99_ms": SLOS},
               "policy_summary": policies, "paired_joint_results": joint,
               "repeatability": repeats, "joint_extends_measured_pareto": pareto_extended,
               "classification": classification,
               "classification_details": classification_details}
    (args.processed_dir / "milestone_e_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=list) + "\n", encoding="utf-8")
    make_figures(sequential, joint, policies, args.figure_dir)
    print(json.dumps({"sequential": len(sequential), "joint": len(joint),
                      "repeatability_groups": len(repeats)}))


if __name__ == "__main__":
    main()
