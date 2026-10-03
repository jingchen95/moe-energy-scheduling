#!/usr/bin/env python3
"""Build and validate rank-specific AFD FFN latency and power models."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.afd_cost_model import (
    fit_binned_curve,
    fit_linear_power,
    join_timing_run,
    leave_one_group_out_power,
    load_routing_trace,
    prediction_metrics,
    write_csv,
)
from src.afd_static_placement import load_placement


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _read_runs(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as stream:
        header = stream.readline().rstrip("\n").split("\t")
        return [dict(zip(header, line.rstrip("\n").split("\t"))) for line in stream]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_gzip_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _correlation(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def build_latency_analysis(
    root: Path, study: Path, processed: Path
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    plan = json.loads((study / "manifest.json").read_text())
    run_rows = _read_runs(study / "runs.tsv")
    routing_row = next(row for row in run_rows if row["stage"] == "routing")
    routing_path = Path(routing_row["run_dir"]) / "routing_trace.npz"
    routing = load_routing_trace(routing_path)
    by_config = {row["config_id"]: row for row in plan["configurations"]}
    joined: list[dict[str, Any]] = []
    for row in run_rows:
        if row["stage"] != "timing" or row["status"] != "complete":
            continue
        definition = by_config[row["config_id"]]
        run_dir = Path(row["run_dir"])
        timing_paths = sorted(run_dir.glob("ffn-timing-rank-*.npz"))
        placement = root / definition["placement"]["path"]
        current, _ = join_timing_run(
            routing,
            timing_paths,
            placement,
            config_id=row["config_id"],
            lifecycle=row["lifecycle"],
            frequency_mhz=int(row["frequency_mhz"]),
        )
        joined.extend(current)
    expected_configs = int(plan["expected_timing_runs"])
    if len({row["config_id"] for row in joined}) != expected_configs:
        raise ValueError("not all timing configurations were joined")
    joined_path = processed / "milestone_n_joined_invocations.csv.gz"
    _write_gzip_csv(joined_path, joined)

    model_rows: list[dict[str, Any]] = []
    validations: list[dict[str, Any]] = []
    residual_rows: list[dict[str, Any]] = []
    curves: dict[tuple[int, int], Any] = {}
    train = [row for row in joined if row["lifecycle"] == "primary"]
    for rank in (0, 1):
        for frequency in (1005, 1635, 2070, 2220, 2520):
            subset = [
                row for row in train
                if row["physical_ffn_rank"] == rank and row["frequency_mhz"] == frequency
            ]
            curve = fit_binned_curve(
                np.asarray([row["n"] for row in subset]),
                np.asarray([row["elapsed_ms"] for row in subset]),
            )
            curves[(rank, frequency)] = curve
            for index, (center, value, count) in enumerate(
                zip(curve.centers, curve.values, curve.counts)
            ):
                model_rows.append({
                    "physical_ffn_rank": rank,
                    "physical_gpu": rank + 2,
                    "frequency_mhz": frequency,
                    "bin": index,
                    "n_center": center,
                    "elapsed_ms_median": value,
                    "samples": int(count),
                })

    evaluation_sets = {
        "heldout_lifecycle": [row for row in joined if row["lifecycle"] == "heldout_repeat"],
        "heldout_distribution": [row for row in joined if row["lifecycle"] == "heldout_skew"],
    }
    for evaluation, rows in evaluation_sets.items():
        for rank in (0, 1):
            for frequency in (1635, 2070):
                subset = [
                    row for row in rows
                    if row["physical_ffn_rank"] == rank and row["frequency_mhz"] == frequency
                ]
                actual = np.asarray([row["elapsed_ms"] for row in subset])
                predicted = curves[(rank, frequency)].predict(
                    np.asarray([row["n"] for row in subset])
                )
                metrics = prediction_metrics(actual, predicted)
                validations.append({
                    "evaluation": evaluation,
                    "physical_ffn_rank": rank,
                    "physical_gpu": rank + 2,
                    "frequency_mhz": frequency,
                    **metrics,
                    "n_min": int(min(row["n"] for row in subset)),
                    "n_max": int(max(row["n"] for row in subset)),
                })
                residual = predicted - actual
                residual_rows.append({
                    "evaluation": evaluation,
                    "physical_ffn_rank": rank,
                    "frequency_mhz": frequency,
                    "corr_active_local_experts": _correlation(
                        residual, np.asarray([row["active_local_experts"] for row in subset])
                    ),
                    "corr_max_expert_load": _correlation(
                        residual,
                        np.asarray([row["max_tokens_per_local_expert"] for row in subset]),
                    ),
                    "corr_routing_concentration": _correlation(
                        residual, np.asarray([row["routing_concentration"] for row in subset])
                    ),
                    "corr_layer": _correlation(
                        residual, np.asarray([row["layer"] for row in subset])
                    ),
                })
    write_csv(processed / "milestone_n_latency_model.csv", model_rows)
    write_csv(processed / "milestone_n_latency_validation.csv", validations)
    write_csv(processed / "milestone_n_latency_residuals.csv", residual_rows)

    rank_comparison = []
    for frequency in (1005, 1635, 2070, 2220, 2520):
        medians = []
        for rank in (0, 1):
            values = [
                row["elapsed_ms"] for row in train
                if row["frequency_mhz"] == frequency
                and row["physical_ffn_rank"] == rank
            ]
            medians.append(float(np.median(values)))
        rank_comparison.append({
            "frequency_mhz": frequency,
            "gpu2_median_ms": medians[0],
            "gpu3_median_ms": medians[1],
            "gpu3_minus_gpu2_percent": (medians[1] - medians[0]) / medians[0] * 100,
        })
    write_csv(processed / "milestone_n_rank_comparison.csv", rank_comparison)
    result = {
        "routing_trace": str(routing_path),
        "routing_trace_sha256": _sha256(routing_path),
        "routing_records": int(len(routing["layers"])),
        "joined_rows": len(joined),
        "joined_dataset": str(joined_path),
        "joined_dataset_sha256": _sha256(joined_path),
        "validation": validations,
        "residual_correlations": residual_rows,
        "rank_comparison": rank_comparison,
    }
    return result, joined


def build_power_dataset(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    symmetric = _read_csv(root / "results/processed/milestone_m_symmetric_runs.csv")
    targets = _read_csv(root / "results/processed/milestone_m_target_split_runs.csv")
    placement_summary = _read_csv(
        root / "results/processed/milestone_m_target_split_placements.csv"
    )
    profile_path = (
        root / "results/raw/milestone-m-afd-profile-20260930T191548Z/runs/"
        "c-math-20260930T191743.393254Z-9ad557de/expert_counts.npz"
    )
    with np.load(profile_path, allow_pickle=False) as archive:
        counts = np.asarray(archive["expert_counts"], dtype=np.int64)
    balanced_mapping, _, _ = load_placement(
        str(root / "data/placements/milestone_m/P_balanced.json")
    )
    mapping = np.asarray(balanced_mapping, dtype=np.int64)
    balanced_loads = np.asarray([
        counts[mapping == rank].sum() for rank in range(2)
    ], dtype=np.int64)
    summary_by_name = {row["placement"]: row for row in placement_summary}
    output: list[dict[str, Any]] = []

    def append(source: str, row: dict[str, str], loads: np.ndarray) -> None:
        duration = float(row["duration_s"])
        clocks = [int(float(value)) for value in row["target_clocks_mhz"].split("/")]
        for rank in (0, 1):
            gpu = rank + 2
            energy = float(row[f"gpu{gpu}_energy_j"])
            output.append({
                "source": source,
                "config_id": row["config_id"],
                "run_id": row["run_id"],
                "rank": rank,
                "physical_gpu": gpu,
                "frequency_mhz": clocks[gpu],
                "routed_assignments": int(loads[rank]),
                "load_rate": float(loads[rank]) / duration,
                "duration_s": duration,
                "avg_power_w": float(row[f"gpu{gpu}_avg_power_w"]),
                "window_energy_j": energy,
            })

    for row in symmetric:
        append("symmetric_uninstrumented", row, balanced_loads)
    for row in targets:
        summary = summary_by_name[row["placement"]]
        loads = np.asarray([int(summary["rank0_load"]), int(summary["rank1_load"])])
        append("target_split_uninstrumented", row, loads)
    provenance = {
        "symmetric_runs_sha256": _sha256(
            root / "results/processed/milestone_m_symmetric_runs.csv"
        ),
        "target_split_runs_sha256": _sha256(
            root / "results/processed/milestone_m_target_split_runs.csv"
        ),
        "target_split_placements_sha256": _sha256(
            root / "results/processed/milestone_m_target_split_placements.csv"
        ),
        "profile_counts_sha256": _sha256(
            root / "results/raw/milestone-m-afd-profile-20260930T191548Z/runs/"
            "c-math-20260930T191743.393254Z-9ad557de/expert_counts.npz"
        ),
    }
    return output, provenance


def build_power_analysis(root: Path, processed: Path) -> dict[str, Any]:
    rows, provenance = build_power_dataset(root)
    write_csv(processed / "milestone_n_power_dataset.csv", rows)
    restricted = [row for row in rows if row["frequency_mhz"] in (1635, 2070)]
    predictions = leave_one_group_out_power(restricted)
    write_csv(processed / "milestone_n_power_validation_rows.csv", predictions)
    validation: list[dict[str, Any]] = []
    models: list[dict[str, Any]] = []
    for rank in (0, 1):
        for frequency in (1635, 2070):
            subset = [
                row for row in restricted
                if row["rank"] == rank and row["frequency_mhz"] == frequency
            ]
            intercept, slope = fit_linear_power(
                np.asarray([row["load_rate"] for row in subset]),
                np.asarray([row["avg_power_w"] for row in subset]),
            )
            models.append({
                "rank": rank,
                "physical_gpu": rank + 2,
                "frequency_mhz": frequency,
                "intercept_w": intercept,
                "slope_w_per_assignment_s": slope,
                "load_rate_min": min(row["load_rate"] for row in subset),
                "load_rate_max": max(row["load_rate"] for row in subset),
                "samples": len(subset),
            })
            heldout = [
                row for row in predictions
                if row["rank"] == rank and row["frequency_mhz"] == frequency
            ]
            power_metrics = prediction_metrics(
                np.asarray([row["avg_power_w"] for row in heldout]),
                np.asarray([row["predicted_power_w"] for row in heldout]),
            )
            energy_metrics = prediction_metrics(
                np.asarray([row["window_energy_j"] for row in heldout]),
                np.asarray([row["predicted_energy_j"] for row in heldout]),
            )
            validation.append({
                "rank": rank,
                "physical_gpu": rank + 2,
                "frequency_mhz": frequency,
                **{f"power_{key}": value for key, value in power_metrics.items()},
                **{f"energy_{key}": value for key, value in energy_metrics.items()},
            })
    write_csv(processed / "milestone_n_power_model.csv", models)
    write_csv(processed / "milestone_n_power_validation.csv", validation)
    return {"provenance": provenance, "models": models, "validation": validation}


def make_plots(
    joined: list[dict[str, Any]], power_rows: list[dict[str, Any]], figures: Path
) -> None:
    import matplotlib.pyplot as plt

    figures.mkdir(parents=True, exist_ok=True)
    primary = [row for row in joined if row["lifecycle"] == "primary"]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for rank, axis in enumerate(axes):
        for frequency in (1005, 1635, 2070, 2220, 2520):
            rows = [
                row for row in primary
                if row["physical_ffn_rank"] == rank and row["frequency_mhz"] == frequency
            ][::20]
            axis.scatter([row["n"] for row in rows], [row["elapsed_ms"] for row in rows],
                         s=5, alpha=0.35, label=str(frequency))
        axis.set_title(f"Physical GPU {rank + 2}")
        axis.set_xlabel("Local routed assignments n")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("FFN boundary elapsed (ms)")
    axes[1].legend(title="MHz", fontsize=8)
    figure.tight_layout()
    figure.savefig(figures / "milestone_n_latency_vs_load.pdf")
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for rank, axis in enumerate(axes):
        for frequency in (1635, 2070):
            rows = [
                row for row in power_rows
                if row["rank"] == rank and row["frequency_mhz"] == frequency
            ]
            axis.scatter([row["load_rate"] for row in rows],
                         [row["avg_power_w"] for row in rows], label=str(frequency))
        axis.set_title(f"Physical GPU {rank + 2}")
        axis.set_xlabel("Routed assignments / measurement second")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Average power (W)")
    axes[1].legend(title="MHz")
    figure.tight_layout()
    figure.savefig(figures / "milestone_n_power_vs_load.pdf")
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument("--processed-dir", type=Path, default=Path("results/processed"))
    parser.add_argument("--figures-dir", type=Path, default=Path("results/figures"))
    args = parser.parse_args()
    root = Path.cwd().resolve()
    processed = args.processed_dir.resolve()
    processed.mkdir(parents=True, exist_ok=True)
    latency, joined = build_latency_analysis(root, args.study_dir.resolve(), processed)
    power = build_power_analysis(root, processed)
    power_rows, _ = build_power_dataset(root)
    make_plots(joined, power_rows, args.figures_dir.resolve())
    summary = {
        "schema_version": 1,
        "latency": latency,
        "power": power,
        "secondary_per_call_energy_surrogate": (
            "power-model estimate times CUDA-event latency; not direct per-call energy"
        ),
    }
    (processed / "milestone_n_model_validation.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
