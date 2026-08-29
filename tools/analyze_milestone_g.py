#!/usr/bin/env python3
"""Analyze the controlled Milestone G load/frequency characterization."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.moe_timing import load_timing_files
from src.placement import placement_loads, placement_from_dict
from src.service_model import (
    fit_amdahl,
    fit_service_time_model,
    prediction_metrics,
    predict_service_time,
)

FMAX = 2520.0
FREQUENCIES = (2520, 1635, 1005)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def correlation(left: list[float], right: list[float]) -> dict[str, float]:
    x = np.asarray(left, dtype=float)
    y = np.asarray(right, dtype=float)
    if x.size < 2 or x.size != y.size or np.std(x) == 0 or np.std(y) == 0:
        return {"n": float(x.size), "pearson": float("nan")}
    return {"n": float(x.size), "pearson": float(np.corrcoef(x, y)[0, 1])}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"cannot write empty table: {path}")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_profile(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        total = np.asarray(archive["expert_counts"], dtype=np.int64)
        decode = np.asarray(archive["decode_expert_counts"], dtype=np.int64)
    if total.shape != decode.shape or np.any(decode < 0) or np.any(decode > total):
        raise ValueError("invalid phase profile")
    return total, decode


def matrix_rows(plan_dir: Path) -> list[dict[str, Any]]:
    plan = json.loads((plan_dir / "matrix-plan.json").read_text(encoding="utf-8"))
    placement_by_name = {row["name"]: row for row in plan["placements"]}
    rows: list[dict[str, Any]] = []
    for controller in sorted((plan_dir / "measured").glob("*-attempt-*")):
        manifest = controller / "runs.tsv"
        if not manifest.exists():
            continue
        with manifest.open(newline="", encoding="utf-8") as stream:
            for entry in csv.DictReader(stream, delimiter="\t"):
                run_dir = Path(entry["run_dir"])
                metrics_path = run_dir / "metrics.json"
                if not metrics_path.exists():
                    raise ValueError(f"missing metrics: {metrics_path}")
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                timing_paths = [run_dir / f"moe_timing_rank{rank}.json" for rank in range(4)]
                if not all(path.exists() for path in timing_paths):
                    raise ValueError(f"missing rank-local timing file under {run_dir}")
                timing = load_timing_files(timing_paths)
                clocks = tuple(int(value) for value in metrics["target_clocks_mhz"])
                placement = entry["label"].split("-gpu0-")[0]
                placement_row = placement_by_name[placement]
                row: dict[str, Any] = {
                    "run_dir": str(run_dir),
                    "controller": str(controller),
                    "label": entry["label"],
                    "placement": placement,
                    "load_level": placement.removeprefix("controlled_").upper(),
                    "frequency_mhz": clocks[0],
                    "frequency_tuple": json.dumps(clocks),
                    "is_initial": entry["label"].endswith("-initial"),
                    "is_repeat": "-repeat" in entry["label"],
                    "tpot_p99_ms": float(metrics["tpot_p99_ms"]),
                    "ttft_p99_ms": float(metrics["ttft_p99_ms"]),
                    "e2e_p99_ms": float(metrics["e2e_p99_ms"]),
                    "total_gpu_energy_j": float(metrics["total_gpu_energy_j"]),
                    "output_tokens_total": int(metrics.get("output_tokens_total", 0)),
                    "failed_requests": int(metrics.get("failed_requests", 0)),
                    "moe_record_count": int(timing["record_count"]),
                    "moe_max_rank_total_ms": float(timing["max_rank_total_ms"]),
                    "moe_max_rank_prefill_ms": float(timing["max_rank_prefill_ms"]),
                    "moe_max_rank_decode_ms": float(timing["max_rank_decode_ms"]),
                    "moe_decode_ms_per_row": float(
                        timing["max_rank_decode_ms"] / max(
                            (value["decode_rows"] for value in timing["per_rank"].values()),
                            default=1.0,
                        )
                    ),
                    "moe_sum_rank_total_ms": float(timing["sum_rank_total_ms"]),
                    "moe_decode_rows_max_rank": float(max(
                        (value["decode_rows"] for value in timing["per_rank"].values()),
                        default=0.0,
                    )),
                    "decode_load_g0": int(placement_row["decode_load"][0]),
                    "decode_load_g1": int(placement_row["decode_load"][1]),
                    "decode_load_g2": int(placement_row["decode_load"][2]),
                    "decode_load_g3": int(placement_row["decode_load"][3]),
                    "aggregate_load_g0": int(placement_row["aggregate_load"][0]),
                    "aggregate_load_g1": int(placement_row["aggregate_load"][1]),
                    "aggregate_load_g2": int(placement_row["aggregate_load"][2]),
                    "aggregate_load_g3": int(placement_row["aggregate_load"][3]),
                    "decode_max_over_mean": float(placement_row["decode_max_over_mean"]),
                    "trace_sha256": plan["trace_sha256"],
                    "placement_sha256": placement_row["sha256"],
                }
                for gpu, target in enumerate(clocks):
                    actual = metrics.get(f"gpu{gpu}_avg_sm_clock")
                    row[f"gpu{gpu}_actual_clock_mhz"] = float(actual) if actual is not None else None
                    row[f"gpu{gpu}_energy_j"] = float(metrics[f"gpu{gpu}_energy_j"])
                    row[f"gpu{gpu}_avg_power_w"] = float(metrics[f"gpu{gpu}_avg_power_w"])
                    row[f"gpu{gpu}_avg_sm_util"] = float(metrics[f"gpu{gpu}_avg_sm_util"])
                rows.append(row)
    rows.sort(key=lambda row: (row["placement"], row["frequency_mhz"], row["label"]))
    initial = [row for row in rows if row["is_initial"]]
    if len(initial) != 12:
        raise ValueError(f"expected 12 initial matrix runs, found {len(initial)}")
    if any(row["failed_requests"] or row["output_tokens_total"] != 6400 for row in rows):
        raise ValueError("matrix contains failed or incomplete requests")
    if len({row["trace_sha256"] for row in rows}) != 1:
        raise ValueError("matrix runs do not share one trace hash")
    return rows


def repeatability_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = {"LOW", "HIGH", "VERY_HIGH"}
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["placement"], row["frequency_mhz"])
        if row["load_level"] in selected and row["frequency_mhz"] in (2520, 1635):
            grouped.setdefault(key, []).append(row)
    result = []
    for (placement, frequency), values in sorted(grouped.items()):
        if len(values) < 3:
            continue
        result.append({
            "placement": placement,
            "load_level": values[0]["load_level"],
            "frequency_mhz": frequency,
            "n_trials": len(values),
            "tpot_p99_mean_ms": float(np.mean([v["tpot_p99_ms"] for v in values])),
            "tpot_p99_std_ms": float(np.std([v["tpot_p99_ms"] for v in values], ddof=1)),
            "moe_decode_ms_per_row_mean": float(np.mean([v["moe_decode_ms_per_row"] for v in values])),
            "moe_decode_ms_per_row_std": float(np.std([v["moe_decode_ms_per_row"] for v in values], ddof=1)),
            "energy_j_mean": float(np.mean([v["total_gpu_energy_j"] for v in values])),
            "energy_j_std": float(np.std([v["total_gpu_energy_j"] for v in values], ddof=1)),
        })
    return result


def fit_cv(loads: np.ndarray, frequencies: np.ndarray, observed: np.ndarray) -> tuple[dict[str, Any], dict[str, float]]:
    model = fit_service_time_model(loads, frequencies, observed, fmax_mhz=FMAX)
    loo: list[float] = []
    for index in range(len(observed)):
        keep = np.ones(len(observed), dtype=bool)
        keep[index] = False
        fitted = fit_service_time_model(loads[keep], frequencies[keep], observed[keep], fmax_mhz=FMAX)
        loo.append(float(predict_service_time(fitted, loads[index:index + 1], frequencies[index:index + 1])[0]))
    return model, prediction_metrics(observed, loo)


def calibrated_pressure_metrics(pressure: np.ndarray, observed: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    design = np.column_stack((np.ones_like(pressure), pressure))
    coefficients, _, _, _ = np.linalg.lstsq(design, observed, rcond=None)
    predicted = design @ coefficients
    return predicted, prediction_metrics(observed, predicted)


def make_plot(path: Path, rows: list[dict[str, Any]], model_rows: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt
    colors = {"LOW": "#2563eb", "MEDIUM": "#16a34a", "HIGH": "#d97706", "VERY_HIGH": "#dc2626"}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for level, color in colors.items():
        values = [row for row in rows if row["is_initial"] and row["load_level"] == level]
        values.sort(key=lambda row: row["frequency_mhz"], reverse=True)
        if not values:
            continue
        high = values[0]
        for target, marker in ((1635, "o"), (1005, "s")):
            target_row = next(row for row in values if row["frequency_mhz"] == target)
            axes[0].scatter(high["decode_load_g0"], target_row["tpot_p99_ms"] - high["tpot_p99_ms"], color=color, marker=marker, label=f"{level} {target} MHz")
            axes[1].scatter(high["decode_load_g0"], target_row["moe_decode_ms_per_row"] - high["moe_decode_ms_per_row"], color=color, marker=marker)
    axes[0].set(xlabel="GPU0 decode routed tokens", ylabel="TPOT p99 penalty (ms)")
    axes[1].set(xlabel="GPU0 decode routed tokens", ylabel="MoE decode service penalty (ms / routed row)")
    axes[0].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for basis, axis in (("moe", axes[0]), ("tpot", axes[1])):
        actual = [float(row[f"actual_{basis}_ms"]) for row in model_rows]
        predicted = [float(row[f"predicted_{basis}_ms"]) for row in model_rows]
        axis.scatter(actual, predicted)
        lower = min(actual + predicted)
        upper = max(actual + predicted)
        axis.plot([lower, upper], [lower, upper], "k--", linewidth=0.8)
        axis.set(xlabel="measured (ms)", ylabel="G model predicted (ms)", title=basis.upper())
    fig.tight_layout()
    fig.savefig(path.with_name("milestone_g_predicted_vs_actual.pdf"))
    plt.close(fig)


def write_doc(
    path: Path, summary: dict[str, Any], rows: list[dict[str, Any]],
    model_rows: list[dict[str, Any]], candidate_rows: list[dict[str, Any]],
) -> None:
    lines = [
        "# Milestone G — Controlled Load–Frequency Service Characterization",
        "",
        "## Research questions",
        "",
        "This milestone measures whether GPU0 DVFS sensitivity grows with controlled decode expert workload and whether a small `T(L,f)` model is accurate enough for a future offline Joint pilot.",
        "",
        "## Platform and timing",
        "",
        "Qwen/Qwen3-30B-A3B, vLLM 0.15.1, TP=1/DP=4/EP=4, four L40S GPUs on one Uranus node, `allgather_reducescatter`, fixed 50-request GSM8K trace at 0.5 requests/s, and clocks H/M/L = 2520/1635/1005 MHz. The MoE interval is one CUDA-event region around FusedMoE dispatch, expert compute, and combine on each rank. Events are rank-local and resolved only at the explicit profile dump; no collective runs in the measured forward. vLLM 0.15.1 cleared phase kwargs before the timing wrapper could retain them, so unknown records are classified explicitly by the observed non-speculative row boundary: 1–2 rows decode and >=129 rows prefill.",
        "",
        "## Controlled load axis",
        "",
        "| level | GPU0 decode load | GPU1 | GPU2 | GPU3 | decode max/mean |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in json.loads((Path(summary["plan_dir"]) / "matrix-plan.json").read_text())["placements"]:
        lines.append(f"| {row['name']} | {row['decode_load'][0]} | {row['decode_load'][1]} | {row['decode_load'][2]} | {row['decode_load'][3]} | {row['decode_max_over_mean']:.3f} |")
    lines.extend([
        "",
        "## DVFS penalties",
        "",
        "Deltas are the target frequency minus the corresponding all-high run for the same placement.",
        "",
        "| level | GPU0 decode load | transition | TPOT p99 penalty (ms) | MoE decode service penalty (ms / routed row) | energy delta (J) |",
        "|---|---:|---|---:|---:|---:|",
    ])
    for row in summary["penalty_rows"]:
        lines.append(f"| {row['load_level']} | {row['gpu0_decode_load']} | {row['frequency_transition']} | {row['tpot_penalty_ms']:.3f} | {row['moe_decode_penalty_ms']:.3f} | {row['energy_delta_j']:.1f} |")
    lines.extend([
        "",
        f"TPOT penalty correlation with GPU0 decode load: Pearson {summary['penalty_correlations']['tpot']['pearson']:.3f}; MoE decode service penalty correlation: Pearson {summary['penalty_correlations']['moe_decode_service']['pearson']:.3f}. These are four load levels × two transitions and are not a population estimate.",
        "",
        "## Model comparison",
        "",
        "| model | target | MAE (ms) | MAPE (%) | max abs (ms) | rank accuracy | LOO MAE (ms) | LOO MAPE (%) | LOO max abs (ms) | LOO rank accuracy |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in model_rows:
        lines.append(f"| {row['model']} | {row['target']} | {row['mae_ms']:.3f} | {row['mape_percent']:.2f} | {row['max_abs_error_ms']:.3f} | {row['rank_order_accuracy']:.3f} | {row['loo_mae_ms'] if row['loo_mae_ms'] is not None else 'n/a'} | {row['loo_mape_percent'] if row['loo_mape_percent'] is not None else 'n/a'} | {row['loo_max_abs_error_ms'] if row['loo_max_abs_error_ms'] is not None else 'n/a'} | {row['loo_rank_order_accuracy'] if row['loo_rank_order_accuracy'] is not None else 'n/a'} |")
    lines.extend([
        "",
        "Model C is the controlled load × frequency model `T = a + b·max_g(L_g fmax/f_g) + c·max_g(L_g(fmax/f_g − 1))`, with loads expressed in millions of decode routed tokens. It is fitted separately to TPOT and rank-max MoE decode service time per routed row.",
        "",
        "## Milestone E offline rescoring",
        "",
        "The eight existing Joint candidates were not rerun. Candidate predictions and 12/13/15 ms boundary classifications are in `results/processed/milestone_g_rescore_e.csv`.",
        "",
        f"Rescore MAE (E/F/G): {summary['candidate_rescore_metrics']['milestone_e']['mae_ms']:.3f} / {summary['candidate_rescore_metrics']['milestone_f']['mae_ms']:.3f} / {summary['candidate_rescore_metrics']['milestone_g']['mae_ms']:.3f} ms. G candidate rank-order accuracy is {summary['candidate_rescore_metrics']['milestone_g']['rank_order_accuracy']:.3f}.",
        "",
        "## Residual and communication diagnostic",
        "",
        summary["communication_diagnosis"],
        "",
        f"## Classification: {summary['classification']}",
        "",
    ])
    if summary["classification"] == "G1_SUPPORTED_MODEL_ADEQUATE":
        lines.append("Controlled load-dependent DVFS sensitivity is measured and the small model is accurate enough for a small offline Joint v2 pilot.")
    elif summary["classification"] == "G1_SUPPORTED_MODEL_INADEQUATE":
        lines.append("Controlled load-dependent DVFS sensitivity is measured, but the remaining error is too large for another Joint search. Add communication/synchronization indicators before optimizing placement again.")
    elif summary["classification"] == "G1_NOT_SUPPORTED":
        lines.append("The controlled experiment did not show a reliable load-dependent DVFS penalty beyond noise; reconsider making placement part of the DVFS objective.")
    else:
        lines.append("The instrumentation or data do not support a reliable decision.")
    lines.extend([
        "",
        "## Limitations",
        "",
        "One model, one workload, one node, three clocks, no replication or migration, no forced-token diagnostic, and no direct all-to-all/synchronization timer. Custom placements can change BF16 reduction order and output identities; token identity is not used as a model feature. The model is diagnostic, not an online scheduler.",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    plan = json.loads((args.plan_dir / "matrix-plan.json").read_text(encoding="utf-8"))
    total, decode = load_profile(args.phase_counts)
    rows = matrix_rows(args.plan_dir)
    initial = [row for row in rows if row["is_initial"]]
    write_csv(args.output_root / "milestone_g_matrix.csv", rows)
    repeats = repeatability_rows(rows)
    write_csv(args.output_root / "milestone_g_repeatability.csv", repeats or [{"status": "missing"}])

    loads = np.asarray([[row[f"decode_load_g{gpu}"] for gpu in range(4)] for row in initial], dtype=float) / 1e6
    frequencies = np.asarray([json.loads(row["frequency_tuple"]) for row in initial], dtype=float)
    tpot = np.asarray([row["tpot_p99_ms"] for row in initial], dtype=float)
    moe = np.asarray([row["moe_decode_ms_per_row"] for row in initial], dtype=float)
    g_tpot, g_tpot_loo = fit_cv(loads, frequencies, tpot)
    g_moe, g_moe_loo = fit_cv(loads, frequencies, moe)
    g_tpot_pred = predict_service_time(g_tpot, loads, frequencies)
    g_moe_pred = predict_service_time(g_moe, loads, frequencies)
    pressure = np.max(loads * FMAX / frequencies, axis=1)
    tpot_pressure_pred, tpot_pressure_metrics = calibrated_pressure_metrics(pressure, tpot)
    moe_pressure_pred, moe_pressure_metrics = calibrated_pressure_metrics(pressure, moe)

    f_summary = json.loads(args.f_summary.read_text(encoding="utf-8"))
    f_model = f_summary["frequency_model"]
    f_tpot_pred = np.asarray([
        f_model["fixed_ms"] + f_model["frequency_ms_at_fmax"] * FMAX / row["frequency_mhz"]
        for row in initial
    ])
    f_tpot_metrics = prediction_metrics(tpot, f_tpot_pred)
    model_table: list[dict[str, Any]] = []
    for name, observed, predicted, loo, pressure_metrics in (
        ("milestone_e_linear_pressure", tpot, tpot_pressure_pred, None, tpot_pressure_metrics),
        ("milestone_f_amdahl", tpot, f_tpot_pred, None, f_tpot_metrics),
        ("milestone_g_load_frequency", tpot, g_tpot_pred, g_tpot_loo, prediction_metrics(tpot, g_tpot_pred)),
        ("milestone_g_load_frequency_moe", moe, g_moe_pred, g_moe_loo, prediction_metrics(moe, g_moe_pred)),
    ):
        metrics = prediction_metrics(observed, predicted)
        model_table.append({"model": name, "target": "tpot_p99_ms" if "moe" not in name else "moe_decode_ms_per_row", **metrics,
                            "loo_mae_ms": loo.get("mae_ms") if loo else None,
                            "loo_mape_percent": loo.get("mape_percent") if loo else None,
                            "loo_max_abs_error_ms": loo.get("max_abs_error_ms") if loo else None,
                            "loo_rank_order_accuracy": loo.get("rank_order_accuracy") if loo else None,
                            "calibrated_pressure_mae_ms": pressure_metrics.get("mae_ms")})
    write_csv(args.output_root / "milestone_g_model.csv", model_table)

    # Rescore the already-measured E candidates using the controlled G model.
    f_candidate_rows = [row for row in csv.DictReader(args.f_candidates.open(newline="", encoding="utf-8")) if row["basis"] == "aggregate"]
    candidate_rows: list[dict[str, Any]] = []
    for row in f_candidate_rows:
        candidate_index = int(row["candidate_index"])
        frequency_tuple = tuple(json.loads(row["frequency_tuple"]))
        placement_path = next(Path(item["placement_file"]) for item in json.loads((args.e_joint_root / "joint-plan.json").read_text())["candidates"] if int(item["candidate_index"]) == candidate_index)
        placement = placement_from_dict(json.loads(placement_path.read_text(encoding="utf-8")))
        decode_load = placement_loads(decode, placement).astype(float)[None, :] / 1e6
        g_pred = float(predict_service_time(g_tpot, decode_load, np.asarray([frequency_tuple], dtype=float))[0])
        actual = float(row["measured_tpot_p99_ms"])
        candidate_rows.append({
            "candidate_index": candidate_index,
            "frequency_tuple": row["frequency_tuple"],
            "actual_tpot_p99_ms": actual,
            "milestone_e_predicted_tpot_ms": float(row["old_linear_predicted_tpot_p99_ms"]),
            "milestone_f_predicted_tpot_ms": float(row["predicted_tpot_p99_ms"]),
            "milestone_g_predicted_tpot_ms": g_pred,
            "g_residual_ms": g_pred - actual,
            "decode_load_g0": float(decode_load[0, 0]),
            "decode_load_g1": float(decode_load[0, 1]),
            "decode_load_g2": float(decode_load[0, 2]),
            "decode_load_g3": float(decode_load[0, 3]),
            "actual_le_12_ms": actual <= 12,
            "actual_le_13_ms": actual <= 13,
            "actual_le_15_ms": actual <= 15,
            "g_pred_le_12_ms": g_pred <= 12,
            "g_pred_le_13_ms": g_pred <= 13,
            "g_pred_le_15_ms": g_pred <= 15,
            "e_pred_le_12_ms": float(row["old_linear_predicted_tpot_p99_ms"]) <= 12,
            "e_pred_le_13_ms": float(row["old_linear_predicted_tpot_p99_ms"]) <= 13,
            "e_pred_le_15_ms": float(row["old_linear_predicted_tpot_p99_ms"]) <= 15,
            "f_pred_le_12_ms": float(row["predicted_tpot_p99_ms"]) <= 12,
            "f_pred_le_13_ms": float(row["predicted_tpot_p99_ms"]) <= 13,
            "f_pred_le_15_ms": float(row["predicted_tpot_p99_ms"]) <= 15,
        })
    write_csv(args.output_root / "milestone_g_rescore_e.csv", candidate_rows)

    penalty_rows: list[dict[str, Any]] = []
    for placement in sorted({row["placement"] for row in initial}):
        values = {row["frequency_mhz"]: row for row in initial if row["placement"] == placement}
        high = values[2520]
        for frequency in (1635, 1005):
            target = values[frequency]
            penalty_rows.append({
                "placement": placement,
                "load_level": high["load_level"],
                "gpu0_decode_load": high["decode_load_g0"],
                "frequency_transition": f"2520->{frequency}",
                "tpot_penalty_ms": target["tpot_p99_ms"] - high["tpot_p99_ms"],
                "moe_decode_penalty_ms": target["moe_decode_ms_per_row"] - high["moe_decode_ms_per_row"],
                "energy_delta_j": target["total_gpu_energy_j"] - high["total_gpu_energy_j"],
            })
    load_values = [float(row["gpu0_decode_load"]) for row in penalty_rows]
    tpot_penalties = [float(row["tpot_penalty_ms"]) for row in penalty_rows]
    moe_penalties = [float(row["moe_decode_penalty_ms"]) for row in penalty_rows]
    penalty_correlations = {
        "tpot": correlation(load_values, tpot_penalties),
        "moe_decode_service": correlation(load_values, moe_penalties),
    }
    for row in penalty_rows:
        row["load_vs_tpot_pearson"] = penalty_correlations["tpot"]["pearson"]
        row["load_vs_moe_pearson"] = penalty_correlations["moe_decode_service"]["pearson"]

    residuals = (tpot - g_tpot_pred).tolist()
    summary = {
        "milestone": "G",
        "plan_dir": str(args.plan_dir),
        "matrix_count": len(initial),
        "all_run_count": len(rows),
        "trace_sha256": plan["trace_sha256"],
        "phase_counts_sha256": sha256(args.phase_counts),
        "controlled_gpu0_decode_loads": {
            row["load_level"]: row["decode_load_g0"] for row in initial if row["frequency_mhz"] == 2520
        },
        "phase_profile": {
            "total_routed_assignments": int(total.sum()),
            "decode_routed_assignments": int(decode.sum()),
            "prefill_routed_assignments": int(total.sum() - decode.sum()),
            "decode_fraction": float(decode.sum() / total.sum()),
        },
        "timing_definition": plan["timing_definition"],
        "penalty_rows": penalty_rows,
        "penalty_correlations": penalty_correlations,
        "model_rows": model_table,
        "g_tpot_coefficients": g_tpot["coefficients"],
        "g_moe_coefficients": g_moe["coefficients"],
        "g_tpot_leave_one_out": g_tpot_loo,
        "g_moe_leave_one_out": g_moe_loo,
        "g_tpot_residual_ms": residuals,
        "communication_indicators_available": False,
        "communication_diagnosis": "No all-to-all or synchronization timer was available without invasive instrumentation; residuals are therefore only a plausibility diagnostic.",
        "candidate_rescore_metrics": {
            "milestone_e": prediction_metrics(
                [float(row["actual_tpot_p99_ms"]) for row in candidate_rows],
                [float(row["milestone_e_predicted_tpot_ms"]) for row in candidate_rows],
            ),
            "milestone_f": prediction_metrics(
                [float(row["actual_tpot_p99_ms"]) for row in candidate_rows],
                [float(row["milestone_f_predicted_tpot_ms"]) for row in candidate_rows],
            ),
            "milestone_g": prediction_metrics(
                [float(row["actual_tpot_p99_ms"]) for row in candidate_rows],
                [float(row["milestone_g_predicted_tpot_ms"]) for row in candidate_rows],
            ),
        },
    }
    load_increases = penalty_correlations["tpot"]["pearson"] > 0.5 and penalty_correlations["moe_decode_service"]["pearson"] > 0.5
    near_slo_mae = float(np.mean([
        abs(float(row["milestone_g_predicted_tpot_ms"]) - float(row["actual_tpot_p99_ms"]))
        for row in candidate_rows if 11.0 <= float(row["actual_tpot_p99_ms"]) <= 15.5
    ]))
    summary["near_slo_mae_ms"] = near_slo_mae
    if not load_increases:
        summary["classification"] = "G1_NOT_SUPPORTED"
    elif near_slo_mae < 0.5 and g_tpot_loo["mae_ms"] < 0.5:
        summary["classification"] = "G1_SUPPORTED_MODEL_ADEQUATE"
    else:
        summary["classification"] = "G1_SUPPORTED_MODEL_INADEQUATE"
    write_csv(args.output_root / "milestone_g_penalties.csv", penalty_rows)
    (args.output_root / "milestone_g_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=True) + "\n", encoding="utf-8"
    )
    model_plot_rows = [
        {"actual_moe_ms": row["moe_decode_ms_per_row"], "predicted_moe_ms": g_moe_pred[index],
         "actual_tpot_ms": row["tpot_p99_ms"], "predicted_tpot_ms": g_tpot_pred[index]}
        for index, row in enumerate(initial)
    ]
    make_plot(args.figure_penalty, rows, model_plot_rows)
    write_doc(
        Path("docs/milestone_g_load_frequency_model.md"),
        summary, rows, model_table, candidate_rows,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan-dir", type=Path, required=True)
    parser.add_argument("--phase-counts", type=Path, required=True)
    parser.add_argument("--f-summary", type=Path, required=True)
    parser.add_argument("--f-candidates", type=Path, required=True)
    parser.add_argument("--e-joint-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("results/processed"))
    parser.add_argument("--figure-penalty", type=Path, default=Path("results/figures/milestone_g_load_vs_dvfs_penalty.pdf"))
    args = parser.parse_args()
    print(json.dumps(analyze(args), indent=2, sort_keys=True, allow_nan=True))


if __name__ == "__main__":
    main()
