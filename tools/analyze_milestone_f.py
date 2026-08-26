"""Analyze Milestone E data with phase-aware and empirical service models."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.expert_profile import aggregate_rank_loads, validate_phase_counts
from src.placement import placement_from_dict
from src.service_model import (
    fit_amdahl,
    fit_service_time_model,
    prediction_metrics,
    predict_service_time,
)


FMAX_MHZ = 2520.0
BASIS_NAMES = ("aggregate", "prefill", "decode")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metrics_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("metrics.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(value, dict) and value.get("target_clocks_mhz"):
            value["_path"] = str(path)
            rows.append(value)
    return rows


def _frequency_tuple(row: dict[str, Any]) -> tuple[int, ...]:
    return tuple(int(value) for value in row["target_clocks_mhz"])


def _load_phase_profile(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    with np.load(path, allow_pickle=False) as archive:
        total = np.asarray(archive["expert_counts"])
        if "prefill_expert_counts" not in archive or "decode_expert_counts" not in archive:
            zeros = np.zeros_like(total)
            return total.astype(np.int64), total.astype(np.int64), zeros.astype(np.int64), False
        prefill, decode = validate_phase_counts(
            archive["prefill_expert_counts"],
            archive["decode_expert_counts"],
            total,
        )
        return total, prefill, decode, True


def _placement(path: Path):
    return placement_from_dict(json.loads(path.read_text(encoding="utf-8")))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _phase_rows(
    counts_by_basis: dict[str, np.ndarray], placements: dict[str, Any]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for placement_name, placement in placements.items():
        mapping = placement.logical_to_rank
        for basis, counts in counts_by_basis.items():
            per_layer = np.zeros((counts.shape[0], 4), dtype=np.int64)
            for layer in range(counts.shape[0]):
                per_layer[layer] = aggregate_rank_loads(
                    counts[layer : layer + 1], mapping[layer : layer + 1]
                )
            for layer, values in enumerate(per_layer):
                for gpu, value in enumerate(values):
                    rows.append({
                        "placement": placement_name,
                        "phase": basis,
                        "layer": layer,
                        "gpu": gpu,
                        "expert_token_load": int(value),
                    })
            aggregate = per_layer.sum(axis=0)
            for gpu, value in enumerate(aggregate):
                rows.append({
                    "placement": placement_name,
                    "phase": basis,
                    "layer": "aggregate",
                    "gpu": gpu,
                    "expert_token_load": int(value),
                })
    return rows


def _uniform_rows(rows: list[dict[str, Any]], placement: str) -> list[dict[str, Any]]:
    result = [
        row for row in rows
        if row.get("placement_name") == placement
        and len(set(_frequency_tuple(row))) == 1
        and int(row.get("failed_requests", 0) or 0) == 0
        and row.get("tpot_p99_ms") is not None
    ]
    result.sort(key=lambda row: _frequency_tuple(row)[0], reverse=True)
    unique: dict[tuple[int, ...], dict[str, Any]] = {}
    for row in result:
        unique.setdefault(_frequency_tuple(row), row)
    return list(unique.values())


def _make_frequency_rows(uniform: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    frequencies = [float(_frequency_tuple(row)[0]) for row in uniform]
    observed = [float(row["tpot_p99_ms"]) for row in uniform]
    model = fit_amdahl(frequencies, observed, fmax_mhz=FMAX_MHZ)
    high = observed[0]
    output: list[dict[str, Any]] = []
    for row, old_pred, new_pred, residual in zip(
        uniform,
        [high * FMAX_MHZ / f for f in frequencies],
        model["predicted_ms"],
        model["residual_ms"],
    ):
        output.append({
            "record_type": "uniform_p1",
            "frequency_mhz": int(_frequency_tuple(row)[0]),
            "frequency_tuple": json.dumps(_frequency_tuple(row)),
            "tpot_p99_ms": float(row["tpot_p99_ms"]),
            "energy_j": float(row["total_gpu_energy_j"]),
            "old_linear_predicted_tpot_ms": float(old_pred),
            "amdahl_predicted_tpot_ms": float(new_pred),
            "amdahl_residual_ms": float(residual),
            "run_id": row.get("run_id", ""),
        })
    return output, model


def _fit_candidate_models(
    sequential: list[dict[str, Any]],
    p1_loads: dict[str, np.ndarray],
    candidates: list[dict[str, Any]],
    candidate_loads: dict[str, dict[str, np.ndarray]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    frequency_matrix = np.asarray([_frequency_tuple(row) for row in sequential], dtype=float)
    actual_seq = np.asarray([float(row["tpot_p99_ms"]) for row in sequential])
    output: list[dict[str, Any]] = []
    models: dict[str, Any] = {}
    reference_frequency = np.full(4, FMAX_MHZ, dtype=float)
    reference_load = np.asarray(p1_loads["aggregate"], dtype=float)
    reference_pressure = float(np.max(reference_load * FMAX_MHZ / reference_frequency))
    reference_tpot = float(
        next(row["tpot_p99_ms"] for row in sequential
             if _frequency_tuple(row) == (int(FMAX_MHZ),) * 4)
    )
    for basis in BASIS_NAMES:
        base = np.asarray(p1_loads[basis], dtype=float) / 1e6
        train_loads = np.repeat(base[None, :], len(sequential), axis=0)
        model = fit_service_time_model(
            train_loads, frequency_matrix, actual_seq, fmax_mhz=FMAX_MHZ
        )
        models[basis] = model
        train_pred = predict_service_time(model, train_loads, frequency_matrix)
        loo_pred: list[float] = []
        for index in range(len(sequential)):
            keep = np.ones(len(sequential), dtype=bool)
            keep[index] = False
            loo = fit_service_time_model(
                train_loads[keep], frequency_matrix[keep], actual_seq[keep],
                fmax_mhz=FMAX_MHZ,
            )
            loo_pred.append(float(predict_service_time(
                loo, train_loads[index:index + 1], frequency_matrix[index:index + 1]
            )[0]))
        models[basis]["training_metrics"] = prediction_metrics(actual_seq, train_pred)
        models[basis]["leave_one_out_metrics"] = prediction_metrics(actual_seq, loo_pred)
        for candidate in candidates:
            key = str(candidate["candidate_index"])
            loads = np.asarray(candidate_loads[key][basis], dtype=float)[None, :] / 1e6
            frequencies = np.asarray([candidate["frequency_tuple"]], dtype=float)
            predicted = float(predict_service_time(model, loads, frequencies)[0])
            actual = float(candidate["metrics"]["tpot_p99_ms"])
            raw = np.asarray(candidate_loads[key]["aggregate"], dtype=float)
            pressure = raw * FMAX_MHZ / frequencies[0]
            old_prediction = reference_tpot * float(np.max(pressure)) / reference_pressure
            output.append({
                "candidate_index": int(candidate["candidate_index"]),
                "label": candidate["label"],
                "frequency_tuple": json.dumps(candidate["frequency_tuple"]),
                "basis": basis,
                "placement": candidate.get("placement_name", "frequency_aware"),
                "aggregate_load_g0": float(raw[0]),
                "aggregate_load_g1": float(raw[1]),
                "aggregate_load_g2": float(raw[2]),
                "aggregate_load_g3": float(raw[3]),
                "old_max_normalized_pressure": float(np.max(pressure) / np.mean(pressure)),
                "old_linear_predicted_tpot_p99_ms": old_prediction,
                "predicted_tpot_p99_ms": predicted,
                "measured_tpot_p99_ms": actual,
                "prediction_residual_ms": predicted - actual,
                "energy_j": float(candidate["metrics"]["total_gpu_energy_j"]),
                "slo_tpot_13_ms": bool(actual <= 13.0),
            })
    return output, models


def _load_candidates(joint_root: Path) -> list[dict[str, Any]]:
    plan = json.loads((joint_root / "joint-plan.json").read_text(encoding="utf-8"))
    candidates: list[dict[str, Any]] = []
    by_index: dict[int, dict[str, Any]] = {}
    for path in sorted((joint_root / "measured").glob("candidate-*/result.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        by_index[int(value["candidate_index"])] = value
    metric_rows = _metrics_rows(joint_root)
    for row in plan["candidates"]:
        index = int(row["candidate_index"])
        result = by_index.get(index, {})
        metrics = next(
            (item for item in metric_rows if str(item.get("_path", "")).find(
                f"candidate-{index:02d}-"
            ) >= 0),
            None,
        )
        if metrics is None:
            raise ValueError(f"missing metrics for Joint candidate {index}")
        candidates.append({
            "candidate_index": index,
            "label": row["label"],
            "frequency_tuple": tuple(int(v) for v in row["frequency_tuple"]),
            "placement_file": Path(row["placement_file"]),
            "placement_name": metrics.get("placement_name", "frequency_aware"),
            "metrics": metrics,
            "result": result,
        })
    return candidates


def _write_doc(
    path: Path,
    *,
    phase_available: bool,
    phase_hash: str,
    placements: dict[str, Any],
    phase_summary: dict[str, dict[str, list[int]]],
    frequency_model: dict[str, Any],
    service_models: dict[str, Any],
    classification: str,
) -> None:
    lines = [
        "# Milestone F — Decode-Aware Empirical Service Model",
        "",
        "## Research question",
        "",
        "Why did the load/frequency proxy used in Milestone E fail to predict the measured Joint frontier, and what is the smallest empirical replacement?",
        "",
        "## Boundary",
        "",
        "This diagnosis does not implement an online scheduler, a Joint optimizer, communication model, or new policy search. It reuses the accepted Milestone D/E measurements.",
        "",
        "## Phase accounting",
        "",
        f"Phase profile available: **{phase_available}**. Profile SHA-256: `{phase_hash}`.",
        "",
        "The vLLM patch keeps the existing EPLB routed-token counter and adds opt-in prefill/decode windows. It uses the scheduler's attention metadata and DP-rank token-block ordering; the dump rejects any per-expert conservation mismatch.",
        "",
        "## Placement loads",
        "",
        "| placement | basis | GPU0 | GPU1 | GPU2 | GPU3 | max/mean |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for placement_name, basis_values in phase_summary.items():
        for basis, values in basis_values.items():
            avg = sum(values) / len(values) if values else 0.0
            ratio = max(values) / avg if avg else float("nan")
            lines.append(
                f"| {placement_name} | {basis} | {values[0]} | {values[1]} | {values[2]} | {values[3]} | {ratio:.3f} |"
            )
    lines.extend([
        "",
        "## Frequency scaling",
        "",
        "The old proxy assumes `TPOT(f) = TPOT(high) * fmax/f`. The replacement fits `T(f) = a + b*fmax/f` to the three measured uniform P1 points. The fitted fixed component, frequency-sensitive component, residuals, and leave-one-configuration-out errors are recorded in `results/processed/milestone_f_frequency_model.csv` and `milestone_f_summary.json`.",
        "",
        f"Amdahl fit: fixed = {frequency_model['fixed_ms']:.4f} ms, frequency-sensitive at max = {frequency_model['frequency_ms_at_fmax']:.4f} ms, sensitive fraction = {frequency_model['frequency_sensitive_fraction_at_fmax']:.3f}, MAE = {frequency_model['mae_ms']:.4f} ms.",
        "",
        "## Service-model diagnosis",
        "",
        "For each candidate, the model uses a max-GPU proxy `a + b*L_g + c*L_g*(fmax/f_g - 1)`. Fits are trained on the accepted 81-row Sequential P1 sweep and evaluated on the eight existing Joint candidates; no Joint measurement is changed or rerun.",
        "",
        f"Diagnostic classification: **{classification}**.",
        "",
        "The old linear pressure score failed because it conflated prefill and decode work, assumed frequency-proportional service time, and treated token load as sufficient to determine the synchronized system tail. The empirical model separates phase loads where available and estimates the frequency-sensitive term from measured service behavior; remaining residual is not attributed to experts without communication/synchronization instrumentation.",
        "",
        "## Limitations",
        "",
        "- One model, one L40S node, one GSM8K trace, and no expert replication or migration.",
        "- The service model is deliberately small and should not be interpreted as a production predictor.",
        "- Eight Joint candidates are a diagnostic sample, not a new search or a proof of a global optimum.",
        "- Any phase-profile mismatch or unsupported execution branch is reported rather than silently imputed.",
        "",
        "The next milestone remains a research decision based on this diagnosis; this milestone does not begin the Joint scheduler.",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    sequential_rows = _metrics_rows(Path(args.sequential_root))
    uniform = _uniform_rows(sequential_rows, "upstream_default_eplb")
    if len(uniform) < 3:
        raise ValueError("need three uniform P1 frequency rows")
    frequency_rows, frequency_model = _make_frequency_rows(uniform)

    phase_path = Path(args.phase_run_dir) / "expert_counts.npz"
    total_counts, prefill_counts, decode_counts, phase_available = _load_phase_profile(phase_path)
    phase_hash = _sha256(phase_path)
    p1_path = Path(args.d_controller) / "placements/upstream_default_eplb.json"
    placements = {
        "P0_default_linear": _placement(Path(args.d_controller) / "placements/default_linear.json"),
        "P1_upstream_eplb": _placement(p1_path),
        "P2_diagnostic_load_concentrated": _placement(
            Path(args.d_controller) / "placements/diagnostic_load_concentrated.json"
        ),
    }
    counts_by_basis = {
        "aggregate": total_counts,
        "prefill": prefill_counts,
        "decode": decode_counts,
    }
    phase_rows = _phase_rows(counts_by_basis, placements)
    output_root = Path(args.output_root)
    _write_csv(
        output_root / "milestone_f_phase_loads.csv",
        phase_rows,
        ["placement", "phase", "layer", "gpu", "expert_token_load"],
    )

    p1_loads = {
        basis: aggregate_rank_loads(counts, placements["P1_upstream_eplb"].logical_to_rank)
        for basis, counts in counts_by_basis.items()
    }
    candidates = _load_candidates(Path(args.joint_root))
    candidate_loads: dict[str, dict[str, np.ndarray]] = {}
    for candidate in candidates:
        placement = _placement(candidate["placement_file"])
        candidate_loads[str(candidate["candidate_index"])] = {
            basis: aggregate_rank_loads(counts, placement.logical_to_rank)
            for basis, counts in counts_by_basis.items()
        }
    candidate_rows, service_models = _fit_candidate_models(
        sequential_rows, p1_loads, candidates, candidate_loads
    )
    _write_csv(
        output_root / "milestone_f_frequency_model.csv",
        frequency_rows,
        ["record_type", "frequency_mhz", "frequency_tuple", "tpot_p99_ms", "energy_j",
         "old_linear_predicted_tpot_ms", "amdahl_predicted_tpot_ms", "amdahl_residual_ms", "run_id"],
    )
    _write_csv(
        output_root / "milestone_f_candidate_predictions.csv",
        candidate_rows,
        ["candidate_index", "label", "frequency_tuple", "basis", "placement",
         "aggregate_load_g0", "aggregate_load_g1", "aggregate_load_g2", "aggregate_load_g3",
         "old_max_normalized_pressure", "old_linear_predicted_tpot_p99_ms",
         "predicted_tpot_p99_ms", "measured_tpot_p99_ms",
         "prediction_residual_ms", "energy_j", "slo_tpot_13_ms"],
    )

    phase_summary: dict[str, dict[str, list[int]]] = {}
    for placement_name, placement in placements.items():
        phase_summary[placement_name] = {}
        for basis, counts in counts_by_basis.items():
            phase_summary[placement_name][basis] = aggregate_rank_loads(
                counts, placement.logical_to_rank
            ).astype(int).tolist()
    aggregate_predictions = [row for row in candidate_rows if row["basis"] == "aggregate"]
    old_accuracy = prediction_metrics(
        [row["measured_tpot_p99_ms"] for row in aggregate_predictions],
        [row["old_linear_predicted_tpot_p99_ms"] for row in aggregate_predictions],
    )
    new_accuracy = prediction_metrics(
        [row["measured_tpot_p99_ms"] for row in aggregate_predictions],
        [row["predicted_tpot_p99_ms"] for row in aggregate_predictions],
    )
    if not phase_available:
        classification = "E_inconclusive_phase_profile_unavailable"
    elif new_accuracy["rank_order_accuracy"] > old_accuracy["rank_order_accuracy"] and new_accuracy["mae_ms"] < old_accuracy["mae_ms"]:
        classification = "B_frequency_or_service_model_dominant"
    elif new_accuracy["mae_ms"] >= old_accuracy["mae_ms"] and new_accuracy["rank_order_accuracy"] <= old_accuracy["rank_order_accuracy"]:
        classification = "D_load_only_insufficient_residual_communication_or_sync_candidate"
    else:
        classification = "E_inconclusive"

    summary = {
        "milestone": "F",
        "phase_profile_available": phase_available,
        "phase_profile_path": str(phase_path),
        "phase_profile_sha256": phase_hash,
        "phase_accounting": {
            "total_sum": int(total_counts.sum()),
            "prefill_sum": int(prefill_counts.sum()),
            "decode_sum": int(decode_counts.sum()),
            "conserves_total": bool(np.array_equal(total_counts, prefill_counts + decode_counts)),
        },
        "frequency_model": frequency_model,
        "frequency_old_proxy": {
            "description": "uniform P1 TPOT(high) * fmax/f",
            "rows": frequency_rows,
        },
        "service_models": service_models,
        "candidate_count": len(candidates),
        "candidate_aggregate_prediction_metrics": {
            "old_linear_pressure": old_accuracy,
            "empirical_service_model": new_accuracy,
        },
        "classification": classification,
        "inputs": {
            "sequential_root": str(args.sequential_root),
            "joint_root": str(args.joint_root),
            "d_controller": str(args.d_controller),
            "phase_run_dir": str(args.phase_run_dir),
        },
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "milestone_f_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    _write_doc(
        Path(args.doc),
        phase_available=phase_available,
        phase_hash=phase_hash,
        placements=placements,
        phase_summary=phase_summary,
        frequency_model=frequency_model,
        service_models=service_models,
        classification=classification,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--d-controller", required=True, type=Path)
    parser.add_argument("--sequential-root", required=True, type=Path)
    parser.add_argument("--joint-root", required=True, type=Path)
    parser.add_argument("--phase-run-dir", required=True, type=Path)
    parser.add_argument("--output-root", default=Path("results/processed"), type=Path)
    parser.add_argument("--doc", default=Path("docs/milestone_f_service_model.md"), type=Path)
    args = parser.parse_args()
    print(json.dumps(analyze(args), indent=2, sort_keys=True, allow_nan=True))


if __name__ == "__main__":
    main()
