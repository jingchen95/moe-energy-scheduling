"""CPU-only validation statistics; no AFD runtime or scheduling dependencies."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np


def error_statistics(actual, predicted) -> dict:
    y, p = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    if y.shape != p.shape or not y.size or not np.all(np.isfinite([y, p])):
        raise ValueError("matching nonempty finite observations required")
    if np.any(y <= 0):
        raise ValueError("positive observations required for relative errors")
    error = p - y
    absolute = np.abs(error)
    variance = np.sum((y - y.mean()) ** 2)
    return {
        "samples": len(y), "mae": float(absolute.mean()),
        "mape_percent": float(np.mean(absolute / y) * 100),
        "median_absolute_error": float(np.median(absolute)),
        "p90_absolute_error": float(np.percentile(absolute, 90)),
        "p95_absolute_error": float(np.percentile(absolute, 95)),
        "p99_absolute_error": float(np.percentile(absolute, 99)),
        "bias": float(error.mean()),
        "r2": float(1 - np.sum(error ** 2) / variance) if variance > 1e-15 else None,
    }


def coverage(training, heldout, centers=None) -> dict:
    train, test = np.asarray(training), np.asarray(heldout)
    if not train.size or not test.size:
        raise ValueError("coverage requires training and held-out samples")
    result = {
        "train_min": float(train.min()), "train_max": float(train.max()),
        "test_min": float(test.min()), "test_max": float(test.max()),
        "extrapolation_fraction": float(np.mean((test < train.min()) | (test > train.max()))),
    }
    if centers is not None:
        values = np.asarray(centers)
        result["endpoint_clamp_fraction"] = float(np.mean((test < values.min()) | (test > values.max())))
    return result


@dataclass
class ResidualRidge:
    mean: np.ndarray
    scale: np.ndarray
    intercept: float
    coefficients: np.ndarray
    alpha: float

    @classmethod
    def fit(cls, x, residual, alpha=10.0):
        x, y = np.asarray(x, dtype=float), np.asarray(residual, dtype=float)
        if x.ndim != 2 or y.shape != (len(x),) or not len(x) or alpha <= 0:
            raise ValueError("invalid Ridge training data")
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
            raise ValueError("non-finite Ridge training data")
        mean, scale = x.mean(axis=0), x.std(axis=0)
        scale = np.where(scale > 1e-12, scale, 1.0)
        z = (x - mean) / scale
        intercept = float(y.mean())
        coefficients = np.linalg.solve(z.T @ z + alpha * np.eye(x.shape[1]), z.T @ (y - intercept))
        return cls(mean, scale, intercept, coefficients, alpha)

    def predict(self, x):
        return self.intercept + (np.asarray(x) - self.mean) / self.scale @ self.coefficients

    def to_dict(self):
        return {"mean": self.mean.tolist(), "scale": self.scale.tolist(),
                "intercept": self.intercept, "coefficients": self.coefficients.tolist(), "alpha": self.alpha}


def average_ranks(values):
    values = np.asarray(values)
    order = np.argsort(values, kind="stable")
    result = np.empty(len(values), dtype=float)
    for value in np.unique(values):
        positions = np.flatnonzero(values[order] == value)
        result[order[positions]] = positions.mean() + 1
    return result


def ranking_statistics(names, measured, predicted, deviations=None) -> tuple[dict, list]:
    """Lower is better. One pooled sample SD flags noise, not significance."""
    y, p = np.asarray(measured), np.asarray(predicted)
    if len(names) != len(y) or y.shape != p.shape or len(y) < 2:
        raise ValueError("ranking requires at least two matching candidates")
    if not np.all(np.isfinite([y, p])):
        raise ValueError("non-finite candidate score")
    std = None if deviations is None else np.asarray(deviations)
    pairs = []
    concordant = discordant = tied_y = tied_p = 0
    for i, j in combinations(range(len(y)), 2):
        dy, dp = y[i] - y[j], p[i] - p[j]
        within_noise = bool(abs(dy) <= np.hypot(std[i], std[j])) if std is not None else None
        inversion = bool(dy * dp < 0)
        concordant += dy * dp > 0
        discordant += inversion
        tied_y += dy == 0 and dp != 0
        tied_p += dp == 0 and dy != 0
        pairs.append({"candidate_a": names[i], "candidate_b": names[j],
                      "measured_difference": float(dy), "predicted_difference": float(dp),
                      "inversion": inversion, "within_pooled_sample_sd": within_noise,
                      "correct": bool(dy * dp > 0) if dy != 0 and dp != 0 else None})
    denom = np.sqrt((concordant + discordant + tied_y) * (concordant + discordant + tied_p))
    yr, pr = average_ranks(y), average_ranks(p)
    resolved = [row for row in pairs if row["within_pooled_sample_sd"] is False and row["correct"] is not None]
    comparable = [row for row in pairs if row["correct"] is not None]
    best_y, best_p = int(np.argmin(y)), int(np.argmin(p))
    k = min(3, len(y))
    return {
        "candidates": len(y), "pairs": len(pairs),
        "pairwise_accuracy": float(np.mean([row["correct"] for row in comparable])) if comparable else None,
        "resolved_pairs": len(resolved),
        "resolved_pairwise_accuracy": float(np.mean([row["correct"] for row in resolved])) if resolved else None,
        "spearman": float(np.corrcoef(yr, pr)[0, 1]) if np.std(yr) > 0 and np.std(pr) > 0 else None,
        "kendall_tau_b": float((concordant - discordant) / denom) if denom > 0 else None,
        "inversions": int(discordant), "top1_measured": names[best_y], "top1_predicted": names[best_p],
        "top1_correct": best_y == best_p,
        "top1_difference_within_noise": bool(abs(y[best_y] - y[best_p]) <= np.hypot(std[best_y], std[best_p])) if std is not None else None,
        "top_k": k, "top_k_overlap": len(set(np.argsort(y)[:k]) & set(np.argsort(p)[:k])) / k,
    }, pairs


def select_feasible(names, latency, energy, budget):
    feasible = [i for i, value in enumerate(latency) if value <= budget]
    return names[min(feasible, key=lambda i: energy[i])] if feasible else None


def oracle_decision_gate(rankings, scenarios, *, selected_model, bridge_validated):
    """Require deployment-relevant evidence, not conditional-energy accuracy."""
    reasons = []
    latency = next(row for row in rankings if row["metric"] == f"latency_{selected_model}")
    if latency["resolved_pairwise_accuracy"] is None:
        reasons.append("No latency pairs resolved beyond observed variability.")
    elif latency["resolved_pairwise_accuracy"] < 0.9:
        reasons.append("Less than 90% accuracy among latency pairs resolved beyond pooled sample SD.")
    infeasible = sum(not row["selected_measured_tpot_feasible"] for row in scenarios)
    if infeasible:
        reasons.append(f"Diagnostic selector chooses a measured mean-TPOT-infeasible candidate in {infeasible}/{len(scenarios)} scenarios.")
    if not bridge_validated:
        reasons.append("No independently validated mapping from instrumented local critical score to uninstrumented p99/duration; no TTFT predictor.")
    return {"decision": "FAIL" if reasons else "PASS", "proceed_to_joint_oracle": not reasons,
            "reasons": reasons, "latency_ranking_minimum_accuracy": 0.9,
            "threshold_is_readiness_screen_not_significance": True}
