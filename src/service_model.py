"""Small empirical models for placement/DVFS diagnosis.

These models deliberately stay linear and inspectable.  They are not intended
to be an online scheduler or a learned performance predictor.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


def _as_vector(values: Sequence[float], name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1 or result.size == 0 or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a non-empty finite vector")
    return result


def predict_amdahl(
    frequencies_mhz: Sequence[float], fixed_ms: float, frequency_ms: float,
    *, fmax_mhz: float,
) -> np.ndarray:
    frequencies = _as_vector(frequencies_mhz, "frequencies_mhz")
    if np.any(frequencies <= 0) or fmax_mhz <= 0:
        raise ValueError("frequencies and fmax_mhz must be positive")
    return fixed_ms + frequency_ms * (fmax_mhz / frequencies)


def fit_amdahl(
    frequencies_mhz: Sequence[float], tpot_ms: Sequence[float],
    *, fmax_mhz: float | None = None,
) -> dict[str, Any]:
    """Fit ``T(f) = fixed + frequency * fmax/f`` by least squares."""
    frequencies = _as_vector(frequencies_mhz, "frequencies_mhz")
    observed = _as_vector(tpot_ms, "tpot_ms")
    if frequencies.size != observed.size or frequencies.size < 2:
        raise ValueError("frequency and TPOT vectors must have >=2 matching entries")
    if np.any(frequencies <= 0):
        raise ValueError("frequencies_mhz must be positive")
    fmax = float(np.max(frequencies) if fmax_mhz is None else fmax_mhz)
    if fmax <= 0:
        raise ValueError("fmax_mhz must be positive")
    design = np.column_stack((np.ones_like(frequencies), fmax / frequencies))
    coefficients, _, rank, _ = np.linalg.lstsq(design, observed, rcond=None)
    if rank < 2:
        raise ValueError("frequency points do not identify the Amdahl model")
    predicted = design @ coefficients
    residual = observed - predicted
    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(np.mean(residual**2)))
    mape = float(np.mean(np.abs(residual) / np.maximum(np.abs(observed), 1e-12)) * 100)
    total_at_max = float(coefficients[0] + coefficients[1])
    fraction = (float(coefficients[1]) / total_at_max
                if total_at_max != 0 else float("nan"))
    return {
        "model": "t_fixed_plus_t_frequency_times_fmax_over_f",
        "fmax_mhz": fmax,
        "fixed_ms": float(coefficients[0]),
        "frequency_ms_at_fmax": float(coefficients[1]),
        "frequency_sensitive_fraction_at_fmax": fraction,
        "frequencies_mhz": frequencies.tolist(),
        "observed_ms": observed.tolist(),
        "predicted_ms": predicted.tolist(),
        "residual_ms": residual.tolist(),
        "mae_ms": mae,
        "rmse_ms": rmse,
        "mape_percent": mape,
    }


def _validate_load_frequency_matrix(
    loads: np.ndarray, frequencies_mhz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    load_matrix = np.asarray(loads, dtype=np.float64)
    freq_matrix = np.asarray(frequencies_mhz, dtype=np.float64)
    if (load_matrix.ndim != 2 or freq_matrix.ndim != 2
            or load_matrix.shape != freq_matrix.shape
            or load_matrix.shape[1] <= 0):
        raise ValueError("loads and frequencies must have matching [configs, GPUs] matrices")
    if (not np.all(np.isfinite(load_matrix)) or np.any(load_matrix < 0)
            or not np.all(np.isfinite(freq_matrix)) or np.any(freq_matrix <= 0)):
        raise ValueError("loads must be non-negative and frequencies positive")
    return load_matrix, freq_matrix


def _system_design(
    loads: np.ndarray, frequencies_mhz: np.ndarray, fmax_mhz: float,
) -> np.ndarray:
    pressure = loads * (fmax_mhz / frequencies_mhz)
    slowdown = loads * (fmax_mhz / frequencies_mhz - 1.0)
    return np.column_stack((
        np.ones(loads.shape[0]),
        np.max(pressure, axis=1),
        np.max(slowdown, axis=1),
    ))


def fit_service_time_model(
    loads: np.ndarray,
    frequencies_mhz: np.ndarray,
    observed_tpot_ms: Sequence[float],
    *,
    fmax_mhz: float,
) -> dict[str, Any]:
    """Fit a system-level max-GPU service proxy.

    The per-GPU proxy is ``a + b*L_g + c*L_g*(fmax/f_g - 1)``.  Since the
    measurements expose system p99 TPOT rather than per-GPU kernel time, the
    fitted observation is the maximum proxy over GPUs, represented by the
    corresponding max-pressure features.  This avoids pretending that p99 TPOT
    is directly attributable to one GPU.
    """
    load_matrix, freq_matrix = _validate_load_frequency_matrix(loads, frequencies_mhz)
    observed = _as_vector(observed_tpot_ms, "observed_tpot_ms")
    if observed.size != load_matrix.shape[0] or fmax_mhz <= 0:
        raise ValueError("observations and configs must have matching lengths")
    design = _system_design(load_matrix, freq_matrix, float(fmax_mhz))
    coefficients, _, rank, _ = np.linalg.lstsq(design, observed, rcond=None)
    if rank < min(design.shape):
        raise ValueError("service-model design is rank deficient")
    predicted = design @ coefficients
    residual = observed - predicted
    return {
        "model": "max_gpu_a_plus_b_pressure_plus_c_frequency_slowdown",
        "fmax_mhz": float(fmax_mhz),
        "a_ms": float(coefficients[0]),
        "b_ms_per_token": float(coefficients[1]),
        "c_ms_per_token": float(coefficients[2]),
        "coefficients": coefficients.tolist(),
        "n_configs": int(load_matrix.shape[0]),
        "predicted_ms": predicted.tolist(),
        "observed_ms": observed.tolist(),
        "residual_ms": residual.tolist(),
        "mae_ms": float(np.mean(np.abs(residual))),
        "rmse_ms": float(np.sqrt(np.mean(residual**2))),
        "mape_percent": float(
            np.mean(np.abs(residual) / np.maximum(np.abs(observed), 1e-12)) * 100
        ),
    }


def predict_service_time(
    model: Mapping[str, Any], loads: np.ndarray, frequencies_mhz: np.ndarray,
) -> np.ndarray:
    """Predict system TPOT from a serialized service-model dictionary."""
    load_matrix, freq_matrix = _validate_load_frequency_matrix(loads, frequencies_mhz)
    fmax = float(model["fmax_mhz"])
    coefficients = np.asarray(model["coefficients"], dtype=np.float64)
    if coefficients.shape != (3,):
        raise ValueError("service model coefficients must contain three values")
    return _system_design(load_matrix, freq_matrix, fmax) @ coefficients


def prediction_metrics(
    observed: Sequence[float], predicted: Sequence[float],
) -> dict[str, float]:
    """Return error and pairwise rank-order diagnostics."""
    actual = _as_vector(observed, "observed")
    estimate = _as_vector(predicted, "predicted")
    if actual.size != estimate.size:
        raise ValueError("observed and predicted lengths must match")
    residual = estimate - actual
    pairs = 0
    concordant = 0
    for i in range(actual.size):
        for j in range(i + 1, actual.size):
            actual_delta = actual[i] - actual[j]
            predicted_delta = estimate[i] - estimate[j]
            if actual_delta == 0 or predicted_delta == 0:
                continue
            pairs += 1
            concordant += int(actual_delta * predicted_delta > 0)
    return {
        "mae_ms": float(np.mean(np.abs(residual))),
        "rmse_ms": float(np.sqrt(np.mean(residual**2))),
        "max_abs_error_ms": float(np.max(np.abs(residual))),
        "mape_percent": float(
            np.mean(np.abs(residual) / np.maximum(np.abs(actual), 1e-12)) * 100
        ),
        "rank_order_accuracy": float(concordant / pairs) if pairs else float("nan"),
        "n": float(actual.size),
    }


def model_to_dict(model: Mapping[str, Any]) -> dict[str, Any]:
    """Make model output JSON serializable without changing its values."""
    result: dict[str, Any] = {}
    for key, value in model.items():
        result[key] = value.tolist() if isinstance(value, np.ndarray) else value
    return result


def fit_load_frequency_model(
    loads: Sequence[float], frequencies_mhz: Sequence[float],
    observed_ms: Sequence[float], *, fmax_mhz: float,
) -> dict[str, Any]:
    """Fit the scalar controlled-load model ``a + bL + cL(fmax/f - 1)``."""
    load_values = _as_vector(loads, "loads")
    frequency_values = _as_vector(frequencies_mhz, "frequencies_mhz")
    observed = _as_vector(observed_ms, "observed_ms")
    if (load_values.size != frequency_values.size
            or load_values.size != observed.size
            or np.any(load_values < 0)
            or np.any(frequency_values <= 0)
            or fmax_mhz <= 0):
        raise ValueError("loads, frequencies, and observations must have matching valid lengths")
    design = np.column_stack((
        np.ones_like(load_values),
        load_values,
        load_values * (float(fmax_mhz) / frequency_values - 1.0),
    ))
    coefficients, _, rank, _ = np.linalg.lstsq(design, observed, rcond=None)
    if rank < min(design.shape):
        raise ValueError("load-frequency design is rank deficient")
    predicted = design @ coefficients
    residual = observed - predicted
    return {
        "model": "a_plus_b_load_plus_c_load_frequency_slowdown",
        "fmax_mhz": float(fmax_mhz),
        "coefficients": coefficients.tolist(),
        "a_ms": float(coefficients[0]),
        "b_ms_per_load": float(coefficients[1]),
        "c_ms_per_load": float(coefficients[2]),
        "predicted_ms": predicted.tolist(),
        "observed_ms": observed.tolist(),
        "residual_ms": residual.tolist(),
        "mae_ms": float(np.mean(np.abs(residual))),
        "rmse_ms": float(np.sqrt(np.mean(residual ** 2))),
        "mape_percent": float(
            np.mean(np.abs(residual) / np.maximum(np.abs(observed), 1e-12)) * 100
        ),
        "n_configs": int(load_values.size),
    }


def predict_load_frequency_model(
    model: Mapping[str, Any], loads: Sequence[float], frequencies_mhz: Sequence[float],
) -> np.ndarray:
    """Predict from a serialized scalar load-frequency model."""
    load_values = _as_vector(loads, "loads")
    frequency_values = _as_vector(frequencies_mhz, "frequencies_mhz")
    if load_values.size != frequency_values.size:
        raise ValueError("loads and frequencies must have matching lengths")
    coefficients = np.asarray(model["coefficients"], dtype=np.float64)
    if coefficients.shape != (3,):
        raise ValueError("load-frequency model coefficients must contain three values")
    fmax = float(model["fmax_mhz"])
    return np.column_stack((
        np.ones_like(load_values),
        load_values,
        load_values * (fmax / frequency_values - 1.0),
    )) @ coefficients
