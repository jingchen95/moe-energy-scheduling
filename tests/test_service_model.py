import numpy as np
import pytest

from src.expert_profile import aggregate_phase_rank_loads, validate_phase_counts
from src.service_model import (
    fit_amdahl,
    fit_service_time_model,
    model_to_dict,
    prediction_metrics,
    predict_service_time,
)


def test_phase_accounting_and_rank_aggregation():
    prefill = np.array([[3, 0, 1, 2]], dtype=np.int64)
    decode = np.array([[1, 2, 0, 1]], dtype=np.int64)
    total = prefill + decode
    validate_phase_counts(prefill, decode, total)
    loads = aggregate_phase_rank_loads(
        prefill, decode, np.array([[0, 0, 1, 1]], dtype=np.int64), total
    )
    assert loads["prefill"].tolist() == [3, 3]
    assert loads["decode"].tolist() == [3, 1]
    assert loads["total"].tolist() == [6, 4]
    with pytest.raises(ValueError, match="conserve"):
        validate_phase_counts(prefill, decode, total + 1)


def test_amdahl_fit_recovers_fixed_and_frequency_components():
    frequencies = [2520, 1635, 1005]
    observed = [5.0, 2.0 + 3.0 * 2520 / 1635, 2.0 + 3.0 * 2520 / 1005]
    model = fit_amdahl(frequencies, observed, fmax_mhz=2520)
    assert model["fixed_ms"] == pytest.approx(2.0, abs=0.1)
    assert model["frequency_ms_at_fmax"] == pytest.approx(3.0, abs=0.1)
    assert model["rmse_ms"] < 0.1


def test_service_model_prediction_and_serialization():
    loads = np.array([[1.0, 2.0], [2.0, 1.0], [1.0, 1.0], [2.0, 2.0]])
    frequencies = np.array([[2520, 2520], [2520, 2520], [1635, 2520], [1635, 2520]])
    observed = [4.0, 4.0, 4.5, 5.0]
    model = fit_service_time_model(loads, frequencies, observed, fmax_mhz=2520)
    predicted = predict_service_time(model, loads, frequencies)
    assert predicted.shape == (4,)
    assert model_to_dict(model)["coefficients"] == pytest.approx(model["coefficients"])
    metrics = prediction_metrics(observed, predicted)
    assert metrics["mae_ms"] < 1.0


def test_service_model_rejects_bad_shapes():
    with pytest.raises(ValueError, match="matching"):
        fit_service_time_model(
            np.ones((2, 4)), np.ones((3, 4)), [1, 2], fmax_mhz=2520
        )
