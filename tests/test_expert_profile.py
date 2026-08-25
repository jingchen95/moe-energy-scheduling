import numpy as np
import pytest

from src.expert_profile import aggregate_rank_loads, load_diagnostics, validate_counts


def test_expert_count_aggregation():
    counts = np.array([[10, 20, 30, 40], [1, 2, 3, 4]], dtype=np.int64)
    mapping = np.array([[0, 0, 1, 1], [0, 1, 0, 1]], dtype=np.int64)
    assert aggregate_rank_loads(counts, mapping).tolist() == [34, 76]


def test_count_validation_and_diagnostics():
    with pytest.raises(ValueError, match="non-negative"):
        validate_counts(np.array([[1, -1]]))
    stats = load_diagnostics(np.array([[1, 3], [2, 2]], dtype=np.int64))
    assert stats["per_layer_mean"] == [2.0, 2.0]
    assert stats["per_layer_max_over_mean"] == [1.5, 1.0]
