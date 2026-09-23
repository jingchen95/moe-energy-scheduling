import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "analyze_layerwise_loads.py"
SPEC = importlib.util.spec_from_file_location("analyze_layerwise_loads", SCRIPT)
analysis = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(analysis)


def test_layer_aggregation_uses_per_layer_mapping():
    counts = [[8, 4, 2, 1], [3, 5, 7, 9]]
    mapping = [[0, 0, 1, 1], [1, 0, 1, 0]]
    assert analysis.aggregate_layer_loads(counts, mapping, num_ranks=2) == [
        [12, 3], [14, 10]
    ]


def test_mapping_requires_equal_experts_per_rank_in_every_layer():
    with pytest.raises(ValueError, match="rank cardinalities"):
        analysis.aggregate_layer_loads(
            [[1, 2, 3, 4]], [[0, 0, 0, 1]], num_ranks=2
        )


def test_normalization_and_diagnostics():
    loads = [[10, 10, 10, 10], [20, 10, 5, 5]]
    normalized = analysis.normalize_layer_loads(loads)
    assert normalized[0] == pytest.approx([0.25] * 4)
    assert normalized[1] == pytest.approx([0.5, 0.25, 0.125, 0.125])
    assert all(sum(row) == pytest.approx(1.0) for row in normalized)

    rows = analysis.layer_diagnostics(loads)
    assert rows[0]["max_over_mean"] == pytest.approx(1.0)
    assert rows[0]["coefficient_of_variation"] == pytest.approx(0.0)
    assert rows[1]["most_loaded_gpu"] == 0
    assert rows[1]["least_loaded_gpu"] == 2
    assert rows[1]["most_loaded_tie_count"] == 1
    assert rows[1]["least_loaded_tie_count"] == 2


def test_adjacent_similarity_is_histogram_intersection():
    left = [0.25, 0.25, 0.25, 0.25]
    right = [0.5, 0.25, 0.125, 0.125]
    assert analysis.adjacent_similarity(left, left) == pytest.approx(1.0)
    assert analysis.adjacent_similarity(left, right) == pytest.approx(0.75)


def test_identity_runs_include_layer_boundaries():
    assert analysis.identity_runs([0, 0, 1, 2, 2, 2, 1]) == [
        {"run_id": 0, "gpu": 0, "start_layer": 0, "end_layer": 1, "length": 2},
        {"run_id": 1, "gpu": 1, "start_layer": 2, "end_layer": 2, "length": 1},
        {"run_id": 2, "gpu": 2, "start_layer": 3, "end_layer": 5, "length": 3},
        {"run_id": 3, "gpu": 1, "start_layer": 6, "end_layer": 6, "length": 1},
    ]


def test_summary_reports_switches_and_longest_runs():
    rows = analysis.layer_diagnostics([
        [8, 4, 2, 2], [9, 3, 2, 2], [1, 8, 4, 3], [1, 9, 3, 3]
    ])
    summary = analysis.summarize_diagnostics(rows)
    assert summary["most_loaded_switches"] == 1
    assert summary["most_run_max_length"] == 2
    assert summary["most_run_count"] == 2
