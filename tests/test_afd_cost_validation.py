import numpy as np
import pytest

from src.afd_cost_validation import (
    ResidualRidge, average_ranks, coverage, error_statistics,
    ranking_statistics, select_feasible, oracle_decision_gate,
)


def test_metrics_signed_bias_and_percentiles():
    value = error_statistics([1, 2, 3], [2, 2, 2])
    assert value["bias"] == 0
    assert value["mae"] == pytest.approx(2 / 3)
    assert value["median_absolute_error"] == 1
    assert error_statistics([2, 2], [2, 2])["r2"] is None


def test_metrics_reject_invalid_or_zero_ground_truth():
    for y, p in [([], []), ([0], [1]), ([1], [np.nan]), ([1], [1, 2])]:
        with pytest.raises(ValueError):
            error_statistics(y, p)


def test_coverage_distinguishes_endpoint_clamping_from_extrapolation():
    value = coverage([0, 1, 2, 3], [0, 1, 4], centers=[1, 2])
    assert value["extrapolation_fraction"] == pytest.approx(1 / 3)
    assert value["endpoint_clamp_fraction"] == pytest.approx(2 / 3)


def test_residual_ridge_train_statistics_and_constant_feature():
    x = np.column_stack([np.arange(20), np.ones(20)])
    model = ResidualRidge.fit(x, 3 + np.arange(20) * 2, alpha=0.01)
    assert np.max(np.abs(model.predict(x) - (3 + np.arange(20) * 2))) < 0.02
    restored = model.to_dict()
    assert restored["scale"][1] == 1
    assert restored["alpha"] == 0.01


def test_ranking_inversions_and_noise_are_separate():
    value, pairs = ranking_statistics(["a", "b", "c"], [1, 2, 3], [2, 1, 3], [1, 1, 0])
    assert value["inversions"] == 1
    assert value["pairwise_accuracy"] == pytest.approx(2 / 3)
    assert value["resolved_pairwise_accuracy"] == 1
    assert not value["top1_correct"] and value["top1_difference_within_noise"]
    assert pairs[0]["within_pooled_sample_sd"]


def test_ranking_ties_and_selection_missing_feasible():
    assert average_ranks([4, 2, 2]).tolist() == [3, 1.5, 1.5]
    value, _ = ranking_statistics(["a", "b"], [1, 1], [1, 2])
    assert value["spearman"] is None and value["kendall_tau_b"] is None
    assert value["resolved_pairs"] == 0 and value["top1_difference_within_noise"] is None
    assert select_feasible(["a", "b"], [2, 3], [5, 1], 1) is None
    assert select_feasible(["a", "b"], [2, 3], [5, 1], 3) == "b"


def test_oracle_gate_rejects_conditional_energy_or_infeasible_selection():
    rankings = [{"metric": "latency_n_only", "resolved_pairwise_accuracy": 1.0}]
    scenarios = [{"selected_measured_tpot_feasible": True}]
    assert oracle_decision_gate(rankings, scenarios, selected_model="n_only", bridge_validated=True)["decision"] == "PASS"
    assert oracle_decision_gate(rankings, scenarios, selected_model="n_only", bridge_validated=False)["decision"] == "FAIL"
    scenarios[0]["selected_measured_tpot_feasible"] = False
    assert oracle_decision_gate(rankings, scenarios, selected_model="n_only", bridge_validated=True)["decision"] == "FAIL"


def test_histogram_features_use_post_placement_local_experts():
    from tools.validate_milestone_n_models import derive_features, ridge_x
    profile = {
        "layers": np.array([0]), "runtime_step_ids": np.array([2]), "stages": np.array([0]),
        "expert_histograms": np.ones((1, 128), dtype=int),
        "token_rows_by_source_rank": np.array([[8], [8]]),
        "run_modes_by_rank": np.array([["replay"], ["replay"]]),
        "local_elapsed_ms_by_rank": np.array([[1.0], [1.0]]),
        "full_elapsed_ms_by_rank": np.array([[2.0], [2.0]]),
    }
    mapping = np.tile(np.repeat([0, 1], 64), (48, 1))
    frame = derive_features(profile, mapping, 0)
    assert frame.n.iloc[0] == 64
    assert frame.active_local_experts.iloc[0] == 64
    assert frame.expert_load_cv.iloc[0] == 0
    assert frame.entropy.iloc[0] == pytest.approx(np.log(64))
    assert ridge_x(frame).shape == (1, 57)


def test_complete_offline_plot_pipeline_without_gpus(tmp_path):
    import pandas as pd
    from tools.validate_milestone_n_models import MODELS, plots

    output = tmp_path / "tables"
    output.mkdir()
    pd.DataFrame([{"model": m, "evaluation": e, "mape_mean": 15.0}
                  for m in MODELS for e in ("heldout_repeat", "heldout_skew")]).to_csv(output / "latency_model_comparison.csv", index=False)
    pd.DataFrame([{"feature": f, "pearson_residual": .2} for f in
                  ("entropy", "active_local_experts", "expert_load_cv")]).to_csv(output / "residual_feature_analysis.csv", index=False)
    pd.DataFrame([{"rank": r, "frequency_mhz": f, "load_rate": 300000,
                   "nvml_average_power_w": 150} for r in (0, 1) for f in (1635, 2070)]).to_csv(output / "power_dataset.csv", index=False)
    frame = pd.DataFrame([{
        "rank": rank, "frequency_mhz": frequency, "replay": mode, "lifecycle": lifecycle,
        "n": 512, "local_ms": .8, "pred_n_only": .9, "pred_ridge_residual": .85,
        "entropy": 3.0, "active_local_experts": 64, "expert_load_cv": .5,
    } for rank in (0, 1) for frequency in (1005, 1635, 2070, 2220, 2520)
      for mode in (0, 1) for lifecycle in ("primary", "heldout_repeat")])
    candidates = pd.DataFrame({
        "measured_tpot_mean_ms": [60, 65], "measured_tpot_std_ms": [1, 2],
        "score_n_only_ms": [25000, 28000], "measured_energy_mean_j": [20000, 21000],
        "measured_energy_std_j": [100, 200], "proxy_energy_j": [20200, 21200],
    })
    figures = tmp_path / "figures"
    plots(frame, candidates, output, figures)
    assert len(list(figures.glob("*.png"))) == 5
    assert all(path.stat().st_size > 1000 for path in figures.glob("*.png"))
    with pytest.raises(FileExistsError):
        plots(frame, candidates, output, figures)
