import hashlib
import json
from pathlib import Path

import numpy as np

from src.afd_cost_model import (
    fit_binned_curve,
    leave_one_group_out_power,
    placement_features,
    prediction_metrics,
)
from src.afd_routing_profile import combine_rank_routing_traces


ROOT = Path(__file__).resolve().parents[1]


def _write_trace_chunk(path: Path, rank: int, histograms: np.ndarray) -> None:
    records = len(histograms)
    np.savez_compressed(
        path,
        expert_histograms=histograms,
        layers=np.arange(records, dtype=np.int16),
        call_sequences=np.zeros(records, dtype=np.int32),
        token_rows=np.full(records, 2 + rank, dtype=np.int32),
        topk_assignments=histograms.sum(axis=1).astype(np.int32),
        ep_rank=np.asarray(rank),
        generation=np.asarray("g"),
        chunk_index=np.asarray(0),
    )


def test_routing_trace_combines_rank_histograms_by_exact_call(tmp_path):
    hist0 = np.zeros((48, 128), dtype=np.int64)
    hist1 = np.zeros((48, 128), dtype=np.int64)
    hist0[:, 0] = 4
    hist1[:, 1] = 6
    _write_trace_chunk(
        tmp_path / "routing-trace-rank-0-pid-10-chunk-000000-g.npz", 0, hist0
    )
    _write_trace_chunk(
        tmp_path / "routing-trace-rank-1-pid-11-chunk-000000-g.npz", 1, hist1
    )
    combined, metadata = combine_rank_routing_traces(tmp_path, "g")
    assert metadata["records"] == 48
    assert metadata["steps_per_layer"] == 1
    assert np.all(combined["expert_histograms"][:, 0] == 4)
    assert np.all(combined["expert_histograms"][:, 1] == 6)
    assert combined["token_rows_by_source_rank"].shape == (2, 48)


def test_placement_features_reconstruct_rank_local_assignment_counts():
    histograms = np.zeros((48, 128), dtype=np.int64)
    histograms[:, 0] = 7
    histograms[:, 64] = 3
    routing = {"layers": np.arange(48), "expert_histograms": histograms}
    mapping = np.tile(np.repeat([0, 1], 64), (48, 1))
    features = placement_features(routing, mapping)
    assert np.all(features[0]["n"] == 7)
    assert np.all(features[1]["n"] == 3)
    assert np.all(features[0]["active_local_experts"] == 1)
    assert np.all(features[1]["max_tokens_per_local_expert"] == 3)


def test_binned_curve_interpolates_and_reports_error():
    n = np.arange(1, 101)
    elapsed = 2.0 + 0.5 * n
    curve = fit_binned_curve(n, elapsed, bins=10)
    prediction = curve.predict(np.asarray([10.0, 50.0, 90.0]))
    metrics = prediction_metrics(np.asarray([7.0, 27.0, 47.0]), prediction)
    assert metrics["mae"] < 1.0
    assert metrics["p95_absolute_error"] < 1.0


def test_power_validation_holds_out_whole_configuration_groups():
    rows = []
    for config_index, load in enumerate((10.0, 20.0, 30.0, 40.0)):
        for repetition in range(3):
            power = 80.0 + 2.0 * load
            rows.append({
                "config_id": f"c{config_index}",
                "rank": 0,
                "frequency_mhz": 1635,
                "load_rate": load,
                "avg_power_w": power,
                "duration_s": 5.0,
                "window_energy_j": power * 5.0,
                "repetition": repetition,
            })
    predictions = leave_one_group_out_power(rows)
    assert len(predictions) == len(rows)
    assert max(abs(row["predicted_power_w"] - row["avg_power_w"])
               for row in predictions) < 1e-9


def test_patch_manifest_and_one_job_scope_are_reproducible():
    manifest = json.loads(
        (ROOT / "patches/afd-plugin-0.26.0-cost-model.json").read_text()
    )
    patch = ROOT / "patches" / manifest["patch_file"]
    assert hashlib.sha256(patch.read_bytes()).hexdigest() == manifest["patch_sha256"]
    launcher = (ROOT / "scripts/milestone_n_afd_ffn_cost_model.sbatch").read_text()
    assert "#SBATCH --exclusive" in launcher
    assert "--afd-profile-control-dir" in launcher
    assert "--afd-timing-control-dir" in launcher
    assert "manage_afd_cost_model_patch.sh revert" in launcher
    assert "pkill" not in launcher
    assert "killall" not in launcher


def test_config_has_separate_instrumentation_and_no_optimizer():
    text = (ROOT / "configs/milestone_n_afd_ffn_cost_model.yaml").read_text()
    assert "routing_and_timing_are_separate: true" in text
    assert "per_call_cuda_synchronize: false" in text
    assert "optimizer" not in text.lower()
