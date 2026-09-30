import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from src.afd_profile_analysis import (
    HIGH_MHZ,
    MEDIUM_MHZ,
    build_two_rank_candidates,
    layer_rank_loads,
    placement_diagnostics,
    profile_diagnostics,
)
from src.afd_routing_profile import (
    RuntimeRoutingProfiler,
    combine_rank_profiles,
    write_profile_control,
)
from src.placement import save_placement


REPO_ROOT = Path(__file__).resolve().parents[1]


def _synthetic_counts() -> np.ndarray:
    base = np.arange(1, 129, dtype=np.int64)
    return np.stack([np.roll(base, layer * 7) for layer in range(48)])


def test_runtime_profile_control_excludes_inactive_flush(tmp_path):
    profiler = RuntimeRoutingProfiler(tmp_path, ep_rank=0)
    write_profile_control(tmp_path, "measured", True)
    profiler.capture(0, torch.tensor([[1, 2], [2, 3]], dtype=torch.int32))
    profiler.capture(1, torch.tensor([[4, 4]], dtype=torch.int32))
    write_profile_control(tmp_path, "measured", False)
    profiler.capture(0, torch.tensor([[9, 9]], dtype=torch.int32))
    paths = list(tmp_path.glob("routing-counts-rank-0-*.npz"))
    assert len(paths) == 1
    with np.load(paths[0], allow_pickle=False) as archive:
        counts = archive["expert_counts"]
    assert counts.shape == (48, 128)
    assert counts[0, 1] == 1
    assert counts[0, 2] == 2
    assert counts[0, 3] == 1
    assert counts[0, 9] == 0
    assert counts[1, 4] == 2


def test_rank_profiles_sum_without_loss(tmp_path):
    write_profile_control(tmp_path, "g", True)
    profilers = [RuntimeRoutingProfiler(tmp_path, rank) for rank in range(2)]
    profilers[0].capture(0, torch.tensor([[1, 2]], dtype=torch.int32))
    profilers[1].capture(0, torch.tensor([[2, 3]], dtype=torch.int32))
    write_profile_control(tmp_path, "g", False)
    for profiler in profilers:
        profiler.capture(0, torch.tensor([[7, 7]], dtype=torch.int32))
    paths = sorted(tmp_path.glob("routing-counts-rank-*.npz"))
    combined, per_rank, metadata = combine_rank_profiles(paths, "g")
    assert combined[0, [1, 2, 3]].tolist() == [1, 2, 1]
    assert np.array_equal(combined, per_rank.sum(axis=0))
    assert metadata["aggregation"].startswith("sum of disjoint")


def test_two_rank_candidates_are_valid_deterministic_and_mirrored(tmp_path):
    counts = _synthetic_counts()
    first = build_two_rank_candidates(counts)
    second = build_two_rank_candidates(counts)
    assert list(first) == ["P_balanced", "P_MH", "P_HM"]
    for name, placement in first.items():
        placement.validate()
        assert placement.num_layers == 48
        assert placement.num_experts == 128
        assert placement.experts_per_rank == 64
        assert np.array_equal(
            placement.logical_to_rank, second[name].logical_to_rank
        )
        path = tmp_path / f"{name}.json"
        save_placement(path, placement)
        assert len(hashlib.sha256(path.read_bytes()).hexdigest()) == 64
    assert np.array_equal(
        first["P_HM"].logical_to_rank,
        1 - first["P_MH"].logical_to_rank,
    )
    mh = layer_rank_loads(counts, first["P_MH"].logical_to_rank)
    hm = layer_rank_loads(counts, first["P_HM"].logical_to_rank)
    assert np.all(mh[:, 0] <= mh[:, 1])
    assert np.all(hm[:, 1] <= hm[:, 0])


def test_profile_diagnostics_quantify_expert_to_rank_smoothing():
    counts = _synthetic_counts()
    mapping = np.tile(np.repeat(np.arange(2), 64), (48, 1))
    result = profile_diagnostics(counts, mapping)
    assert result["logical_expert"]["aggregate_max_over_mean"] > 1
    assert sum(result["default_rank"]["aggregate_loads"]) == int(counts.sum())
    assert min(result["default_rank"]["aggregate_loads"]) > 0
    assert result["smoothing"]["mean_layer_cv_reduction_fraction"] > 0


def test_frequency_aware_candidate_improves_mh_pressure_proxy():
    counts = _synthetic_counts()
    candidates = build_two_rank_candidates(counts)
    balanced = placement_diagnostics(
        counts, candidates["P_balanced"], (MEDIUM_MHZ, HIGH_MHZ)
    )
    aligned = placement_diagnostics(
        counts, candidates["P_MH"], (MEDIUM_MHZ, HIGH_MHZ)
    )
    assert (
        aligned["aggregate_normalized_max_over_mean"]
        < balanced["aggregate_normalized_max_over_mean"]
    )


def test_profile_patch_manifest_and_launcher_scope():
    manifest = json.loads(
        (REPO_ROOT / "patches" / "afd-plugin-0.26.0-static-profile.json").read_text()
    )
    patch = REPO_ROOT / "patches" / manifest["patch_file"]
    assert hashlib.sha256(patch.read_bytes()).hexdigest() == manifest["patch_sha256"]
    launcher = (REPO_ROOT / "scripts" / "milestone_m_afd_profile.sbatch").read_text()
    assert "--afd-profile-control-dir" in launcher
    assert "--request-rate 16" in launcher
    assert "--warmup-requests 8" in launcher
    assert "18-run" not in launcher
