import hashlib
import json
from pathlib import Path

import pytest

from src.afd_static_placement import (
    EXPECTED_EXPERTS,
    EXPECTED_LAYERS,
    EXPECTED_RANKS,
    balanced_round_robin_mapping,
    combine_runtime_evidence,
    different_block_mapping,
    global_to_local_map,
    linear_default_mapping,
    placement_document,
    validate_static_mapping,
    write_placement,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "mapping_factory",
    [linear_default_mapping, balanced_round_robin_mapping, different_block_mapping],
)
def test_two_rank_static_placements_preserve_exact_cardinality(mapping_factory):
    mapping = mapping_factory()
    assert len(mapping) == EXPECTED_LAYERS
    for row in mapping:
        assert len(row) == EXPECTED_EXPERTS
        assert [row.count(rank) for rank in range(EXPECTED_RANKS)] == [64, 64]


def test_custom_mappings_are_deterministic_and_materially_different():
    balanced_first = balanced_round_robin_mapping()
    balanced_second = balanced_round_robin_mapping()
    different = different_block_mapping()
    default = linear_default_mapping()
    assert balanced_first == balanced_second
    assert balanced_first != default
    assert different != default
    assert different != balanced_first


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda value: value[:-1], "expected 48 layers"),
        (lambda value: [row[:-1] for row in value], "expected 128 experts"),
        (lambda value: [[0] * 128 for _ in value], "expected 64 experts/rank"),
        (lambda value: [[2] + list(row[1:]) for row in value], "rank IDs must be"),
    ],
)
def test_static_mapping_rejects_malformed_ownership(mutation, match):
    with pytest.raises(ValueError, match=match):
        validate_static_mapping(mutation([list(row) for row in linear_default_mapping()]))


def test_global_to_local_map_is_dense_and_rejects_non_owner():
    row = balanced_round_robin_mapping()[0]
    rank_zero = global_to_local_map(row, 0)
    rank_one = global_to_local_map(row, 1)
    assert sorted(value for value in rank_zero if value >= 0) == list(range(64))
    assert sorted(value for value in rank_one if value >= 0) == list(range(64))
    assert all((left >= 0) != (right >= 0) for left, right in zip(rank_zero, rank_one))


def test_placement_serialization_is_deterministic(tmp_path):
    document = placement_document("balanced", balanced_round_robin_mapping())
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    assert write_placement(first, document) == write_placement(second, document)
    assert first.read_bytes() == second.read_bytes()


def _write_evidence(path: Path, rank: int, mapping):
    layers = []
    for layer_idx, row in enumerate(mapping):
        live_map = list(global_to_local_map(row, rank))
        layers.append({
            "layer": layer_idx,
            "global_to_local": live_map,
            "owned_logical_experts": [
                expert for expert, local_id in enumerate(live_map) if local_id >= 0
            ],
        })
    path.write_text(json.dumps({
        "schema_version": 1,
        "ep_rank": rank,
        "pid": 100 + rank,
        "placement_mode": "custom",
        "placement_file": "/tmp/placement.json",
        "placement_sha256": "abc",
        "layers": layers,
    }))


def test_runtime_evidence_reconstructs_requested_ownership(tmp_path):
    mapping = different_block_mapping()
    evidence = []
    for rank in range(EXPECTED_RANKS):
        path = tmp_path / f"rank-{rank}.json"
        _write_evidence(path, rank, mapping)
        evidence.append(path)
    combined = combine_runtime_evidence(evidence, mapping)
    assert combined["status"] == "pass"
    assert combined["actual_logical_to_rank"] == [list(row) for row in mapping]


def test_runtime_evidence_rejects_mapping_mismatch(tmp_path):
    evidence = []
    for rank in range(EXPECTED_RANKS):
        path = tmp_path / f"rank-{rank}.json"
        _write_evidence(path, rank, balanced_round_robin_mapping())
        evidence.append(path)
    with pytest.raises(ValueError, match="does not match"):
        combine_runtime_evidence(evidence, different_block_mapping())


def test_patch_manifest_matches_source_controlled_patch():
    manifest = json.loads(
        (REPO_ROOT / "patches" / "afd-plugin-0.26.0-static-placement.json").read_text()
    )
    patch = REPO_ROOT / "patches" / manifest["patch_file"]
    assert hashlib.sha256(patch.read_bytes()).hexdigest() == manifest["patch_sha256"]
    assert manifest["vllm_version"] == "0.26.0"


def test_static_validation_matrix_is_exactly_three_hh_cases():
    launcher = (
        REPO_ROOT / "scripts" / "milestone_m_static_placement_validation.sbatch"
    ).read_text()
    assert "2520,2520,2520,2520" in launcher
    assert 'default) placement_file=""' in launcher
    assert "balanced) placement_file=" in launcher
    assert "different) placement_file=" in launcher
    assert "routing profile" not in launcher.lower()
    assert "MILESTONE_M_STATIC_CONFIGS" in launcher
    assert "default,balanced,different" in launcher
    assert (
        '"${AFD_PYTHON}" -m tools.validate_milestone_m_static_placement'
        in launcher
    )
