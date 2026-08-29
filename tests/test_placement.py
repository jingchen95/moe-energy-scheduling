import json

import numpy as np
import pytest

from src.placement import (
    Placement,
    diagnostic_concentrated_placement,
    controlled_load_placements,
    frequency_aware_placement,
    linear_placement,
    physical_to_rank_placement,
    placement_from_dict,
    placement_loads,
    upstream_balanced_placement,
)


def test_mapping_validation_requires_equal_cardinality():
    mapping = np.tile(np.repeat(np.arange(4), 32), (2, 1))
    Placement("ok", mapping).validate()
    mapping[0, 0] = 1
    with pytest.raises(ValueError, match="32 experts per rank"):
        Placement("bad", mapping).validate()


def test_file_validation_rejects_metadata_mismatch():
    value = linear_placement(2).to_dict()
    value["num_experts"] = 127
    with pytest.raises(ValueError, match="num_experts"):
        placement_from_dict(json.loads(json.dumps(value)))


def test_physical_map_conversion_and_fake_upstream_policy():
    physical = np.array([[2, 0, 3, 1]])
    result = physical_to_rank_placement(physical, name="x", num_ranks=2)
    assert result.logical_to_rank.tolist() == [[0, 1, 0, 1]]

    def fake_policy(weight, num_replicas, num_groups, num_nodes, num_ranks):
        assert weight.shape == (1, 4)
        return physical, None, None

    balanced = upstream_balanced_placement(
        np.array([[4, 1, 3, 2]], dtype=np.int64), num_groups=1,
        num_ranks=2, policy=fake_policy,
    )
    assert balanced.logical_to_rank.tolist() == [[0, 1, 0, 1]]


def test_diagnostic_placement_concentrates_hot_experts_deterministically():
    counts = np.arange(128, dtype=np.int64)[None, :]
    result = diagnostic_concentrated_placement(counts)
    result.validate()
    assert np.where(result.logical_to_rank[0] == 0)[0].tolist() == list(range(96, 128))
    assert np.bincount(result.logical_to_rank[0], minlength=4).tolist() == [32] * 4
    loads = placement_loads(counts, result)
    assert loads[0] > loads[1] > loads[2] > loads[3]


def test_frequency_aware_placement_obeys_capacity_and_cardinality():
    counts = np.arange(1, 129, dtype=np.int64)[None, :]
    result = frequency_aware_placement(counts, (2520, 1635, 1635, 1005))
    result.validate()
    assert np.bincount(result.logical_to_rank[0], minlength=4).tolist() == [32] * 4
    assert result.metadata["speed_factors"] == pytest.approx(
        [1.0, 1635 / 2520, 1635 / 2520, 1005 / 2520]
    )
    raw = result.metadata["per_rank_raw_load"]
    assert raw[0] > raw[3]


def test_frequency_aware_swaps_never_worsen_objective():
    counts = np.array([np.roll(np.arange(1, 129), shift) for shift in range(3)])
    result = frequency_aware_placement(counts, (2520, 1635, 1635, 1005),
                                       max_swap_passes=8)
    before = result.metadata["per_layer_before_objective"]
    after = result.metadata["per_layer_after_objective"]
    assert all(final <= initial for initial, final in zip(before, after))


def test_controlled_load_levels_preserve_invariants_and_order_gpu0_load():
    counts = np.tile(np.arange(1, 129, dtype=np.int64), (2, 1))
    levels = controlled_load_placements(counts)
    assert [p.name for p in levels] == [
        "controlled_low", "controlled_medium", "controlled_high",
        "controlled_very_high",
    ]
    loads = [placement_loads(counts, placement) for placement in levels]
    for placement in levels:
        placement.validate()
    assert [int(value[0]) for value in loads] == sorted(int(value[0]) for value in loads)
    assert loads[0][0] < loads[-1][0]
    assert np.array_equal(
        levels[-1].logical_to_rank == 0,
        diagnostic_concentrated_placement(counts).logical_to_rank == 0,
    )
