import itertools

import pytest

from tools.analyze_rank_permutation_dvfs_alignment import (
    ALL_GROUP_AT_GPU_PERMUTATIONS,
    H,
    IDENTITY,
    apply_group_permutation,
    choose_aligned_permutation,
    frequency_tuples,
    invert_group_at_gpu,
    permute_mapping_row,
)


def test_all_24_group_permutations_are_present_once():
    assert len(ALL_GROUP_AT_GPU_PERMUTATIONS) == 24
    assert len(set(ALL_GROUP_AT_GPU_PERMUTATIONS)) == 24
    assert set(ALL_GROUP_AT_GPU_PERMUTATIONS) == set(itertools.permutations(range(4)))


def test_permutation_preserves_exactly_the_four_p1_groups():
    groups = (11, 22, 33, 44)
    for permutation in ALL_GROUP_AT_GPU_PERMUTATIONS:
        physical = apply_group_permutation(groups, permutation)
        assert sorted(physical) == sorted(groups)

    original = (0, 0, 1, 1, 2, 2, 3, 3)
    group_at_gpu = (2, 0, 3, 1)
    group_to_gpu = invert_group_at_gpu(group_at_gpu)
    remapped = permute_mapping_row(original, group_to_gpu)
    for group, gpu in enumerate(group_to_gpu):
        original_members = {index for index, value in enumerate(original) if value == group}
        remapped_members = {index for index, value in enumerate(remapped) if value == gpu}
        assert remapped_members == original_members


def test_optimal_assignment_matches_light_groups_to_slow_gpus():
    groups = (0.10, 0.20, 0.30, 0.40)
    frequencies = (H, 1005, 1635, H)
    result = choose_aligned_permutation(groups, frequencies)
    assert result["group_at_gpu"] == (2, 0, 1, 3)
    assert result["physical_load_fractions"] == pytest.approx((0.30, 0.10, 0.20, 0.40))
    assert result["rank_order_inversions"] == 0


def test_equal_frequency_has_no_alignment_advantage_and_keeps_identity():
    groups = (0.31, 0.18, 0.29, 0.22)
    result = choose_aligned_permutation(groups, (H, H, H, H))
    assert result["group_at_gpu"] == IDENTITY
    assert result["max_pressure"] == pytest.approx(max(groups))
    assert result["identity_distance"] == 0


def test_alignment_is_reproducible():
    groups = (0.37, 0.18, 0.24, 0.21)
    frequencies = (1635, H, 1005, H)
    first = choose_aligned_permutation(groups, frequencies)
    second = choose_aligned_permutation(groups, frequencies)
    assert first == second
    assert len(frequency_tuples()) == 81
