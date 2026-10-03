import numpy as np
import pytest

from src.afd_target_split import (
    TARGET_SLOW_FRACTIONS,
    mirror_placement,
    split_diagnostics,
    target_split_placement,
)
from src.placement import Placement


def _counts() -> np.ndarray:
    row = np.arange(1, 129, dtype=np.int64)
    return np.tile(row, (48, 1))


def _base() -> Placement:
    mapping = np.tile(np.arange(128, dtype=np.int64) % 2, (48, 1))
    return Placement("P_balanced", mapping, 2)


@pytest.mark.parametrize("target", TARGET_SLOW_FRACTIONS)
def test_target_split_is_deterministic_cardinality_safe_and_close(target: float) -> None:
    first = target_split_placement(
        _counts(), _base(), target, name="target"
    )
    second = target_split_placement(
        _counts(), _base(), target, name="target"
    )
    assert np.array_equal(first.logical_to_rank, second.logical_to_rank)
    assert all(np.bincount(row, minlength=2).tolist() == [64, 64]
               for row in first.logical_to_rank)
    diagnostics = split_diagnostics(
        _counts(), first, slow_rank=0, target_slow_fraction=target
    )
    assert abs(diagnostics["achieved_slow_fraction"] - target) < 0.002


def test_mirror_preserves_groups_and_moves_slow_fraction_to_rank1() -> None:
    primary = target_split_placement(
        _counts(), _base(), 0.4, name="primary"
    )
    mirrored = mirror_placement(primary, name="mirrored")
    assert np.array_equal(mirrored.logical_to_rank, 1 - primary.logical_to_rank)
    diagnostics = split_diagnostics(
        _counts(), mirrored, slow_rank=1, target_slow_fraction=0.4
    )
    assert abs(diagnostics["achieved_slow_fraction"] - 0.4) < 0.002


@pytest.mark.parametrize("target", (0.0, -0.1, 0.51))
def test_invalid_target_is_rejected(target: float) -> None:
    with pytest.raises(ValueError):
        target_split_placement(_counts(), _base(), target, name="invalid")
