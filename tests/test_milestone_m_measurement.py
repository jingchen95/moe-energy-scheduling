import random
from pathlib import Path

from src.afd_static_placement import load_placement
from tools.analyze_milestone_m_measurements import classify
from tools.prepare_milestone_m_measurement import (
    CONFIGURATIONS,
    PLACEMENT_FILES,
    PLACEMENT_SHA256,
    RANDOMIZATION_SEED,
    canonical_mapping_sha256,
)
from tools.validate_milestone_m_heterogeneous_smoke import _validate_clocks


def test_primary_matrix_is_exact_and_deterministically_ordered() -> None:
    assert CONFIGURATIONS == (
        ("C0", "P_balanced", (2520, 2520)),
        ("C1", "P_MH", (2520, 2520)),
        ("C2", "P_balanced", (1635, 2520)),
        ("C3", "P_MH", (1635, 2520)),
        ("C4", "P_balanced", (2520, 1635)),
        ("C5", "P_HM", (2520, 1635)),
    )
    order = list(range(6))
    random.Random(RANDOMIZATION_SEED).shuffle(order)
    assert order == [0, 1, 5, 2, 4, 3]
    assert 3 * len(CONFIGURATIONS) == 18


def test_frozen_placements_and_canonical_hashes_are_stable() -> None:
    observed_mapping_hashes = set()
    for name, relative in PLACEMENT_FILES.items():
        mapping, file_sha, document_name = load_placement(str(relative.resolve()))
        assert document_name == name
        assert file_sha == PLACEMENT_SHA256[name]
        assert len(mapping) == 48
        assert all(row.count(0) == row.count(1) == 64 for row in mapping)
        mapping_sha = canonical_mapping_sha256(mapping)
        assert len(mapping_sha) == 64
        observed_mapping_hashes.add(mapping_sha)
    assert len(observed_mapping_hashes) == 3


def _comparison(*, energy: float = 0.0, tpot: float = 0.0) -> dict[str, float]:
    row = {
        "total_gpu_energy_j_relative_change": energy,
        "tpot_p99_ms_relative_change": tpot,
        "ttft_p99_ms_relative_change": 0.0,
        "throughput_requests_s_relative_change": 0.0,
    }
    for metric in ("total_gpu_energy_j", "tpot_p99_ms"):
        row[f"{metric}_baseline_mean"] = 100.0
        row[f"{metric}_baseline_sample_std"] = 0.5
        row[f"{metric}_treatment_sample_std"] = 0.5
    return row


def test_conservative_decision_classification() -> None:
    control = _comparison()
    strong = {
        "primary_mh": _comparison(energy=-0.04),
        "primary_hm": _comparison(energy=-0.04),
        "control_hh": control,
    }
    assert classify(strong)[0] == "A"
    weak = {
        "primary_mh": _comparison(energy=-0.015),
        "primary_hm": _comparison(),
        "control_hh": control,
    }
    assert classify(weak)[0] == "B"
    absent = {
        "primary_mh": _comparison(),
        "primary_hm": _comparison(),
        "control_hh": control,
    }
    assert classify(absent)[0] == "C"


def test_full_decode_only_is_explicit_and_profile_is_disabled() -> None:
    script = Path("scripts/milestone_m_primary.sbatch").read_text()
    common = Path("scripts/afd_common.sh").read_text()
    assert "full_decode_only" in script
    assert "--afd-profile-control-dir" not in script
    assert "FULL_DECODE_ONLY" in common
    assert "--enforce-eager" in common


def test_clock_validator_ignores_deprecated_application_field(tmp_path: Path) -> None:
    readback = tmp_path / "readback.csv"
    readback.write_text("\n".join(
        f"{gpu}, {clock}, [Requested functionality has been deprecated]"
        for gpu, clock in enumerate((2520, 2520, 1635, 2520))
    ) + "\n")
    samples = tmp_path / "samples.csv"
    samples.write_text(
        "gpu,sm_util_percent,sm_clock_mhz\n"
        "0,1,2520\n1,1,2520\n2,1,1635\n3,1,2520\n"
    )

    result = _validate_clocks(
        samples, readback, [2520, 2520, 1635, 2520]
    )

    assert result["current_clock_readback_mhz"] == [2520, 2520, 1635, 2520]
