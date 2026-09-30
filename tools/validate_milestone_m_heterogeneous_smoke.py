#!/usr/bin/env python3
"""Validate the P_MH / MH heterogeneous AFD smoke gate."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean

from tools.validate_afd_smoke import validate_gpu_samples, validate_outputs
from tools.validate_milestone_m_runtime_ownership import validate_runtime_ownership


def _validate_clocks(
    samples_path: Path,
    readback_path: Path,
    targets: list[int],
) -> dict[str, object]:
    readbacks = [
        [int(field.strip()) for field in line.split(",")]
        for line in readback_path.read_text().splitlines()
        if line.strip()
    ]
    if len(readbacks) != 4:
        raise ValueError("expected four clock readback rows")
    applications = {row[0]: row[2] for row in readbacks}
    if [applications[gpu] for gpu in range(4)] != targets:
        raise ValueError(f"application-clock readback mismatch: {applications}")

    samples = list(csv.DictReader(samples_path.open(newline="", encoding="utf-8")))
    result: dict[str, object] = {"target_clocks_mhz": targets}
    for gpu in range(4):
        active = [
            row for row in samples
            if int(row["gpu"]) == gpu and float(row["sm_util_percent"]) > 0
        ]
        if not active:
            raise ValueError(f"GPU {gpu} has no active clock samples")
        clocks = [float(row["sm_clock_mhz"]) for row in active]
        if not any(abs(value - targets[gpu]) <= 5 for value in clocks):
            raise ValueError(
                f"GPU {gpu} never reached target {targets[gpu]} MHz while active"
            )
        result[f"gpu{gpu}_active_average_clock_mhz"] = mean(clocks)
        result[f"gpu{gpu}_active_samples"] = len(clocks)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, required=True)
    args = parser.parse_args()
    study = args.study_dir.resolve(strict=True)
    manifest = json.loads((study / "manifest.json").read_text())
    baseline = Path(manifest["baseline_run"])
    baseline_rows = validate_outputs(
        baseline / "responses-pass-1.jsonl",
        baseline / "responses-pass-2.jsonl",
    )
    first = study / "responses-pass-1.jsonl"
    second = study / "responses-pass-2.jsonl"
    observed_rows = validate_outputs(first, second)
    if observed_rows != baseline_rows:
        raise ValueError("heterogeneous smoke outputs differ from validated baseline")

    placement = Path(manifest["placement"]["path"])
    ownership = validate_runtime_ownership(placement, study / "ownership")
    if ownership["placement_sha256"] != manifest["placement"]["sha256"]:
        raise ValueError("placement file hash drift")
    if (
        ownership["actual_mapping_sha256"]
        != manifest["placement"]["canonical_mapping_sha256"]
    ):
        raise ValueError("actual ownership hash mismatch")

    command_text = "\n".join(
        (study / "servers" / name).read_text()
        for name in ("attention-command.txt", "ffn-command.txt")
    )
    if "FULL_DECODE_ONLY" not in command_text or "--enforce-eager" in command_text:
        raise ValueError("smoke did not use the planned FULL_DECODE_ONLY mode")
    gpu_usage = validate_gpu_samples(study / "gpu-samples.csv")
    for gpu in (2, 3):
        if gpu_usage[str(gpu)]["max_sm_util_percent"] <= 0:
            raise ValueError(f"FFN GPU {gpu} showed no execution activity")
    clocks = _validate_clocks(
        study / "gpu-samples.csv",
        study / "actual-clocks-before-client.csv",
        list(manifest["target_clocks_mhz"]),
    )
    summary = {
        "schema_version": 1,
        "status": "pass",
        "configuration": "P_MH / Attention HH / FFN MH",
        "execution_mode": manifest["execution_mode"],
        "outputs_match_validated_baseline": True,
        "placement_sha256": ownership["placement_sha256"],
        "actual_mapping_sha256": ownership["actual_mapping_sha256"],
        "gpu_usage": gpu_usage,
        "clocks": clocks,
    }
    (study / "validation-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
