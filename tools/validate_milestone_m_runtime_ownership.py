#!/usr/bin/env python3
"""Validate live two-rank AFD ownership against one frozen placement."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.afd_static_placement import (
    combine_runtime_evidence,
    load_placement,
    sha256_file,
)
from tools.prepare_milestone_m_measurement import canonical_mapping_sha256


def validate_runtime_ownership(
    placement_file: Path,
    evidence_dir: Path,
) -> dict[str, object]:
    mapping, placement_sha, placement_name = load_placement(str(placement_file))
    evidence_files = sorted(evidence_dir.glob("ownership-*.json"))
    combined = combine_runtime_evidence(evidence_files, mapping)
    expected_mapping_sha = canonical_mapping_sha256(mapping)
    if combined["actual_mapping_sha256"] != expected_mapping_sha:
        raise ValueError("runtime ownership canonical hash does not match placement")
    return {
        **combined,
        "placement_name": placement_name,
        "placement_file": str(placement_file.resolve()),
        "placement_sha256": placement_sha,
        "expected_mapping_sha256": expected_mapping_sha,
        "evidence_sha256": {
            path.name: sha256_file(path) for path in evidence_files
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--placement-file", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = validate_runtime_ownership(
        args.placement_file.resolve(strict=True),
        args.evidence_dir.resolve(strict=True),
    )
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
