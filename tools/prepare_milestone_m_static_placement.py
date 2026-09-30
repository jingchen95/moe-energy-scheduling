#!/usr/bin/env python3
"""Prepare deterministic two-rank AFD static-placement correctness inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.afd_static_placement import (
    balanced_round_robin_mapping,
    different_block_mapping,
    linear_default_mapping,
    placement_document,
    sha256_file,
    write_placement,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--patch", type=Path, required=True)
    parser.add_argument("--baseline-run", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    placements_dir = output_dir / "placements"

    balanced = balanced_round_robin_mapping()
    different = different_block_mapping()
    default = linear_default_mapping()
    documents = {
        "balanced_round_robin": placement_document(
            "afd_balanced_round_robin",
            balanced,
            metadata={
                "algorithm": "rank=(logical_expert+layer) mod 2",
                "profile_used": False,
                "replication": False,
            },
        ),
        "different_block": placement_document(
            "afd_different_16_expert_blocks",
            different,
            metadata={
                "algorithm": "rank=(floor(logical_expert/16)+layer) mod 2",
                "profile_used": False,
                "replication": False,
            },
        ),
    }
    placement_records = {}
    for key, document in documents.items():
        path = placements_dir / f"{key}.json"
        placement_records[key] = {
            "path": str(path),
            "sha256": write_placement(path, document),
        }

    baseline_responses = args.baseline_run / "responses-pass-1.jsonl"
    if not baseline_responses.is_file():
        raise FileNotFoundError(f"missing baseline responses: {baseline_responses}")
    if not args.patch.is_file():
        raise FileNotFoundError(f"missing patch: {args.patch}")

    manifest = {
        "schema_version": 1,
        "study": "milestone_m_static_placement_hh_correctness",
        "baseline_run": str(args.baseline_run.resolve()),
        "baseline_responses_sha256": sha256_file(baseline_responses),
        "patch": str(args.patch.resolve()),
        "patch_sha256": sha256_file(args.patch),
        "frequencies_mhz": [2520, 2520, 2520, 2520],
        "configurations": [
            {"id": "default", "placement": None},
            {"id": "balanced", "placement": placement_records["balanced_round_robin"]},
            {"id": "different", "placement": placement_records["different_block"]},
        ],
        "mapping_differences": {
            "balanced_vs_default": sum(
                left != right
                for left_row, right_row in zip(balanced, default)
                for left, right in zip(left_row, right_row)
            ),
            "different_vs_default": sum(
                left != right
                for left_row, right_row in zip(different, default)
                for left, right in zip(left_row, right_row)
            ),
            "different_vs_balanced": sum(
                left != right
                for left_row, right_row in zip(different, balanced)
                for left, right in zip(left_row, right_row)
            ),
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
