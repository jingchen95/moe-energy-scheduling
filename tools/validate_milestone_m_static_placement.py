#!/usr/bin/env python3
"""Validate live AFD ownership evidence and cross-placement smoke outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.afd_static_placement import (
    combine_runtime_evidence,
    linear_default_mapping,
    load_placement,
    sha256_file,
)
from tools.validate_afd_smoke import validate_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    study_dir = args.study_dir.resolve()
    manifest = json.loads((study_dir / "manifest.json").read_text())
    baseline_run = Path(manifest["baseline_run"])
    baseline_rows = validate_outputs(
        baseline_run / "responses-pass-1.jsonl",
        baseline_run / "responses-pass-2.jsonl",
    )
    results = []
    for config in manifest["configurations"]:
        config_dir = study_dir / config["id"]
        responses_first = config_dir / "responses-pass-1.jsonl"
        responses_second = config_dir / "responses-pass-2.jsonl"
        response_rows = validate_outputs(responses_first, responses_second)
        if response_rows != baseline_rows:
            raise ValueError(
                f"{config['id']}: outputs differ from unmodified AFD baseline"
            )
        if config["placement"] is None:
            expected_mapping = linear_default_mapping()
            requested_sha = None
        else:
            placement_path = Path(config["placement"]["path"])
            expected_mapping, requested_sha, _name = load_placement(str(placement_path))
            if requested_sha != config["placement"]["sha256"]:
                raise ValueError(f"{config['id']}: placement hash drift")
        evidence_files = sorted((config_dir / "ownership").glob("ownership-*.json"))
        ownership = combine_runtime_evidence(evidence_files, expected_mapping)
        ownership["requested_placement_sha256"] = requested_sha
        ownership_path = config_dir / "actual-ownership.json"
        ownership_path.write_text(json.dumps(ownership, indent=2, sort_keys=True) + "\n")
        results.append({
            "configuration": config["id"],
            "placement_sha256": requested_sha,
            "actual_ownership_sha256": sha256_file(ownership_path),
            "response_sha256": sha256_file(responses_first),
            "status": "pass",
        })
    summary = {
        "schema_version": 1,
        "status": "pass",
        "baseline_response_sha256": manifest["baseline_responses_sha256"],
        "configurations": results,
    }
    (study_dir / "validation-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
