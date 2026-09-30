#!/usr/bin/env python3
"""Record immutable plan/runtime identities beside one measured run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--ownership", type=Path, required=True)
    parser.add_argument("--config-id", required=True)
    parser.add_argument("--repetition", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    ownership = json.loads(args.ownership.read_text())
    record = {
        "schema_version": 1,
        "config_id": args.config_id,
        "repetition": args.repetition,
        "experiment_plan_sha256": sha256_file(args.plan),
        "git_commit": plan["git_commit"],
        "plugin_commit": plan["plugin_commit"],
        "vllm_version": plan["vllm_version"],
        "model": plan["model"],
        "topology": plan["topology"],
        "execution_mode": plan["execution_mode"],
        "trace_sha256": plan["trace_sha256"],
        "placement_sha256": ownership["placement_sha256"],
        "actual_mapping_sha256": ownership["actual_mapping_sha256"],
        "routing_profiling": plan["routing_profiling"],
        "custom_kernel_timing": plan["custom_kernel_timing"],
    }
    args.output.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
