#!/usr/bin/env python3
"""Regenerate the cost-profile patch and exact hashes from a pinned checkout."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


TARGETS = (
    "afd_plugin/connectors/metadata.py",
    "afd_plugin/model_executor/models/qwen3_moe.py",
    "afd_plugin/v1/worker/attention_metadata.py",
    "afd_plugin/v1/worker/attention_model_runner.py",
    "afd_plugin/v1/worker/attention_worker.py",
    "afd_plugin/v1/worker/ffn_model_runner.py",
    "afd_plugin/v1/worker/ffn_worker.py",
)
PIN = "4c2541a53ec8ad507508073eac81ac9feecc6c9a"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    def git(*arguments):
        return subprocess.check_output(["git", "-C", str(root), *arguments])
    if git("rev-parse", "HEAD").decode().strip() != PIN:
        raise ValueError("plugin checkout does not match pin")
    patch = git("diff", "HEAD", "--", *TARGETS)
    if not patch:
        raise ValueError("no patch differences")
    directory = Path(__file__).resolve().parents[1] / "patches"
    patch_path = directory / "afd-plugin-0.26.0-cost-model.patch"
    patch_path.write_bytes(patch)
    manifest = {
        "schema_version": 1, "name": "afd-plugin-0.26.0-cost-model",
        "afd_plugin_commit": PIN, "vllm_version": "0.26.0",
        "profiler_schema_version": 3,
        "targets": [{
            "path": target,
            "source_sha256_before": hashlib.sha256(git("show", f"HEAD:{target}")).hexdigest(),
            "source_sha256_after": hashlib.sha256((root / target).read_bytes()).hexdigest(),
        } for target in TARGETS],
        "patch_file": patch_path.name,
        "patch_sha256": hashlib.sha256(patch).hexdigest(),
        "scope": "static ownership, shared Attention-DP identity/window payload, graph-safe same-run timing",
    }
    (directory / "afd-plugin-0.26.0-cost-model.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
