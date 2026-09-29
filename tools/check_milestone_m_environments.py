#!/usr/bin/env python3
"""Assert that Milestone M uses an isolated pinned AFD runtime."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.afd_environment import (
    environment_record,
    inspect_runtime,
    validate_runtime_pair,
)


REPOSITORY = "https://github.com/vllm-project/afd-plugin.git"
PINNED_COMMIT = "4c2541a53ec8ad507508073eac81ac9feecc6c9a"


def _git(plugin_root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(plugin_root), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-python", type=Path, required=True)
    parser.add_argument("--afd-python", type=Path, required=True)
    parser.add_argument("--plugin-root", type=Path, required=True)
    parser.add_argument("--uv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    old = inspect_runtime(args.old_python)
    afd = inspect_runtime(args.afd_python)
    commit = _git(args.plugin_root, "rev-parse", "HEAD")
    clean = not bool(_git(args.plugin_root, "status", "--porcelain"))
    uv_version = subprocess.run(
        [str(args.uv.resolve()), "--version"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    validate_runtime_pair(
        old,
        afd,
        expected_plugin_commit=PINNED_COMMIT,
        plugin_commit=commit,
    )
    record = environment_record(
        old,
        afd,
        plugin_repository=REPOSITORY,
        plugin_commit=commit,
        plugin_clean=clean,
        uv_version=uv_version,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
