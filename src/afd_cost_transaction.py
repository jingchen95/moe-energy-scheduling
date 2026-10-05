"""Fail-closed resume markers for identical-code formal timing lifecycles."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from src.afd_profile_protocol import IDENTITY_SOURCE, PROFILER_SCHEMA_VERSION
from src.afd_static_placement import sha256_file


REQUIRED_FILES = (
    "requests.jsonl", "resolved_config.yaml", "trace.jsonl", "resolved_trace.jsonl",
    "energy.json", "metrics.json", "gpu_samples.csv", "git_commit.txt",
    "afd_same_run_cost_profile.npz", "afd_same_run_cost_finalize.json",
    "cost-timing-validation.json", "live-validation.json", "vllm_patch_manifest.json",
    "server_metadata.json", "afd-profile-start-rpc.json", "afd-profile-stop-rpc.json",
    "stdout-stderr.log",
)


def execution_fingerprint(plan: dict, definition: dict, patch_sha: str) -> dict:
    encoded = json.dumps({"plan": plan, "configuration": definition}, sort_keys=True, separators=(",", ":")).encode()
    return {
        "source_commit": plan["git_commit"], "patch_sha256": patch_sha,
        "profiler_schema_version": PROFILER_SCHEMA_VERSION,
        "trace_sha256": plan["trace_sha256"],
        "placement_sha256": definition["placement"]["sha256"],
        "configuration_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def read_complete(directory: Path, fingerprint: dict) -> Path | None:
    marker = directory / "COMPLETE.json"
    if not marker.exists():
        return None
    value = json.loads(marker.read_text())
    if value.get("status") != "complete" or value.get("execution_fingerprint") != fingerprint:
        raise ValueError("resume execution fingerprint mismatch; no reuse permitted")
    root = directory.resolve()
    for name, expected in value["artifact_sha256"].items():
        path = (directory / name).resolve(strict=True)
        path.relative_to(root)
        if sha256_file(path) != expected:
            raise ValueError(f"completed artifact hash drift: {name}")
    run = (directory / value["run_dir"]).resolve(strict=True)
    run.relative_to(root)
    for name in REQUIRED_FILES:
        relative = str((run / name).relative_to(root))
        if relative not in value["artifact_sha256"]:
            raise ValueError(f"resume marker missing required artifact: {name}")
    return run


def mark_complete(directory: Path, run: Path, attempt: Path, fingerprint: dict, definition: dict) -> None:
    metrics = json.loads((run / "metrics.json").read_text())
    if metrics["completed_requests"] != 300 or metrics["failed_requests"] != 0:
        raise ValueError("COMPLETE requires 300/300 requests with zero failures")
    expected = [2520, 2520, definition["frequency_mhz"], definition["frequency_mhz"]]
    if any(abs(float(metrics[f"gpu{gpu}_avg_sm_clock"]) - clock) > 5 for gpu, clock in enumerate(expected)):
        raise ValueError("COMPLETE clock mismatch")
    final = json.loads((run / "afd_same_run_cost_finalize.json").read_text())
    if final.get("schema_version") != PROFILER_SCHEMA_VERSION or final.get("identity_source") != IDENTITY_SOURCE or not final.get("exact_same_run_rank_alignment"):
        raise ValueError("COMPLETE requires finalized shared runtime identity")
    if sha256_file(run / "afd_same_run_cost_profile.npz") != final["profile_sha256"]:
        raise ValueError("COMPLETE profile hash mismatch")
    if sha256_file(run / "trace.jsonl") != fingerprint["trace_sha256"]:
        raise ValueError("COMPLETE trace mismatch")
    if (run / "git_commit.txt").read_text().strip() != fingerprint["source_commit"]:
        raise ValueError("COMPLETE source mismatch")
    if sha256_file(run / "expert_placement.json") != fingerprint["placement_sha256"]:
        raise ValueError("COMPLETE placement mismatch")
    for name in ("cost-timing-validation.json", "live-validation.json"):
        if str(json.loads((run / name).read_text()).get("status", "")).lower() != "pass":
            raise ValueError(f"COMPLETE requires passing {name}")
    ownership = json.loads((attempt / "runtime-ownership.json").read_text())
    if ownership["actual_mapping_sha256"] != definition["placement"]["canonical_mapping_sha256"]:
        raise ValueError("COMPLETE ownership mismatch")
    if (attempt / "compute-apps-after.csv").read_text().strip():
        raise ValueError("COMPLETE requires successful owned-server shutdown")
    for name in REQUIRED_FILES:
        if not (run / name).is_file():
            raise ValueError(f"COMPLETE missing required file: {name}")
    for name in ("server/attention.log", "server/ffn.log", "server/attention-command.txt", "server/ffn-command.txt"):
        if not (attempt / name).is_file():
            raise ValueError(f"COMPLETE missing role log/command: {name}")
    root = directory.resolve()
    artifacts = {
        str(path.resolve().relative_to(root)): sha256_file(path)
        for path in attempt.rglob("*") if path.is_file()
    }
    value = {"status": "complete", "execution_fingerprint": fingerprint,
             "run_dir": str(run.resolve().relative_to(root)), "artifact_sha256": artifacts}
    temporary = directory / f"COMPLETE.{attempt.name}.tmp"
    with temporary.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
    # A stale COMPLETE marker is never replaced, even if this attempt passed.
    if (directory / "COMPLETE.json").exists():
        raise FileExistsError("COMPLETE marker already exists")
    temporary.rename(directory / "COMPLETE.json")


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["check", "mark"])
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--order", type=int, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--patch-sha", required=True)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--attempt", type=Path)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    definition = next(row for row in plan["configurations"] if row["order"] == args.order)
    fingerprint = execution_fingerprint(plan, definition, args.patch_sha)
    if args.action == "check":
        run = read_complete(args.directory, fingerprint)
        if run:
            print(run)
    else:
        mark_complete(args.directory, args.run, args.attempt, fingerprint, definition)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
