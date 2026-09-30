#!/usr/bin/env python3
"""Validate one profiling-only 2A2F run and preserve output identity."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = json.loads((args.study_dir / "manifest.json").read_text())
    metrics = json.loads((args.run_dir / "metrics.json").read_text())
    requests = [
        json.loads(line)
        for line in (args.run_dir / "requests.jsonl").read_text().splitlines()
        if line
    ]
    if len(requests) != 300 or metrics["completed_requests"] != 300:
        raise ValueError("profiling run did not complete 300 requests")
    if metrics["failed_requests"] != 0 or any(row["status"] != "ok" for row in requests):
        raise ValueError("profiling run contains failed requests")
    if sha256(args.run_dir / "trace.jsonl") != manifest["trace_sha256"]:
        raise ValueError("run trace does not match frozen profile trace")
    if any(row["output_tokens"] <= 1 for row in requests):
        raise ValueError("profiling run contains an invalid output-token count")

    with np.load(args.run_dir / "expert_counts.npz", allow_pickle=False) as archive:
        counts = np.asarray(archive["expert_counts"], dtype=np.int64)
        per_rank = np.asarray(archive["per_rank_expert_counts"], dtype=np.int64)
    if counts.shape != (48, 128) or per_rank.shape != (2, 48, 128):
        raise ValueError("routing counts have an invalid shape")
    if np.any(counts < 0) or not np.array_equal(counts, per_rank.sum(axis=0)):
        raise ValueError("rank-local routing counts do not conserve the total")
    assignments_by_layer = counts.sum(axis=1)
    if assignments_by_layer.min() <= 0:
        raise ValueError("one or more layers have no captured routes")
    if not np.all(assignments_by_layer == assignments_by_layer[0]):
        raise ValueError("logical routed-token assignments differ across layers")

    output_identity = [{
        "request_id": row["request_id"],
        "prompt_tokens": row.get("server_prompt_tokens"),
        "output_tokens": row["output_tokens"],
        "output_text_sha256": row["output_text_sha256"],
        "output_token_ids": row.get("output_token_ids"),
    } for row in requests]
    canonical = json.dumps(
        output_identity, separators=(",", ":"), sort_keys=True
    ).encode()
    summary = {
        "schema_version": 1,
        "status": "pass",
        "trace_sha256": manifest["trace_sha256"],
        "completed_requests": 300,
        "failed_requests": 0,
        "input_tokens_total_server": sum(
            int(row.get("server_prompt_tokens") or 0) for row in requests
        ),
        "output_tokens_total": sum(int(row["output_tokens"]) for row in requests),
        "output_identity_sha256": hashlib.sha256(canonical).hexdigest(),
        "expert_counts_sha256": sha256(args.run_dir / "expert_counts.npz"),
        "expert_counts_shape": list(counts.shape),
        "per_rank_counts_shape": list(per_rank.shape),
        "logical_assignments_per_layer": int(assignments_by_layer[0]),
        "logical_routed_tokens_per_layer": int(assignments_by_layer[0] // 8),
        "top_k": 8,
        "per_rank_assignments": per_rank.sum(axis=(1, 2)).astype(int).tolist(),
    }
    (args.run_dir / "profile-validation.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.run_dir / "output-identity.json").write_text(
        json.dumps(output_identity, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
