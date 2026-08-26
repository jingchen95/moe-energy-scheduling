#!/usr/bin/env python3
"""Compare deterministic output token sequences from two validation runs."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def rows(path: Path) -> dict[str, dict]:
    return {row["request_id"]: row for row in
            (json.loads(line) for line in (path / "requests.jsonl").read_text().splitlines())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument(
        "--minimum-common-prefix",
        type=int,
        default=None,
        help="Allow later numerical divergence only when every request shares this token prefix.",
    )
    args = parser.parse_args()
    reference = rows(args.reference)
    candidate = rows(args.candidate)
    if reference.keys() != candidate.keys():
        raise SystemExit("validation request IDs differ")
    mismatches = []
    common_prefix_tokens = {}
    for identifier in reference:
        expected = reference[identifier].get("output_token_ids") or []
        observed = candidate[identifier].get("output_token_ids") or []
        prefix = 0
        for left, right in zip(expected, observed):
            if left != right:
                break
            prefix += 1
        common_prefix_tokens[identifier] = prefix
        if expected != observed:
            mismatches.append(identifier)
    minimum_observed_prefix = min(common_prefix_tokens.values(), default=0)
    accepted = not mismatches or (
        args.minimum_common_prefix is not None
        and minimum_observed_prefix >= args.minimum_common_prefix
    )
    result = {"reference": str(args.reference), "candidate": str(args.candidate),
              "requests": len(reference), "exact_matches": len(reference) - len(mismatches),
              "exact_match_fraction": ((len(reference) - len(mismatches)) / len(reference)
                                       if reference else 1.0),
              "token_sequence_mismatches": mismatches,
              "common_prefix_tokens": common_prefix_tokens,
              "minimum_observed_prefix": minimum_observed_prefix,
              "median_observed_prefix": statistics.median(common_prefix_tokens.values())
              if common_prefix_tokens else 0,
              "required_common_prefix": args.minimum_common_prefix,
              "accepted": accepted}
    print(json.dumps(result, indent=2))
    if not accepted:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
