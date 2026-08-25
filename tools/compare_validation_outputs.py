#!/usr/bin/env python3
"""Compare deterministic output token sequences from two validation runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def rows(path: Path) -> dict[str, dict]:
    return {row["request_id"]: row for row in
            (json.loads(line) for line in (path / "requests.jsonl").read_text().splitlines())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()
    reference = rows(args.reference)
    candidate = rows(args.candidate)
    if reference.keys() != candidate.keys():
        raise SystemExit("validation request IDs differ")
    mismatches = [identifier for identifier in reference
                  if reference[identifier].get("output_token_ids")
                  != candidate[identifier].get("output_token_ids")]
    result = {"reference": str(args.reference), "candidate": str(args.candidate),
              "requests": len(reference), "token_sequence_mismatches": mismatches}
    print(json.dumps(result, indent=2))
    if mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
