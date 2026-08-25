#!/usr/bin/env python3
"""Create a small validation trace that retains output token IDs."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from src.workload import load_trace, write_trace


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=15)
    parser.add_argument("--request-rate", type=float, default=0.5)
    args = parser.parse_args()
    rows = load_trace(args.input)[:args.count]
    output = []
    for index, row in enumerate(rows):
        metadata = dict(row.metadata)
        metadata["store_output_token_ids"] = True
        output.append(replace(
            row, arrival_offset_s=index / args.request_rate, metadata=metadata,
        ))
    write_trace(args.output, output)


if __name__ == "__main__":
    main()
