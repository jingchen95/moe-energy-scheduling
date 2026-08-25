#!/usr/bin/env python3
"""Change only arrival offsets while preserving a canonical trace's content/order."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from src.workload import load_trace, write_trace


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--request-rate", type=float, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.request_rate <= 0:
        parser.error("--request-rate must be positive")
    rows = load_trace(args.input)
    if args.limit is not None:
        rows = rows[:args.limit]
    write_trace(args.output, [replace(row, arrival_offset_s=index / args.request_rate)
                              for index, row in enumerate(rows)])


if __name__ == "__main__":
    main()
