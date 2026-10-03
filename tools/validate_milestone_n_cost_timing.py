#!/usr/bin/env python3
"""Fail-fast validation of one AFD FFN timing run against the routing trace."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.afd_cost_model import join_timing_run, load_routing_trace


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--routing-trace", type=Path, required=True)
    parser.add_argument("--timing-run", type=Path, required=True)
    parser.add_argument("--placement", type=Path, required=True)
    parser.add_argument("--config-id", required=True)
    parser.add_argument("--lifecycle", required=True)
    parser.add_argument("--frequency-mhz", type=int, required=True)
    args = parser.parse_args()
    paths = sorted(args.timing_run.glob("ffn-timing-rank-*.npz"))
    routing = load_routing_trace(args.routing_trace)
    rows, _ = join_timing_run(
        routing,
        paths,
        args.placement,
        config_id=args.config_id,
        lifecycle=args.lifecycle,
        frequency_mhz=args.frequency_mhz,
    )
    by_rank = {
        str(rank): sum(int(row["physical_ffn_rank"]) == rank for row in rows)
        for rank in (0, 1)
    }
    result = {
        "status": "pass",
        "config_id": args.config_id,
        "joined_records": len(rows),
        "records_by_rank": by_rank,
        "exact_layer_call_alignment": True,
        "exact_token_row_alignment": True,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
