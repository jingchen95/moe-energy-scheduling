#!/usr/bin/env python3
"""Describe legacy rank boundaries without repairing or joining them."""

import argparse
import json
from pathlib import Path

import numpy as np


def diagnose(study: Path) -> dict:
    configurations = []
    for config in sorted((study / "timing-configurations").iterdir()):
        runs = sorted((config / "runs").iterdir())
        if not runs:
            continue
        for run in runs:
            ranks = []
            for rank in (0, 1):
                paths = list((config / "timing-control").glob(f"afd-cost-rank-{rank}-pid-*-{run.name}.json"))
                if len(paths) != 1:
                    raise ValueError(f"ambiguous rank metadata in {config}")
                metadata = json.loads(paths[0].read_text())
                parts = []
                for chunk in metadata["chunks"]:
                    with np.load(config / "timing-control" / chunk["path"], allow_pickle=False) as data:
                        parts.append(np.stack([data[key] for key in
                            ("step_sequences", "stages", "layers", "token_rows")], axis=1))
                rows = np.concatenate(parts)
                steps = np.unique(rows[:, 0])
                def boundary(ids):
                    return [{"local_step": int(step), "stages": np.unique(rows[rows[:, 0] == step, 1]).tolist(),
                             "layers": int(np.sum(rows[:, 0] == step)),
                             "token_rows": np.unique(rows[rows[:, 0] == step, 3]).tolist()} for step in ids]
                ranks.append({"rank": rank, "records": len(rows), "steps": len(steps),
                              "first_local_step": int(steps[0]), "last_local_step": int(steps[-1]),
                              "head": boundary(steps[:6]), "tail": boundary(steps[-6:]),
                              "run_mode": "not recorded in legacy schema",
                              "shared_runtime_identity": "not recorded in legacy schema"})
            configurations.append({"configuration": config.name, "run": run.name,
                                   "finalized": (run / "metrics.json").exists(), "ranks": ranks})
    return {"study": str(study), "configurations": configurations,
            "conclusion": "one extra rank-1 step; activation versus deactivation cannot be established",
            "legacy_training_reuse": False, "truncation_or_offset_join": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = diagnose(args.study)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
