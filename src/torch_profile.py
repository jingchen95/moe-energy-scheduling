"""Pinned Torch Profiler settings and the shared fixed-trace client entry point."""

from __future__ import annotations

import json
import os
import sys


# The server binds its unique run directory at /run-output.
PROFILER_CONFIG = {
    "profiler": "torch",
    "torch_profiler_dir": "/run-output/profile",
    "torch_profiler_with_stack": False,
    "torch_profiler_record_shapes": False,
    "torch_profiler_with_memory": False,
    "torch_profiler_with_flops": False,
    "torch_profiler_use_gzip": True,
    # Read raw GPU events, not nested operator totals or rank-local text tables.
    "torch_profiler_dump_cuda_time_total": False,
    "ignore_frontend": True,
    "delay_iterations": 0,
    "max_iterations": 0,
}


def main() -> None:
    if sys.argv[1:] == ["--config"]:
        print(json.dumps(PROFILER_CONFIG, sort_keys=True))
        return
    os.environ["MOE_PROFILE_BACKEND"] = "torch"
    # Keep one workload implementation for the Torch and historical Nsight runs.
    from src.nsys_profile import main as run_client

    run_client()


if __name__ == "__main__":
    main()
