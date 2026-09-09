"""Pinned Torch Profiler settings and the shared fixed-trace client entry point."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import sys
import tempfile


# The server binds its unique run directory at /run-output.
PROFILER_CONFIG = {
    "profiler": "torch",
    "torch_profiler_dir": "/run-output/profile",
    "torch_profiler_with_stack": False,
    "torch_profiler_record_shapes": False,
    "torch_profiler_with_memory": False,
    "torch_profiler_with_flops": False,
    # Gzip export first writes a full JSON via tempfile; export directly instead.
    "torch_profiler_use_gzip": False,
    # Read raw GPU events, not nested operator totals or rank-local text tables.
    "torch_profiler_dump_cuda_time_total": False,
    "ignore_frontend": True,
    "delay_iterations": 0,
    "max_iterations": 0,
}


def check_storage(
    run_dir: Path, runtime_cache: Path, min_free_bytes: int = 5 * 1024**3,
) -> dict:
    """Check the actual container paths before expensive model initialization."""
    expected_tmp = (run_dir / "tmp").resolve()
    actual_tmp = Path(tempfile.gettempdir()).resolve()
    if actual_tmp != expected_tmp:
        raise RuntimeError(f"temporary directory is {actual_tmp}, expected {expected_tmp}")
    result = {"tempfile_directory": str(actual_tmp), "minimum_free_bytes": min_free_bytes,
              "directories": {}}
    for directory in (run_dir / "profile", runtime_cache, expected_tmp):
        usage = shutil.disk_usage(directory)
        if usage.free < min_free_bytes:
            raise RuntimeError(
                f"insufficient free space at {directory}: {usage.free} bytes; "
                f"require at least {min_free_bytes} bytes"
            )
        # Check permissions/quota as well as filesystem free space. Remove only
        # this disposable probe, never cache files or previous run artifacts.
        with tempfile.NamedTemporaryFile(dir=directory, prefix="storage-probe-") as probe:
            probe.write(b"torch profiler storage preflight\n")
            probe.flush()
            os.fsync(probe.fileno())
        result["directories"][str(directory)] = {
            "total_bytes": usage.total, "free_bytes": usage.free,
        }
    return result


def main() -> None:
    if sys.argv[1:] == ["--config"]:
        print(json.dumps(PROFILER_CONFIG, sort_keys=True))
        return
    if sys.argv[1:] == ["--check-storage"]:
        print(json.dumps(check_storage(Path("/run-output"), Path("/runtime-cache")),
                         indent=2, sort_keys=True))
        return
    os.environ["MOE_PROFILE_BACKEND"] = "torch"
    # Keep one workload implementation for the Torch and historical Nsight runs.
    from src.nsys_profile import main as run_client

    run_client()


if __name__ == "__main__":
    main()
