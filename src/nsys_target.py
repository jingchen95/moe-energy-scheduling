"""Let the model process stop while Nsight remains alive to export its report."""

from __future__ import annotations

import os
from pathlib import Path
import signal
import subprocess
import sys


def run_target(pid_path: Path, log_path: Path, command: list[str]) -> int:
    # Reserve artifacts before spawning; never replace another run's PID/log.
    with pid_path.open("x", encoding="utf-8") as pid_file, log_path.open("x") as log:
        child = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )

        def forward_signal(signum: int, _frame: object) -> None:
            if child.poll() is None:
                os.killpg(child.pid, signum)

        previous = {
            sig: signal.signal(sig, forward_signal)
            for sig in (signal.SIGINT, signal.SIGTERM)
        }
        try:
            pid_file.write(f"{child.pid}\n")
            pid_file.flush()
            status = child.wait()
            print(f"profile target exit status: {status}", flush=True)
            # The driver intentionally sends TERM to its recorded server group.
            return 0 if status in (0, -signal.SIGTERM) else 1
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
            for sig, handler in previous.items():
                signal.signal(sig, handler)


def main() -> None:
    if len(sys.argv) < 4:
        raise SystemExit("usage: python -m src.nsys_target PID_FILE LOG_FILE COMMAND...")
    raise SystemExit(run_target(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3:]))


if __name__ == "__main__":
    main()
