"""Exercise the process boundary that must leave Nsight alive for export."""

import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

from src.nsys_target import run_target


class ProfileTargetTests(unittest.TestCase):
    def test_exit_status_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            pid = Path(directory) / "pid"
            log = Path(directory) / "log"
            self.assertEqual(run_target(pid, log, [sys.executable, "-c", "print('ok')"]), 0)
            self.assertEqual(log.read_text().strip(), "ok")
            with self.assertRaises(FileExistsError):
                run_target(pid, log, [sys.executable, "-c", "raise Exception()"])
            self.assertEqual(log.read_text().strip(), "ok")

    def test_stopping_model_group_allows_supervisor_to_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            pid = Path(directory) / "pid"
            log = Path(directory) / "log"
            supervisor = subprocess.Popen([
                sys.executable, "-m", "src.nsys_target", str(pid), str(log),
                sys.executable, "-c", "import time; time.sleep(30)",
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            child_pid = None
            try:
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    if pid.exists() and pid.read_text().strip():
                        child_pid = int(pid.read_text())
                        break
                    time.sleep(0.02)
                self.assertIsNotNone(child_pid)
                self.assertEqual(os.getpgid(child_pid), child_pid)
                self.assertNotEqual(os.getpgid(supervisor.pid), child_pid)
                os.killpg(child_pid, signal.SIGTERM)
                stdout, stderr = supervisor.communicate(timeout=5)
                self.assertEqual(supervisor.returncode, 0, stderr)
                self.assertIn("profile target exit status", stdout)
            finally:
                if supervisor.poll() is None:
                    supervisor.terminate()
                    supervisor.communicate(timeout=5)


if __name__ == "__main__":
    unittest.main()
