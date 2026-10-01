"""Graphics-only NVIDIA GPU clock discovery and control.

The actuation/readback pattern is adapted from vLLM_test's
``paper/scripts/xpyd/phase3d_control.py``. Memory clocks are never changed.
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import re
import subprocess
import time
from dataclasses import asdict, dataclass
from typing import Iterable, Sequence


class GPUControlError(RuntimeError):
    pass


CLOCK_READBACK_TOLERANCE_MHZ = 15


@dataclass(frozen=True)
class GPUClockState:
    index: int
    name: str
    uuid: str
    pci_bus_id: str
    graphics_clock_mhz: int


def _command_prefix() -> list[str]:
    sudo = os.environ.get("SUDO", "sudo").strip()
    return ([sudo] if sudo else []) + [os.environ.get("NVIDIA_SMI", "nvidia-smi")]


def _run(arguments: Sequence[str], *, privileged: bool = False) -> str:
    command = (_command_prefix() if privileged else
               [os.environ.get("NVIDIA_SMI", "nvidia-smi")]) + list(arguments)
    try:
        result = subprocess.run(command, check=True, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "stderr", None) or str(exc)
        raise GPUControlError(f"command failed: {' '.join(command)}: {detail.strip()}") from exc
    return result.stdout


def supported_graphics_clocks(gpu: int) -> list[int]:
    output = _run([
        "-i", str(gpu), "--query-supported-clocks=gr",
        "--format=csv,noheader,nounits",
    ])
    clocks = sorted({int(line.strip()) for line in output.splitlines()
                     if re.fullmatch(r"\s*\d+\s*", line)})
    if not clocks:
        raise GPUControlError(f"GPU {gpu} reported no supported graphics clocks")
    return clocks


def read_clock(gpu: int) -> GPUClockState:
    output = _run([
        "-i", str(gpu),
        "--query-gpu=index,name,uuid,pci.bus_id,clocks.current.graphics",
        "--format=csv,noheader,nounits",
    ]).strip()
    fields = [field.strip() for field in output.split(",")]
    if len(fields) != 5 or int(fields[0]) != gpu:
        raise GPUControlError(f"malformed GPU {gpu} clock readback: {output!r}")
    return GPUClockState(gpu, fields[1], fields[2], fields[3], int(fields[4]))


def read_clocks(gpus: Iterable[int] = range(4)) -> list[GPUClockState]:
    return [read_clock(gpu) for gpu in gpus]


def set_graphics_clock(
    gpu: int,
    clock_mhz: int,
    *,
    verify_timeout_s: float = 2.0,
    verify_tolerance_mhz: int = CLOCK_READBACK_TOLERANCE_MHZ,
) -> GPUClockState:
    if verify_tolerance_mhz < 0:
        raise ValueError("clock verification tolerance must be non-negative")
    if clock_mhz not in supported_graphics_clocks(gpu):
        raise GPUControlError(f"GPU {gpu}: {clock_mhz} MHz is not hardware-supported")
    _run(["-i", str(gpu), "-lgc", f"{clock_mhz},{clock_mhz}"], privileged=True)
    deadline = time.monotonic() + verify_timeout_s
    state = read_clock(gpu)
    while (
        abs(state.graphics_clock_mhz - clock_mhz) > verify_tolerance_mhz
        and time.monotonic() < deadline
    ):
        time.sleep(0.2)
        state = read_clock(gpu)
    if abs(state.graphics_clock_mhz - clock_mhz) > verify_tolerance_mhz:
        reset_graphics_clocks([gpu], verify=False)
        raise GPUControlError(
            f"GPU {gpu}: requested {clock_mhz} MHz but read "
            f"{state.graphics_clock_mhz} MHz (tolerance {verify_tolerance_mhz} MHz)"
        )
    return state


def set_graphics_clocks(clocks_mhz: Sequence[int]) -> list[GPUClockState]:
    if len(clocks_mhz) != 4:
        raise ValueError("exactly four per-GPU clocks are required")
    changed: list[int] = []
    try:
        states = []
        for gpu, clock in enumerate(clocks_mhz):
            states.append(set_graphics_clock(gpu, int(clock)))
            changed.append(gpu)
        return states
    except BaseException:
        if changed:
            reset_graphics_clocks(changed, verify=False)
        raise


def reset_graphics_clocks(
    gpus: Iterable[int] = range(4), *, verify: bool = True,
) -> list[GPUClockState]:
    gpu_list = list(gpus)
    errors: list[str] = []
    for gpu in gpu_list:
        try:
            _run(["-i", str(gpu), "-rgc"], privileged=True)
        except GPUControlError as exc:
            errors.append(str(exc))
    if errors:
        raise GPUControlError("; ".join(errors))
    if not verify:
        return []
    # A clock can remain at the former P-state briefly after its lock is removed.
    time.sleep(0.5)
    return read_clocks(gpu_list)


class GraphicsClockSession:
    """Apply four clock locks and reset them on every managed exit path."""

    def __init__(self, clocks_mhz: Sequence[int]):
        self.clocks_mhz = list(clocks_mhz)
        self._active = False

    def __enter__(self) -> list[GPUClockState]:
        states = set_graphics_clocks(self.clocks_mhz)
        self._active = True
        atexit.register(self.reset)
        return states

    def reset(self) -> None:
        if self._active:
            reset_graphics_clocks()
            self._active = False

    def __exit__(self, *_: object) -> None:
        self.reset()


def _print(value: object) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    list_parser = subparsers.add_parser("list")
    list_parser.add_argument("--gpus", default="0,1,2,3")
    set_parser = subparsers.add_parser("set")
    set_parser.add_argument("--gpu", type=int, required=True)
    set_parser.add_argument("--clock", type=int, required=True)
    all_parser = subparsers.add_parser("set-all")
    all_parser.add_argument("--clocks", required=True)
    max_parser = subparsers.add_parser("set-max")
    max_parser.add_argument("--gpus", default="0,1,2,3")
    reset_parser = subparsers.add_parser("reset")
    reset_parser.add_argument("--gpus", default="0,1,2,3")
    args = parser.parse_args()

    if args.command == "list":
        gpus = [int(value) for value in args.gpus.split(",")]
        _print([{"state": asdict(read_clock(gpu)),
                 "supported_graphics_mhz": supported_graphics_clocks(gpu)} for gpu in gpus])
    elif args.command == "set":
        _print(asdict(set_graphics_clock(args.gpu, args.clock)))
    elif args.command == "set-all":
        values = [int(value) for value in args.clocks.split(",")]
        if len(values) == 1:
            values *= 4
        _print([asdict(state) for state in set_graphics_clocks(values)])
    elif args.command == "set-max":
        gpus = [int(value) for value in args.gpus.split(",")]
        if gpus != list(range(4)):
            raise ValueError("set-max currently requires GPUs 0,1,2,3")
        values = [max(supported_graphics_clocks(gpu)) for gpu in gpus]
        _print([asdict(state) for state in set_graphics_clocks(values)])
    else:
        gpus = [int(value) for value in args.gpus.split(",")]
        _print([asdict(state) for state in reset_graphics_clocks(gpus)])


if __name__ == "__main__":
    main()
