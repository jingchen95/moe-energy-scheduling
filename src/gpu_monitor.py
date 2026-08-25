"""Four-GPU NVML cumulative-energy boundaries and diagnostic sampling.

The fixed-period schedule and counter-first accounting are adapted from
``vLLM_test/paper/scripts/xpyd/{nvml_readonly,phase3b_energy}.py``. This module
is monolithic-serving specific only in assuming a caller-selected GPU list.
"""

from __future__ import annotations

import csv
import json
import math
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class GPUSample:
    timestamp_unix_s: float
    timestamp_monotonic_s: float
    gpu: int
    power_w: float
    energy_mj: int | None
    sm_util_percent: float
    memory_util_percent: float
    sm_clock_mhz: int
    memory_clock_mhz: int
    temperature_c: int
    power_limit_w: float


CSV_FIELDS = tuple(GPUSample.__dataclass_fields__)


def energy_delta_j(start_mj: int, end_mj: int) -> float:
    if start_mj < 0 or end_mj < 0:
        raise ValueError("energy counters must be non-negative")
    if end_mj < start_mj:
        raise ValueError("NVML total-energy counter decreased")
    return (end_mj - start_mj) / 1000.0


def integrate_power_j(samples: Sequence[GPUSample]) -> float:
    ordered = sorted(samples, key=lambda sample: sample.timestamp_monotonic_s)
    return sum(
        (right.timestamp_monotonic_s - left.timestamp_monotonic_s)
        * (left.power_w + right.power_w) / 2.0
        for left, right in zip(ordered, ordered[1:])
    )


class NVMLReader:
    def __init__(self, gpu_indices: Iterable[int] = range(4)):
        try:
            import pynvml
        except ImportError as exc:
            raise RuntimeError("nvidia-ml-py/pynvml is required for GPU monitoring") from exc
        self.nvml = pynvml
        pynvml.nvmlInit()
        self.gpu_indices = tuple(gpu_indices)
        count = pynvml.nvmlDeviceGetCount()
        if any(gpu < 0 or gpu >= count for gpu in self.gpu_indices):
            raise ValueError(f"requested GPU outside NVML device range 0..{count - 1}")
        self.handles = {gpu: pynvml.nvmlDeviceGetHandleByIndex(gpu)
                        for gpu in self.gpu_indices}
        self.energy_supported = {
            gpu: self._energy_supported(handle) for gpu, handle in self.handles.items()
        }

    def _energy_supported(self, handle: Any) -> bool:
        try:
            self.nvml.nvmlDeviceGetTotalEnergyConsumption(handle)
            return True
        except self.nvml.NVMLError_NotSupported:
            return False

    def close(self) -> None:
        self.nvml.nvmlShutdown()

    def __enter__(self) -> "NVMLReader":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def sample_all(self) -> list[GPUSample]:
        timestamp_unix_s = time.time()
        timestamp_monotonic_s = time.monotonic()
        rows: list[GPUSample] = []
        for gpu in self.gpu_indices:
            handle = self.handles[gpu]
            utilization = self.nvml.nvmlDeviceGetUtilizationRates(handle)
            energy = (self.nvml.nvmlDeviceGetTotalEnergyConsumption(handle)
                      if self.energy_supported[gpu] else None)
            rows.append(GPUSample(
                timestamp_unix_s=timestamp_unix_s,
                timestamp_monotonic_s=timestamp_monotonic_s,
                gpu=gpu,
                power_w=self.nvml.nvmlDeviceGetPowerUsage(handle) / 1000.0,
                energy_mj=energy,
                sm_util_percent=float(utilization.gpu),
                memory_util_percent=float(utilization.memory),
                sm_clock_mhz=int(self.nvml.nvmlDeviceGetClockInfo(
                    handle, self.nvml.NVML_CLOCK_SM)),
                memory_clock_mhz=int(self.nvml.nvmlDeviceGetClockInfo(
                    handle, self.nvml.NVML_CLOCK_MEM)),
                temperature_c=int(self.nvml.nvmlDeviceGetTemperature(
                    handle, self.nvml.NVML_TEMPERATURE_GPU)),
                power_limit_w=self.nvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000.0,
            ))
        return rows


class FixedPeriodMonitor:
    """Sample against deadlines derived from one monotonic epoch."""

    def __init__(self, reader: NVMLReader, output: Path, interval_s: float = 0.2):
        if interval_s <= 0:
            raise ValueError("sampling interval must be positive")
        self.reader = reader
        self.output = output
        self.interval_s = interval_s
        self.samples: list[GPUSample] = []
        self.missed_deadlines = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("monitor already started")
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(target=self._run, name="gpu-monitor", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            epoch = time.monotonic()
            slot = 0
            while not self._stop.is_set():
                deadline = epoch + slot * self.interval_s
                wait = deadline - time.monotonic()
                if wait > 0 and self._stop.wait(wait):
                    break
                now = time.monotonic()
                if now - deadline >= self.interval_s:
                    skipped = int((now - deadline) // self.interval_s)
                    self.missed_deadlines += skipped
                    slot += skipped
                self.samples.extend(self.reader.sample_all())
                slot += 1
        except BaseException as exc:
            self._error = exc

    def stop(self) -> list[GPUSample]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(5.0, self.interval_s * 4))
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("GPU monitor thread did not stop")
        self._write()
        if self._error is not None:
            raise RuntimeError("GPU monitor failed") from self._error
        return self.samples

    def _write(self) -> None:
        with self.output.open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(asdict(sample) for sample in self.samples)


def summarize_energy_window(
    start: Sequence[GPUSample],
    end: Sequence[GPUSample],
    diagnostic_samples: Sequence[GPUSample],
) -> dict[str, Any]:
    starts = {sample.gpu: sample for sample in start}
    ends = {sample.gpu: sample for sample in end}
    if starts.keys() != ends.keys():
        raise ValueError("start and end GPU sets differ")
    result: dict[str, Any] = {
        "energy_source": "nvml_total_energy_counter",
        "measurement_start_unix_s": min(row.timestamp_unix_s for row in start),
        "measurement_end_unix_s": max(row.timestamp_unix_s for row in end),
        "measurement_start_monotonic_s": min(row.timestamp_monotonic_s for row in start),
        "measurement_end_monotonic_s": max(row.timestamp_monotonic_s for row in end),
    }
    total = 0.0
    lower = min(row.timestamp_monotonic_s for row in start)
    upper = max(row.timestamp_monotonic_s for row in end)
    for gpu in sorted(starts):
        rows = [sample for sample in diagnostic_samples
                if sample.gpu == gpu and lower <= sample.timestamp_monotonic_s <= upper]
        start_energy = starts[gpu].energy_mj
        end_energy = ends[gpu].energy_mj
        if start_energy is not None and end_energy is not None:
            joules = energy_delta_j(start_energy, end_energy)
        else:
            result["energy_source"] = "trapezoidal_power_fallback"
            boundary_rows = [starts[gpu], *rows, ends[gpu]]
            joules = integrate_power_j(boundary_rows)
        total += joules
        result[f"gpu{gpu}_energy_j"] = joules
        result[f"gpu{gpu}_avg_power_w"] = mean(row.power_w for row in rows) if rows else None
        result[f"gpu{gpu}_peak_power_w"] = max((row.power_w for row in rows), default=None)
        result[f"gpu{gpu}_avg_sm_util"] = mean(row.sm_util_percent for row in rows) if rows else None
        result[f"gpu{gpu}_avg_memory_util"] = mean(row.memory_util_percent for row in rows) if rows else None
        result[f"gpu{gpu}_avg_sm_clock"] = mean(row.sm_clock_mhz for row in rows) if rows else None
    result["total_gpu_energy_j"] = total
    return result


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration-s", type=float, default=5.0)
    parser.add_argument("--interval-s", type=float, default=0.2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with NVMLReader() as reader:
        start = reader.sample_all()
        monitor = FixedPeriodMonitor(reader, args.output, args.interval_s)
        monitor.start()
        time.sleep(args.duration_s)
        end = reader.sample_all()
        samples = monitor.stop()
    print(json.dumps(summarize_energy_window(start, end, samples), indent=2))


if __name__ == "__main__":
    main()
