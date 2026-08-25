"""Analysis helpers for measured placement-DVFS perturbations."""

from __future__ import annotations

from typing import Any, Mapping


def dvfs_sensitivity(
    reference: Mapping[str, Any], perturbation: Mapping[str, Any], gpu: int,
) -> dict[str, float | int]:
    if not 0 <= gpu < 4:
        raise ValueError("gpu must be in [0, 3]")
    required = ("ttft_p99_ms", "tpot_p99_ms", "e2e_p99_ms",
                "total_gpu_energy_j", "throughput_requests_s", f"gpu{gpu}_energy_j")
    for key in required:
        if reference.get(key) is None or perturbation.get(key) is None:
            raise ValueError(f"missing sensitivity metric {key}")
    return {
        "gpu": gpu,
        "delta_ttft_p99_ms": float(perturbation["ttft_p99_ms"] - reference["ttft_p99_ms"]),
        "delta_tpot_p99_ms": float(perturbation["tpot_p99_ms"] - reference["tpot_p99_ms"]),
        "delta_e2e_p99_ms": float(perturbation["e2e_p99_ms"] - reference["e2e_p99_ms"]),
        "delta_total_energy_j": float(
            perturbation["total_gpu_energy_j"] - reference["total_gpu_energy_j"]
        ),
        "delta_gpu_energy_j": float(
            perturbation[f"gpu{gpu}_energy_j"] - reference[f"gpu{gpu}_energy_j"]
        ),
        "delta_throughput_requests_s": float(
            perturbation["throughput_requests_s"] - reference["throughput_requests_s"]
        ),
    }
