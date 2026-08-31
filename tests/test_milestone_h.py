import json

from tools.analyze_milestone_h import select_oracles


def _row(placement, frequency, tpot, energy, ttft=100.0):
    return {
        "placement": placement,
        "placement_name": placement,
        "frequency_tuple": json.dumps(list(frequency)),
        "ttft_p99_ms": ttft,
        "tpot_p99_ms": tpot,
        "total_gpu_energy_j": energy,
        "failed_requests": 0,
        "run_dir": f"/tmp/{placement}-{frequency[0]}",
    }


def test_ground_truth_oracles_use_measured_latency_and_energy_only():
    rows = []
    high_tpot = {"P0": 10.0, "P1": 9.0, "P2": 7.0, "P3": 8.0, "P4": 11.0}
    for placement in high_tpot:
        rows.extend([
            _row(placement, (2520, 2520, 2520, 2520), high_tpot[placement],
                  {"P0": 100.0, "P1": 95.0, "P2": 90.0, "P3": 92.0, "P4": 93.0}[placement]),
            _row(placement, (1635, 1635, 1635, 1635), 12.0,
                  {"P0": 80.0, "P1": 75.0, "P2": 70.0, "P3": 65.0, "P4": 60.0}[placement]),
            _row(placement, (1005, 1005, 1005, 1005), 14.0,
                  {"P0": 70.0, "P1": 68.0, "P2": 66.0, "P3": 55.0, "P4": 50.0}[placement]),
        ])

    sequential, summary = select_oracles(rows)

    assert sequential["placement"] == "P2"
    assert [row["sequential_placement"] for row in summary] == ["P2"] * 3
    assert [row["joint_placement"] for row in summary] == ["P4"] * 3
    assert all(row["joint_strictly_lower_energy"] for row in summary)
