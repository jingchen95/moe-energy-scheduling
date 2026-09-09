import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from src.metrics import active_request_time_ms, latency_metrics
from src.moe_timing import fused_moe_time_share_percent, load_timing_files
from src.nsys_moe_share import estimate_moe_share, load_kernel_summary
from src.workload import load_trace, scheduled_monotonic_times


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class MinimalBenchmarkTests(unittest.TestCase):
    def test_fixed_trace(self) -> None:
        trace = REPOSITORY_ROOT / "data/traces/gsm8k-50-r0.5.jsonl"
        self.assertEqual(
            hashlib.sha256(trace.read_bytes()).hexdigest(),
            "721638a6250b929de05c2c355c8bc9b20258db7b28fedb94acb5572136571cbc",
        )
        requests = load_trace(trace)
        self.assertEqual(len(requests), 50)
        self.assertEqual(
            [request.arrival_offset_s for request in requests],
            [float(2 * index) for index in range(50)],
        )
        self.assertEqual(
            scheduled_monotonic_times(requests, 10.0),
            [10.0 + 2.0 * index for index in range(50)],
        )

    def test_active_request_time_is_interval_union(self) -> None:
        ttft_ms, tpot_ms, e2e_ms = latency_metrics(10.0, 10.2, 11.0, 5)
        self.assertAlmostEqual(ttft_ms, 200.0)
        self.assertAlmostEqual(tpot_ms, 200.0)
        self.assertAlmostEqual(e2e_ms, 1000.0)

        rows = [
            {"status": "ok", "actual_send_s": 2.0, "e2e_ms": 500.0},
            {"status": "ok", "actual_send_s": 0.5, "e2e_ms": 1000.0},
            {"status": "ok", "actual_send_s": 0.0, "e2e_ms": 1000.0},
        ]
        self.assertAlmostEqual(active_request_time_ms(rows), 2000.0)

    def test_moe_share_uses_max_rank_not_rank_sum(self) -> None:
        documents = [
            {"schema_version": 1, "rank": 0, "records": [
                {"elapsed_ms": 2.0}, {"elapsed_ms": 4.0}
            ]},
            {"schema_version": 1, "rank": 1, "records": [
                {"elapsed_ms": 5.0}
            ]},
        ]
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for rank, document in enumerate(documents):
                path = Path(directory) / f"moe_timing_rank{rank}.json"
                path.write_text(json.dumps(document), encoding="utf-8")
                paths.append(path)
            timing = load_timing_files(paths, expected_ranks=2)

        self.assertEqual(timing["max_rank_total_ms"], 6.0)
        self.assertEqual(timing["sum_rank_total_ms"], 11.0)
        self.assertAlmostEqual(fused_moe_time_share_percent(timing, 20.0), 30.0)

    def test_nsys_moe_share_is_conservative_kernel_fraction(self) -> None:
        csv_text = """notice before CSV
\"Time (%)\",\"Total Time (ns)\",\"Instances\",\"Name\"
40.0,400,2,\"fused_moe_kernel\"
10.0,100,2,\"ncclDevKernel_AllGather\"
5.0,50,2,\"ncclDevKernel_ReduceScatter\"
45.0,450,2,\"flash_attention_kernel\"
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "kernels.csv"
            path.write_text(csv_text, encoding="utf-8")
            rows = load_kernel_summary(path)
        result = estimate_moe_share([rows])
        self.assertEqual(result["total_cuda_kernel_time"], 1000.0)
        self.assertEqual(result["moe_compute_kernel_time"], 400.0)
        self.assertEqual(
            result["moe_allgather_reducescatter_kernel_time"], 150.0
        )
        self.assertAlmostEqual(
            result["conservative_moe_cuda_kernel_time_share_percent"], 55.0
        )


if __name__ == "__main__":
    unittest.main()
