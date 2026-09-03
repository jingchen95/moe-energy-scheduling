import json
from pathlib import Path
import tempfile
import unittest

from src.config import build_vllm_command, load_yaml
from src.metrics import active_request_time_ms, latency_metrics
from src.moe_timing import fused_moe_time_share_percent, load_timing_files
from src.workload import TraceRequest, scheduled_monotonic_times


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class MinimalBenchmarkTests(unittest.TestCase):
    def test_vllm_command_uses_tp1_dp4_ep(self) -> None:
        config = load_yaml(REPOSITORY_ROOT / "configs/model.yaml")
        command = build_vllm_command(config, ["--enforce-eager"])

        self.assertEqual(command[command.index("--tensor-parallel-size") + 1], "1")
        self.assertEqual(command[command.index("--data-parallel-size") + 1], "4")
        self.assertEqual(
            command[command.index("--all2all-backend") + 1],
            "allgather_reducescatter",
        )
        self.assertIn("--enable-expert-parallel", command)
        self.assertNotIn("--enable-eplb", command)
        self.assertEqual(
            command[command.index("--expert-placement-strategy") + 1], "linear"
        )
        self.assertEqual(command[-1], "--enforce-eager")

    def test_trace_schedule_uses_one_fixed_epoch(self) -> None:
        requests = [
            TraceRequest("a", "math", "a", 1, 1, 0.0),
            TraceRequest("b", "math", "b", 1, 1, 0.125),
            TraceRequest("c", "math", "c", 1, 1, 2.0),
        ]
        self.assertEqual(
            scheduled_monotonic_times(requests, 10.0),
            [10.0, 10.125, 12.0],
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


if __name__ == "__main__":
    unittest.main()
