import gzip
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

from src.torch_moe_share import analyze_run, load_gpu_trace, summarize_traces
from src.torch_profile import PROFILER_CONFIG


def trace_document(device):
    events = [
        {"cat": "kernel", "ph": "X", "name": name, "ts": 10, "dur": duration,
         "args": {"device": device, "correlation": 7}}
        for name, duration in [
            ("fused_moe_kernel", 40), ("ncclDevKernel_AllGather", 10),
            ("ncclDevKernel_ReduceScatter", 5), ("flash_attention", 45),
        ]
    ]
    events += [
        {"cat": "cuda_runtime", "ph": "X", "name": "cudaGraphLaunch",
         "dur": 1000, "args": {"correlation": 7}},
        {"cat": "cpu_op", "ph": "X", "name": "fused_moe", "dur": 10000},
        {"cat": "gpu_memcpy", "ph": "X", "name": "Memcpy", "dur": 10000},
    ]
    # This only controls the viewer; it does not change ts/dur units.
    return {"displayTimeUnit": "ms", "traceEvents": events}


def write_trace(directory, device, document=None):
    path = directory / f"worker-{device}.pt.trace.json.gz"
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        json.dump(trace_document(device) if document is None else document, stream)
    return path


class TorchTraceTests(unittest.TestCase):
    def test_gpu_only_weighted_fraction_and_microseconds(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = [write_trace(Path(directory), device) for device in range(4)]
            result = summarize_traces(paths)
        self.assertEqual(result["total_cuda_kernel_time"], 400)
        self.assertEqual(result["total_cuda_kernel_time_ms"], 0.4)
        self.assertEqual(result["time_unit"], "us")
        self.assertAlmostEqual(result["conservative_moe_cuda_kernel_time_share_percent"], 55)
        self.assertEqual([x["graph_kernel_count"] for x in result["per_device"]], [4] * 4)

    def test_missing_duplicate_trace_or_duplicate_gpu_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = [write_trace(Path(directory), device) for device in range(4)]
            for bad_paths in (paths[:3], paths[:3] + [paths[0]]):
                with self.assertRaises(ValueError):
                    summarize_traces(bad_paths)
            write_trace(Path(directory), 3, trace_document(0))
            with self.assertRaisesRegex(ValueError, "expected CUDA devices"):
                summarize_traces(paths)

    def test_graph_node_metadata_is_also_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            doc = trace_document(0)
            doc["traceEvents"] = doc["traceEvents"][:4]
            for event in doc["traceEvents"]:
                event["args"]["graph id"] = 0
                event["args"]["graph node id"] = 123
            result = load_gpu_trace(write_trace(Path(directory), 0, doc))
            self.assertEqual(result["graph_kernel_count"], 4)

    def test_cpu_only_or_missing_graph_kernels_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            doc = trace_document(0)
            doc["traceEvents"] = doc["traceEvents"][4:]
            with self.assertRaisesRegex(ValueError, "expected one GPU"):
                load_gpu_trace(write_trace(Path(directory), 0, doc))
            doc = trace_document(0)
            doc["traceEvents"][4]["args"]["correlation"] = 99
            with self.assertRaisesRegex(ValueError, "no verifiable CUDA Graph"):
                load_gpu_trace(write_trace(Path(directory), 0, doc))

    def test_invalid_durations_and_truncated_json_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            for bad in (-1, float("nan"), float("inf"), True, "40"):
                with self.subTest(duration=bad):
                    doc = trace_document(0)
                    doc["traceEvents"][0]["dur"] = bad
                    with self.assertRaisesRegex(ValueError, "invalid kernel duration"):
                        load_gpu_trace(write_trace(Path(directory), 0, doc))
            path = Path(directory) / "broken.pt.trace.json"
            path.write_text('{"traceEvents": [')
            with self.assertRaises(json.JSONDecodeError):
                load_gpu_trace(path)

    def test_no_recognizable_moe_compute_is_not_reported_as_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for device in range(4):
                doc = trace_document(device)
                doc["traceEvents"][0]["name"] = "generic_gemm"
                paths.append(write_trace(Path(directory), device, doc))
            with self.assertRaisesRegex(ValueError, "no recognized MoE"):
                summarize_traces(paths)

    def test_failed_benchmark_or_stop_cannot_publish_percentage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            benchmark = root / "benchmark"
            benchmark.mkdir()
            (root / "profile").mkdir()
            for device in range(4):
                write_trace(root / "profile", device)
            controls = {action: {"http_status": 200} for action in ("start", "stop")}
            metrics = {"status": "ok", "completed_requests": 3, "failed_requests": 0}
            (benchmark / "metrics.json").write_text(json.dumps(metrics))
            (benchmark / "profile_control.json").write_text(json.dumps(controls))
            (benchmark / "profile_config.json").write_text(json.dumps({
                "profiler": "torch", "profile_request_ids": ["a", "b", "c"],
            }))
            (benchmark / "requests.jsonl").write_text("\n".join(
                json.dumps({"request_id": rid, "status": "ok"}) for rid in ("a", "b", "c")
            ))
            self.assertEqual(analyze_run(root)["profiler"], "torch")
            controls["stop"] = {"http_status": 500}
            (benchmark / "profile_control.json").write_text(json.dumps(controls))
            with self.assertRaisesRegex(ValueError, "stop did not complete"):
                analyze_run(root)
            metrics["status"] = "invalid_profile"
            (benchmark / "metrics.json").write_text(json.dumps(metrics))
            with self.assertRaisesRegex(ValueError, "benchmark incomplete/failed"):
                analyze_run(root)
            self.assertFalse((root / "moe-kernel-share.json").exists())

    def test_minimal_profiler_settings(self):
        self.assertEqual(PROFILER_CONFIG["profiler"], "torch")
        self.assertTrue(PROFILER_CONFIG["ignore_frontend"])
        for key in ("torch_profiler_with_stack", "torch_profiler_record_shapes",
                    "torch_profiler_with_memory", "torch_profiler_with_flops",
                    "torch_profiler_dump_cuda_time_total"):
            self.assertFalse(PROFILER_CONFIG[key])


@unittest.skipUnless(importlib.util.find_spec("aiohttp"), "run client tests in pinned SIF")
class ProfileClientFailureTests(unittest.IsolatedAsyncioTestCase):
    async def exercise_client(self, failures):
        from src import nsys_profile as client

        async def control(action):
            if action in failures:
                raise RuntimeError(f"{action}_profile failed: HTTP 500")
            return {"http_status": 200, "body": ""}

        rows = [{"status": "ok", "actual_send_s": float(i), "e2e_ms": 100,
                 "ttft_ms": 10, "tpot_ms": 5} for i in range(3)]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "benchmark"
            with patch.object(client, "warm_up", new=AsyncMock()), \
                    patch.object(client, "SETTLE_SECONDS", 0), \
                    patch.object(client, "replay_trace", new=AsyncMock(return_value=rows)), \
                    patch.object(client, "_profile_control", new=AsyncMock(side_effect=control)):
                if failures:
                    with self.assertRaisesRegex(RuntimeError, "invalid profile"):
                        await client.run_profile(output)
                else:
                    await client.run_profile(output)
            metrics = json.loads((output / "metrics.json").read_text())
            controls = json.loads((output / "profile_control.json").read_text())
            if "start" not in failures:
                self.assertEqual(len((output / "requests.jsonl").read_text().splitlines()), 3)
            return metrics, controls

    async def test_successful_client_still_records_all_responses(self):
        metrics, controls = await self.exercise_client(set())
        self.assertEqual(metrics["status"], "ok")
        self.assertEqual(controls["stop"]["http_status"], 200)

    async def test_stop_failure_preserves_responses_and_invalid_metrics(self):
        metrics, controls = await self.exercise_client({"stop"})
        self.assertEqual(metrics["status"], "invalid_profile")
        self.assertEqual(metrics["completed_requests"], 3)
        self.assertIn("HTTP 500", controls["stop"]["error"])

    async def test_start_failure_attempts_stop_and_records_both_errors(self):
        metrics, controls = await self.exercise_client({"start", "stop"})
        self.assertEqual(metrics["completed_requests"], 0)
        self.assertEqual(len(metrics["errors"]), 3)
        self.assertIn("HTTP 500", controls["start"]["error"])
        self.assertIn("HTTP 500", controls["stop"]["error"])


@unittest.skipUnless(importlib.util.find_spec("vllm"), "run runtime checks in pinned SIF")
class PinnedRuntimeTests(unittest.TestCase):
    def test_profiler_config_validates_without_changing_graph_hash(self):
        from vllm.config import ProfilerConfig
        import torch

        configured = ProfilerConfig(**PROFILER_CONFIG)
        self.assertEqual(configured.profiler, "torch")
        self.assertEqual(configured.compute_hash(), ProfilerConfig().compute_hash())
        self.assertTrue(torch.profiler.kineto_available())

    def test_real_cpu_trace_export_is_rejected_as_gpu_timing(self):
        import torch

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cpu.pt.trace.json"
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profiler:
                x = torch.ones(16)
                x = x + 1
            profiler.export_chrome_trace(str(path))
            self.assertGreater(path.stat().st_size, 0)
            with self.assertRaisesRegex(ValueError, "expected one GPU"):
                load_gpu_trace(path)


if __name__ == "__main__":
    unittest.main()
