import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

from src.torch_profile_plots import category, load_data, main, overlaps, render


class ProfilePlotTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "profile").mkdir()
        (self.root / "benchmark").mkdir()
        (self.root / "status.txt").write_text("PASS\n")
        (self.root / "git-commit.txt").write_text("test-revision\n")
        workers = []
        for device in range(4):
            events = [{
                "cat": "kernel", "ph": "X", "name": name,
                "ts": 100 + index * 40, "dur": duration,
                "args": {"device": device, "stream": 7, "graph node id": 1},
            } for index, (name, duration) in enumerate([
                ("fused_moe_kernel", 40), ("ncclDevKernel_AllGather", 20),
                ("vllm::moe::topkGating<int>", 3), ("unknown_kernel", 37),
            ])]
            name = f"worker-{device}.pt.trace.json"
            (self.root / "profile" / name).write_text(json.dumps({
                "baseTimeNanoseconds": 1000, "traceEvents": events,
            }))
            workers.append({"device": device, "path": f"/old/location/{name}",
                            "kernel_count": 4})
        (self.root / "moe-kernel-share.json").write_text(json.dumps({
            "status": "ok_heuristic_attribution", "run_scope": "test",
            "per_device": workers, "total_cuda_kernel_time": 400,
            "classified_moe_kernel_time": 240,
        }))
        rows = [{"status": "ok", "request_id": f"request-{i}",
                 "actual_send_s": i * 2, "ttft_ms": 40, "tpot_ms": 10,
                 "e2e_ms": 1310, "output_tokens": 128} for i in range(3)]
        (self.root / "benchmark/requests.jsonl").write_text(
            "\n".join(json.dumps(row) for row in rows)
        )

    def test_topk_is_separate_and_unknown_is_not_assumed_moe(self):
        self.assertEqual(category("vllm::moe::topkGating<int>"), "routing")
        self.assertEqual(category("fused_moe_kernel"), "compute")
        self.assertEqual(category("ncclDevKernel_ReduceScatter"), "communication")
        self.assertEqual(category("generic_gemm"), "other")

    def test_window_clipping_includes_crossing_events_not_edge_contacts(self):
        self.assertTrue(overlaps(9, 2, 10, 20))
        self.assertTrue(overlaps(19, 2, 10, 20))
        self.assertFalse(overlaps(9, 1, 10, 20))
        self.assertFalse(overlaps(20, 1, 10, 20))

    def test_raw_totals_extra_routing_and_portable_paths(self):
        data = load_data(self.root, 2)
        self.assertAlmostEqual(sum(data["percent_by_category"].values()), 100)
        self.assertEqual(data["original_moe_percent"], 60)
        self.assertEqual(data["moe_percent_with_topk"], 63)
        self.assertEqual(len(data["window_kernels"]), 16)
        self.assertEqual(data["window_origin_trace_us"], 100)

    def test_failed_run_and_mismatched_clock_rejected(self):
        (self.root / "status.txt").write_text("FAIL")
        with self.assertRaisesRegex(ValueError, "PASS"):
            load_data(self.root, 2)
        (self.root / "status.txt").write_text("PASS")
        path = self.root / "profile/worker-1.pt.trace.json"
        document = json.loads(path.read_text())
        document["baseTimeNanoseconds"] = 2000
        path.write_text(json.dumps(document))
        with self.assertRaisesRegex(ValueError, "clock bases differ"):
            load_data(self.root, 2)

    def test_graph_launch_correlation_without_graph_node_metadata(self):
        path = self.root / "profile/worker-0.pt.trace.json"
        document = json.loads(path.read_text())
        for event in document["traceEvents"]:
            event["args"].pop("graph node id")
            event["args"]["correlation"] = 42
        document["traceEvents"].append({
            "cat": "cuda_runtime", "ph": "X", "name": "cudaGraphLaunch",
            "ts": 90, "dur": 200, "args": {"correlation": 42},
        })
        path.write_text(json.dumps(document))
        data = load_data(self.root, 2)
        self.assertEqual(data["window_origin_trace_us"], 100)
        self.assertEqual(data["total_kernel_time_us"], 400)

    def test_no_overwrite(self):
        with patch("sys.argv", ["plots", str(self.root), str(self.root)]):
            with self.assertRaises(FileExistsError):
                main()

    @unittest.skipUnless(importlib.util.find_spec("PIL") and Path("/plot-fonts").is_dir(),
                         "requires pinned SIF with host fonts bound at /plot-fonts")
    def test_render_png_svg_and_self_contained_html(self):
        from PIL import Image

        output = self.root / "plots"
        output.mkdir()
        render(load_data(self.root, 2), output, Path("/plot-fonts"))
        for name in ("overview.png", "kernel-timeline.png"):
            with Image.open(output / name) as image:
                self.assertEqual(image.width, 1800)
                image.verify()
        root = ET.parse(output / "kernel-timeline.svg").getroot()
        self.assertEqual(len(root.findall(".//{http://www.w3.org/2000/svg}title")), 16)
        page = (output / "report.html").read_text()
        self.assertIn("data:image/png;base64,", page)
        self.assertIn("topkGating&lt;int&gt;", page)
        self.assertNotIn("<script", page)

    @unittest.skipUnless(importlib.util.find_spec("PIL") and Path("/plot-fonts").is_dir(),
                         "requires pinned SIF with host fonts bound at /plot-fonts")
    def test_cli_records_source_hashes_without_external_git(self):
        output = self.root / "cli-output"
        with patch("sys.argv", ["plots", str(self.root), str(output),
                                "--source-revision", "fixture-revision"]):
            main()
        data = json.loads((output / "plot-data.json").read_text())
        self.assertEqual(data["plot_source_revision"], "fixture-revision")
        self.assertEqual(len(data["plot_source_files_sha256"]["torch_profile_plots.py"]), 64)
        self.assertTrue((output / "report.html").is_file())


if __name__ == "__main__":
    unittest.main()
