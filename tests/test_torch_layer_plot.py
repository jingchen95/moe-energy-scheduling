import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

from src.torch_layer_plot import FIRST_NORM, POST_NORM, PRE_NORM, analyze_document, main, render


def document_fixture():
    names = [PRE_NORM, "ampere_bf16_gemm", "triton_qk_norm", "triton_rope",
             "reshape_and_cache_flash_kernel", "flash_fwd_splitkv_kernel",
             "flash_fwd_splitkv_combine_kernel", "memcpy32_post", "internal::gemvx::kernel",
             POST_NORM, "internal::gemvx::kernel", "ncclDevKernel_AllGather",
             "vllm::moe::topkGating", "fill", "moe_align", "count_experts", "copy", "index",
             "fused_moe_kernel", "act_and_mul_kernel", "fused_moe_kernel", "reduce", "copy",
             "ncclDevKernel_ReduceScatter"]
    events = [{"cat": "cuda_runtime", "ph": "X", "name": "cudaGraphLaunch",
               "ts": 50, "dur": 10000, "args": {"correlation": 42}}]
    for layer in range(2):
        for index, name in enumerate(names):
            if layer == index == 0:
                name = FIRST_NORM
            events.append({"cat": "kernel", "ph": "X", "name": name,
                           "ts": 100 + (layer * len(names) + index) * 10, "dur": 5,
                           "args": {"device": 0, "stream": 7, "correlation": 42}})
    events.append({"cat": "kernel", "ph": "X", "name": PRE_NORM, "ts": 580, "dur": 5,
                   "args": {"device": 0, "stream": 7, "correlation": 42}})
    return {"traceEvents": events, "baseTimeNanoseconds": 1000}


class LayerPlotTests(unittest.TestCase):
    def test_one_complete_layer_and_nonoverlapping_categories(self):
        data = analyze_document(document_fixture(), 2, 1, 0)
        self.assertEqual(len(data["events"]), 24)
        self.assertEqual(data["total_kernel_us"], 120)
        self.assertEqual(data["elapsed_us"], 235)
        self.assertEqual(data["kernel_time_us"],
                         {"attention": 40, "moe": 60, "communication": 10, "norm": 10})
        self.assertAlmostEqual(sum(data["kernel_share_percent"].values()), 100)
        self.assertEqual(data["events"][0]["start_us"], 0)

    def test_missing_projection_or_norm_or_expert_is_rejected(self):
        for name in ("ampere_bf16_gemm", POST_NORM, "fused_moe_kernel"):
            doc = document_fixture()
            index = next(i for i, e in enumerate(doc["traceEvents"]) if e["name"] == name)
            doc["traceEvents"].pop(index)
            with self.subTest(name=name), self.assertRaises(ValueError):
                analyze_document(doc, 2, 1, 0)

    def test_partial_graph_and_missing_replay_rejected(self):
        doc = document_fixture()
        doc["traceEvents"] = [e for e in doc["traceEvents"] if e["name"] != "ncclDevKernel_ReduceScatter"]
        with self.assertRaisesRegex(ValueError, "only 0"):
            analyze_document(doc, 2, 1, 0)
        with self.assertRaisesRegex(ValueError, "only 1"):
            analyze_document(document_fixture(), 2, 1, 1)

    def test_embedding_layer_not_claimed_as_clean_layer(self):
        with self.assertRaisesRegex(ValueError, "embedding"):
            analyze_document(document_fixture(), 2, 0, 0)

    def test_cpu_events_are_not_summed(self):
        doc = document_fixture()
        doc["traceEvents"].append({"cat": "cpu_op", "name": "linear", "ph": "X",
                                   "ts": 100, "dur": 90000, "args": {"correlation": 42}})
        self.assertEqual(analyze_document(doc, 2, 1, 0)["total_kernel_us"], 120)

    def test_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "status.txt").write_text("PASS")
            with patch("sys.argv", ["layer_plot", str(root), str(root)]):
                with self.assertRaises(FileExistsError):
                    main()

    @unittest.skipUnless(importlib.util.find_spec("PIL") and Path("/plot-fonts").is_dir(),
                         "requires pinned SIF and font bind")
    def test_actual_png_and_hover_svg(self):
        from PIL import Image
        data = analyze_document(document_fixture(), 2, 1, 0)
        data["source_run"] = "synthetic-unit-test"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            render(data, root, Path("/plot-fonts"))
            with Image.open(root / "layer-distribution.png") as image:
                self.assertEqual(image.size, (1800, 1240))
                image.verify()
            svg = ET.parse(root / "layer-timeline.svg").getroot()
            self.assertEqual(len(svg.findall(".//{http://www.w3.org/2000/svg}title")), 24)
            self.assertIn("data:image/png;base64,", (root / "report.html").read_text())


if __name__ == "__main__":
    unittest.main()
