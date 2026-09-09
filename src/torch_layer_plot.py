"""Infer one Qwen3-MoE decode layer from an existing Torch trace and plot it.

This is offline, order-based attribution, not module-labelled ground truth.
Validated against the pinned vLLM 0.15.1 model and the run's compiled graph.
"""

from __future__ import annotations

import argparse
import base64
from collections import defaultdict
import csv
import hashlib
import html
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET


PRE_NORM = "triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_1"
FIRST_NORM = "triton_red_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_0"
POST_NORM = "triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_0"
COLORS = {"attention": "#168AAD", "moe": "#7157CB",
          "communication": "#D97706", "norm": "#94A3B8"}
LABELS = {"attention": "Attention-related kernels", "moe": "MoE compute / routing / token prep",
          "communication": "MoE EP communication (AG + RS)", "norm": "RMSNorm / residual"}


def has_name(events: list[dict], part: str) -> int:
    return sum(part in e["name"] for e in events)


def is_projection(event: dict) -> bool:
    return "gemm" in event["name"].lower() or "gemvx::kernel" in event["name"]


def split_layers(kernels: list[dict], n_layers: int) -> list[list[dict]]:
    """Use the per-layer ReduceScatter endpoint, validating every full sequence."""
    groups, current = [], []
    for event in sorted(kernels, key=lambda e: e["ts"]):
        current.append(event)
        if "ncclDevKernel_ReduceScatter" in event["name"]:
            groups.append(current)
            current = []
    if len(groups) != n_layers:
        raise ValueError(f"expected {n_layers} layer endpoints, got {len(groups)}")
    if len(current) != 1 or current[0]["name"] != PRE_NORM:
        raise ValueError("expected only model-final RMSNorm after the last layer")
    for index, group in enumerate(groups):
        expected_pre = FIRST_NORM if index == 0 else PRE_NORM
        if group[0]["name"] != expected_pre:
            raise ValueError(f"layer {index}: missing input RMSNorm boundary")
        post = [i for i, e in enumerate(group) if e["name"] == POST_NORM]
        if len(post) != 1:
            raise ValueError(f"layer {index}: ambiguous post-attention RMSNorm")
        att, moe = group[1:post[0]], group[post[0] + 1:]
        if (has_name(att, "flash_fwd_splitkv_kernel") != 1
                or has_name(att, "flash_fwd_splitkv_combine_kernel") != 1
                or has_name(att, "reshape_and_cache_flash_kernel") != 1
                or has_name(moe, "fused_moe_kernel") != 2
                or has_name(moe, "topkGating") != 1
                or has_name(moe, "act_and_mul_kernel") != 1
                or has_name(moe, "ncclDevKernel_AllGather") != 1
                or has_name(moe, "ncclDevKernel_ReduceScatter") != 1
                or sum(is_projection(e) for e in att) != 2
                or not is_projection(moe[0])):
            raise ValueError(f"layer {index}: unsupported Attention/MoE kernel sequence")
        if has_name(att, "ncclDevKernel") or has_name(moe, "flash_fwd"):
            raise ValueError(f"layer {index}: unexpected cross-boundary operation")
    return groups


def analyze_document(document: dict, n_layers: int, layer_index: int, replay_index: int) -> dict:
    if not 0 < layer_index < n_layers:
        raise ValueError("choose an interior layer; layer 0 fuses embedding with RMSNorm")
    if replay_index < 0:
        raise ValueError("replay index must be nonnegative")
    calls = sorted((e for e in document["traceEvents"]
                    if e.get("cat") in {"cuda_runtime", "cuda_driver"}
                    and e.get("ph") == "X" and "GraphLaunch" in e.get("name", "")),
                   key=lambda e: e["ts"])
    by_correlation = defaultdict(list)
    for e in document["traceEvents"]:
        if e.get("cat") == "kernel" and e.get("ph") == "X":
            by_correlation[e.get("args", {}).get("correlation")].append(e)
    candidates = []
    for call in calls:
        group = by_correlation.get(call["args"]["correlation"], [])
        if (has_name(group, "ncclDevKernel_ReduceScatter") == n_layers
                and has_name(group, "flash_fwd_splitkv_combine_kernel") == n_layers):
            candidates.append((call, group))
    if replay_index >= len(candidates):
        raise ValueError(f"only {len(candidates)} full decode graph replays found")
    call, kernels = candidates[replay_index]
    groups = split_layers(kernels, n_layers)
    events = groups[layer_index]
    if {e["args"]["device"] for e in kernels} != {0}:
        raise ValueError("this diagnostic expects the active GPU 0 trace")
    for event in events:
        if not math.isfinite(event["dur"]) or event["dur"] < 0:
            raise ValueError("invalid kernel duration")
    origin = events[0]["ts"]
    post = next(i for i, e in enumerate(events) if e["name"] == POST_NORM)
    totals = dict.fromkeys(COLORS, 0.0)
    rows = []
    for index, event in enumerate(events):
        if index in (0, post):
            kind = "norm"
        elif index < post:
            kind = "attention"
        elif "ncclDevKernel" in event["name"]:
            kind = "communication"
        else:
            kind = "moe"
        totals[kind] += event["dur"]
        rows.append({"stage": kind, "start_us": event["ts"] - origin,
                     "duration_us": event["dur"], "stream": event["args"]["stream"],
                     "name": event["name"]})
    total = sum(totals.values())
    elapsed = max(e["ts"] + e["dur"] for e in events) - origin
    return {
        "layer_index_zero_based_inferred": layer_index, "n_layers": n_layers,
        "decode_graph_index_zero_based": replay_index, "decode_graph_candidates": len(candidates),
        "graph_correlation": call["args"]["correlation"],
        "graph_launch_trace_us": call["ts"], "layer_origin_trace_us": origin,
        "baseTimeNanoseconds": document["baseTimeNanoseconds"],
        "classification": "order_inferred_not_module_labelled",
        "kernel_time_us": totals, "kernel_share_percent": {k: v / total * 100 for k, v in totals.items()},
        "total_kernel_us": total, "elapsed_us": elapsed,
        "events": rows,
        "limitations": [
            "One GPU, one inferred layer, one decode graph replay; not a whole-model average.",
            "Layer identity follows 48 ordered, validated layer patterns, not trace module labels.",
            "Attention includes QKV/O projection, QK normalization, RoPE, KV write and attention kernels.",
            "MoE includes router projection, top-k, token preparation, expert GEMMs, activation and combine kernels.",
            "Fused residual/RMSNorm is separate; adjacent-layer residual work cannot be split within a fused kernel.",
            "NCCL duration includes communication/synchronization; it is not pure transfer time or pure compute.",
            "Kernel-duration sums exclude CPU, memcpy/memset activity records and gaps; overlaps may be summed.",
        ],
    }


def render(data: dict, output: Path, font_dir: Path) -> None:
    from PIL import Image, ImageDraw, ImageFont

    image = Image.new("RGB", (1800, 1240), "#F3F6FA")
    draw = ImageDraw.Draw(image)

    def text(x, y, value, size=23, bold=False):
        filename = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
        draw.text((x, y), value, fill="#172B46", font=ImageFont.truetype(str(font_dir / filename), size))

    layer = data["layer_index_zero_based_inferred"]
    text(55, 30, "Inside one Transformer layer | Attention vs MoE", 40, True)
    text(55, 92, f"GPU 0 | Layer {layer + 1}/{data['n_layers']} (index {layer}, inferred) | "
         f"Full decode graph replay #{data['decode_graph_index_zero_based'] + 1}", 25)
    text(55, 137, data["source_run"] + " | CUDA Graph kept enabled", 22)
    draw.rounded_rectangle((35, 190, 1765, 585), radius=18, fill="white")
    text(65, 207, "Distribution of summed GPU kernel durations in this layer execution", 27, True)
    for index, kind in enumerate(COLORS):
        y = 270 + index * 60
        text(65, y, LABELS[kind], 23)
        draw.rectangle((610, y, 1350, y + 31), fill="#EDF1F6")
        draw.rectangle((610, y, 610 + 740 * data["kernel_share_percent"][kind] / 100, y + 31), fill=COLORS[kind])
        text(1380, y - 3, f"{data['kernel_time_us'][kind]:.2f} us  |  {data['kernel_share_percent'][kind]:.1f}%", 25, True)
    text(65, 523, f"Kernel sum: {data['total_kernel_us']:.2f} us     |     "
         f"First-to-last kernel span: {data['elapsed_us']:.2f} us", 25, True)

    draw.rounded_rectangle((35, 610, 1765, 1085), radius=18, fill="white")
    text(65, 630, "Actual kernel timeline | separate rows for Attention, MoE, communication and normalization", 25, True)
    elapsed = data["elapsed_us"]
    width = 1330
    svg = ET.Element("svg", xmlns="http://www.w3.org/2000/svg", viewBox="0 0 1800 550")
    ET.SubElement(svg, "rect", width="1800", height="550", fill="#F3F6FA")

    def svg_text(x, y, label, size=22):
        node = ET.SubElement(svg, "text", x=str(x), y=str(y), fill="#172B46", style=f"font:{size}px sans-serif")
        node.text = label

    svg_text(55, 45, "One-layer kernel timeline - hover for exact kernel names / durations", 29)
    for tick in range(6):
        x = 385 + tick * width / 5
        draw.line((x, 700, x, 1000), fill="#E1E7EF")
        text(x - 8, 1020, f"{elapsed * tick / 5:.0f}", 18)
        ET.SubElement(svg, "line", x1=str(x), x2=str(x), y1="100", y2="405", stroke="#DEE5EF")
        svg_text(x - 8, 440, f"{elapsed * tick / 5:.0f}", 18)
    short = {"attention": "Attention", "moe": "MoE compute / prep", "communication": "EP communication", "norm": "Norm / residual"}
    for index, kind in enumerate(COLORS):
        y = 715 + index * 73
        sy = 110 + index * 73
        text(65, y + 5, short[kind], 25)
        svg_text(55, sy + 30, short[kind], 23)
        draw.rectangle((385, y, 1715, y + 42), fill="#F1F4F8")
        ET.SubElement(svg, "rect", x="385", y=str(sy), width=str(width), height="42", fill="white")
        for event in data["events"]:
            if event["stage"] != kind:
                continue
            x = 385 + width * event["start_us"] / elapsed
            w = width * event["duration_us"] / elapsed
            draw.rectangle((x, y, x + w, y + 42), fill=COLORS[kind], outline="white", width=1)
            rect = ET.SubElement(svg, "rect", x=str(x), y=str(sy), width=str(w), height="42", fill=COLORS[kind])
            ET.SubElement(rect, "title").text = (f"{event['name']}\nStream {event['stream']} | "
                f"Start {event['start_us']:.3f} us | Duration {event['duration_us']:.3f} us")
    text(700, 1055, "Time since layer start (us)", 20)
    svg_text(650, 490, "Time since layer start (us); whitespace is not attributed to compute.")
    text(55, 1110, "Approximate layer attribution from kernel order + pinned model/compiled source; NOT native module labels.", 22)
    text(55, 1150, "Attention includes projections / KV work; MoE includes routing / activation / combine. EP communication is separate.", 21)
    text(55, 1190, "Single decode example, not an average. Profiled Vera smoke run; warmup CUDA OOM warnings remain unresolved.", 21)
    with (output / "layer-distribution.png").open("xb") as stream:
        image.save(stream, format="PNG")
    svg_text_data = ET.tostring(svg, encoding="unicode")
    with (output / "layer-timeline.svg").open("x") as stream:
        stream.write(svg_text_data)
    png = base64.b64encode((output / "layer-distribution.png").read_bytes()).decode("ascii")
    page = ("<!doctype html><html><meta charset='utf-8'><title>One-layer Attention/MoE</title>"
            "<style>body{font:18px sans-serif;background:#f3f6fa;margin:24px}img,svg{width:100%;max-width:1800px}</style>"
            f"<h1>{html.escape(data['source_run'])}: inferred layer {layer + 1}</h1>"
            "<p>One decode execution; offline order-based attribution. Hover the timeline for kernel details.</p>"
            f"<img src='data:image/png;base64,{png}' alt='Layer timing distribution'>{svg_text_data}</html>")
    with (output / "report.html").open("x") as stream:
        stream.write(page)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--layer-index", type=int, default=23)
    parser.add_argument("--replay-index", type=int, default=32)
    parser.add_argument("--font-dir", type=Path, default=Path("/plot-fonts"))
    args = parser.parse_args()
    if (args.run_dir / "status.txt").read_text().strip() != "PASS":
        raise ValueError("expected a PASS run")
    args.output_dir.mkdir()
    saved = json.loads((args.run_dir / "moe-kernel-share.json").read_text())
    gpu0 = next(row for row in saved["per_device"] if row["device"] == 0)
    path = args.run_dir / "profile" / Path(gpu0["path"]).name
    print(f"Reading GPU 0 trace: {path}", flush=True)
    with path.open() as stream:
        document = json.load(stream)
    data = analyze_document(document, 48, args.layer_index, args.replay_index)
    data["source_run"] = args.run_dir.name
    data["source_trace"] = path.name
    data["source_revision"] = (args.run_dir / "git-commit.txt").read_text().strip()
    data["plot_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    evidence = [Path("/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3_moe.py")]
    evidence += sorted((args.run_dir / "runtime-cache/vllm/torch_compile_cache").glob(
        "*/rank_0_0/backbone/computation_graph.py"))
    data["mapping_source_sha256"] = {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in evidence if p.is_file()
    }
    with (args.output_dir / "layer-data.json").open("x") as stream:
        json.dump(data, stream, indent=2)
    with (args.output_dir / "layer-kernels.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(data["events"][0]))
        writer.writeheader()
        writer.writerows(data["events"])
    render(data, args.output_dir, args.font_dir)
    print(json.dumps({k: data[k] for k in ("decode_graph_candidates", "kernel_time_us",
                                         "kernel_share_percent", "total_kernel_us", "elapsed_us")}))
    print(f"Saved layer-distribution.png, layer-timeline.svg and report.html: {args.output_dir}")


if __name__ == "__main__":
    main()
