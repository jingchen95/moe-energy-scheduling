"""Render an existing successful Torch run; never launch or modify the model.

Requires only Pillow (already in the pinned SIF) plus the standard library.
The timeline SVG has native hover titles with exact kernel names and durations.
"""

from __future__ import annotations

import argparse
import base64
from collections import defaultdict
import hashlib
import html
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

from src.nsys_moe_share import MOE_COMM_PATTERN, MOE_COMPUTE_PATTERN


COLORS = {
    "compute": "#2368C4", "communication": "#D97706",
    "routing": "#7C3AED", "other": "#94A3B8",
}
LABELS = {
    "compute": "Recognized MoE compute / token prep",
    "communication": "EP AllGather + ReduceScatter",
    "routing": "MoE topkGating (missed by original rule)",
    "other": "Other / unassigned kernels",
}


def category(name: str) -> str:
    if MOE_COMPUTE_PATTERN.search(name):
        return "compute"
    if MOE_COMM_PATTERN.search(name):
        return "communication"
    if "vllm::moe::topkGating" in name:
        return "routing"
    return "other"


def overlaps(start: float, duration: float, lower: float, upper: float) -> bool:
    return start < upper and start + duration > lower


def load_data(run: Path, window_ms: float) -> dict:
    if not math.isfinite(window_ms) or window_ms <= 0:
        raise ValueError("window_ms must be finite and positive")
    if (run / "status.txt").read_text().strip() != "PASS":
        raise ValueError("plot only a validated PASS run")
    saved = json.loads((run / "moe-kernel-share.json").read_text())
    if saved.get("status") != "ok_heuristic_attribution":
        raise ValueError("expected validated heuristic kernel statistics")
    devices = sorted(saved["per_device"], key=lambda row: row["device"])
    if [row["device"] for row in devices] != [0, 1, 2, 3]:
        raise ValueError("expected four distinct CUDA devices")
    totals = defaultdict(float)
    window = []
    origin = None
    base = None
    for worker in devices:
        # Resolve by basename within this run so copied run directories work.
        path = run / "profile" / Path(worker["path"]).name
        print(f"Reading GPU {worker['device']}: {path.name}", flush=True)
        with path.open() as stream:
            document = json.load(stream)
        if base is None:
            base = document["baseTimeNanoseconds"]
        if base != document["baseTimeNanoseconds"]:
            raise ValueError("trace clock bases differ; cannot align GPU timeline")
        kernels = [e for e in document["traceEvents"]
                   if e.get("cat") == "kernel" and e.get("ph") == "X"]
        if len(kernels) != worker["kernel_count"]:
            raise ValueError("kernel count differs from saved report")
        if {e["args"]["device"] for e in kernels} != {worker["device"]}:
            raise ValueError("worker device differs from saved report")
        if origin is None:
            graph_calls = {e.get("args", {}).get("correlation")
                           for e in document["traceEvents"]
                           if e.get("cat") in {"cuda_runtime", "cuda_driver"}
                           and e.get("ph") == "X" and "GraphLaunch" in e.get("name", "")}
            graph_calls.discard(None)
            graph = [e["ts"] for e in kernels if e["args"].get("correlation") in graph_calls or any(
                e["args"].get(key, 0) > 0 for key in ("graph id", "graph node id")
            )]
            if not graph:
                raise ValueError("GPU 0 has no verifiable graph-kernel timestamps for timeline origin")
            origin = min(graph)
        for event in kernels:
            kind = category(event["name"])
            totals[kind] += event["dur"]
            if overlaps(event["ts"], event["dur"], origin, origin + window_ms * 1000):
                window.append({
                    "device": worker["device"], "stream": event["args"]["stream"],
                    "name": event["name"], "category": kind,
                    "start_us": event["ts"] - origin, "duration_us": event["dur"],
                })
        del kernels, document
    total = sum(totals.values())
    if not math.isclose(total, saved["total_cuda_kernel_time"], rel_tol=1e-9):
        raise ValueError("raw kernel total differs from saved report")
    original = totals["compute"] + totals["communication"]
    if not math.isclose(original, saved["classified_moe_kernel_time"], rel_tol=1e-9):
        raise ValueError("original classification differs from saved report")
    requests = [json.loads(line) for line in
                (run / "benchmark/requests.jsonl").read_text().splitlines()]
    if len(requests) != 3 or any(r["status"] != "ok" for r in requests):
        raise ValueError("expected three successful measured requests")
    server_log = run / "vllm.log"
    oom_warnings = server_log.read_text().count("CUDA warning: out of memory") if server_log.exists() else 0
    return {
        "source_run": run.name, "source_revision": (run / "git-commit.txt").read_text().strip(),
        "run_scope": saved["run_scope"], "total_kernel_time_us": total,
        "oom_warning_count": oom_warnings,
        "kernel_time_us_by_category": dict(totals),
        "percent_by_category": {k: 100 * totals[k] / total for k in COLORS},
        "original_moe_percent": 100 * original / total,
        "moe_percent_with_topk": 100 * (original + totals["routing"]) / total,
        "window_ms": window_ms, "window_origin_trace_us": origin,
        "baseTimeNanoseconds": base, "window_kernels": window,
        "requests": [{k: r[k] for k in ("request_id", "actual_send_s", "ttft_ms",
                                        "tpot_ms", "e2e_ms", "output_tokens")}
                     for r in requests],
    }


def render(data: dict, output: Path, font_dir: Path) -> None:
    from PIL import Image, ImageDraw, ImageFont

    fonts = {}

    def font(size, bold=False):
        key = (size, bold)
        if key not in fonts:
            filename = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
            fonts[key] = ImageFont.truetype(str(font_dir / filename), size)
        return fonts[key]

    def label(draw, xy, text, size=22, fill="#172B46", bold=False):
        draw.text(xy, text, font=font(size, bold), fill=fill)

    def save_png(image, name):
        with (output / name).open("xb") as stream:
            image.save(stream, format="PNG")

    image = Image.new("RGB", (1800, 1160), "#F3F6FA")
    draw = ImageDraw.Draw(image)
    label(draw, (60, 35), "MoE profiling | measured run overview", 42, bold=True)
    label(draw, (60, 98), data["source_run"], 22)
    label(draw, (60, 140), "4 GPUs | CUDA Graph capture | Original kernel classification + explicit MoE topkGating", 23)
    draw.rounded_rectangle((40, 195, 1760, 665), radius=18, fill="white")
    label(draw, (70, 218), "Share of summed GPU kernel durations (all 4 GPUs)", 29, bold=True)
    for i, kind in enumerate(COLORS):
        y = 292 + 70 * i
        percent = data["percent_by_category"][kind]
        label(draw, (75, y + 3), LABELS[kind], 22)
        draw.rectangle((650, y, 1550, y + 33), fill="#EDF1F6")
        draw.rectangle((650, y, 650 + 900 * percent / 100, y + 33), fill=COLORS[kind])
        label(draw, (1580, y - 2), f"{percent:.2f}%", 28, bold=True)
    label(draw, (75, 585),
          f"Identified MoE incl. topkGating: {data['moe_percent_with_topk']:.2f}%   |   "
          f"Original report: {data['original_moe_percent']:.2f}%", 25, bold=True)
    label(draw, (75, 628), "Not E2E time. Unassigned kernels may include more MoE work; overlapping durations are summed.", 20)

    draw.rounded_rectangle((40, 695, 1760, 1030), radius=18, fill="white")
    label(draw, (70, 715), "Measured requests | client wall-clock timeline", 29, bold=True)
    label(draw, (1330, 761), "TTFT / TPOT / E2E", 20, bold=True)
    max_s = max(r["actual_send_s"] + r["e2e_ms"] / 1000 for r in data["requests"])
    axis_s = math.ceil(max_s)
    for tick in range(axis_s + 1):
        x = 280 + 980 * tick / axis_s
        draw.line((x, 795, x, 975), fill="#E1E7EF")
        label(draw, (x - 6, 988), f"{tick}s", 18)
    for i, row in enumerate(data["requests"]):
        y = 810 + 61 * i
        label(draw, (75, y), row["request_id"], 23)
        x = 280 + 980 * row["actual_send_s"] / axis_s
        end = x + 980 * row["e2e_ms"] / 1000 / axis_s
        first = x + 980 * row["ttft_ms"] / 1000 / axis_s
        draw.rectangle((x, y, end, y + 28), fill="#0D9488")
        draw.rectangle((x, y, first, y + 28), fill="#D97706")
        label(draw, (1330, y), f"{row['ttft_ms']:.1f}ms / {row['tpot_ms']:.2f}ms / {row['e2e_ms']/1000:.3f}s", 20)
    label(draw, (65, 1060), "Orange = waiting for first token; teal = remaining generation. 128 output tokens per request.", 22)
    label(draw, (65, 1100), f"{data['run_scope']} | Profiled latency only | "
          f"CUDA OOM warnings in server log: {data['oom_warning_count']}", 21)
    save_png(image, "overview.png")

    rows = sorted({(e["device"], e["stream"]) for e in data["window_kernels"]})
    height = 360 + 78 * len(rows)
    timeline = Image.new("RGB", (1800, height), "#F3F6FA")
    draw = ImageDraw.Draw(timeline)
    label(draw, (60, 30), "GPU kernel timeline | real trace excerpt", 40, bold=True)
    label(draw, (60, 90), f"Shared {data['window_ms']:g} ms window, starting at GPU 0's first observed CUDA Graph kernel", 24)
    short = ["MoE compute / prep", "EP communication", "MoE routing", "Other / unassigned"]
    for i, (kind, title) in enumerate(zip(COLORS, short)):
        x = 60 + i * 425
        draw.rectangle((x, 150, x + 22, 172), fill=COLORS[kind])
        label(draw, (x + 33, 145), title, 22)
    svg = ET.Element("svg", xmlns="http://www.w3.org/2000/svg",
                     width="1800", height=str(height), viewBox=f"0 0 1800 {height}")
    ET.SubElement(svg, "rect", width="1800", height=str(height), fill="#F3F6FA")

    def svg_text(x, y, text, size=22):
        node = ET.SubElement(svg, "text", x=str(x), y=str(y), fill="#172B46",
                             style=f"font: {size}px sans-serif")
        node.text = text

    svg_text(60, 62, "GPU kernel timeline - hover rectangles for exact names / durations", 32)
    svg_text(60, 110, f"First {data['window_ms']:g} ms after GPU 0's first observed graph kernel. Separate rows per CUDA stream.")
    for i, (kind, title) in enumerate(zip(COLORS, short)):
        ET.SubElement(svg, "rect", x=str(60 + i * 425), y="150", width="22", height="22", fill=COLORS[kind])
        svg_text(93 + i * 425, 170, title)
    for tick in range(9):
        x = 260 + tick * 1460 / 8
        draw.line((x, 210, x, height - 150), fill="#DCE3EB")
        label(draw, (x - 12, height - 135), f"{data['window_ms'] * tick / 8:g}", 18)
        ET.SubElement(svg, "line", x1=str(x), x2=str(x), y1="210", y2=str(height-150), stroke="#DCE3EB")
        svg_text(x-12, height-112, f"{data['window_ms'] * tick / 8:g}", 18)
    for index, (device, stream) in enumerate(rows):
        y = 220 + index * 78
        name = f"GPU {device} / s{stream}"
        label(draw, (55, y + 5), name, 23)
        svg_text(55, y + 29, name, 23)
        draw.rectangle((260, y, 1720, y + 40), fill="white")
        ET.SubElement(svg, "rect", x="260", y=str(y), width="1460", height="40", fill="white")
        for event in data["window_kernels"]:
            if (event["device"], event["stream"]) != (device, stream):
                continue
            start = max(0, event["start_us"])
            end = min(data["window_ms"] * 1000, event["start_us"] + event["duration_us"])
            x = 260 + 1460 * start / (data["window_ms"] * 1000)
            width = 1460 * (end - start) / (data["window_ms"] * 1000)
            color = COLORS[event["category"]]
            draw.rectangle((x, y, x + width, y + 40), fill=color)
            rect = ET.SubElement(svg, "rect", x=str(x), y=str(y), width=str(width), height="40", fill=color)
            ET.SubElement(rect, "title").text = (
                f"GPU {device}, stream {stream}\n{event['name']}\n"
                f"Start: {event['start_us']:.3f} us; duration: {event['duration_us']:.3f} us"
            )
    footer = "Time since excerpt start (ms). Idle gaps are blank; this excerpt is not the complete capture."
    label(draw, (260, height - 78), footer, 21)
    svg_text(260, height-55, footer, 21)
    save_png(timeline, "kernel-timeline.png")
    svg_text_data = ET.tostring(svg, encoding="unicode")
    with (output / "kernel-timeline.svg").open("x") as stream:
        stream.write(svg_text_data)
    png = base64.b64encode((output / "overview.png").read_bytes()).decode("ascii")
    page = ("<!doctype html><html lang='en'><meta charset='utf-8'><title>MoE profiler charts</title>"
            "<style>body{background:#f3f6fa;font:18px sans-serif;margin:24px}img,svg{width:100%;height:auto}"
            "p{line-height:1.6}main{max-width:1800px;margin:auto}</style><main>"
            f"<h1>{html.escape(data['source_run'])}</h1>"
            "<p>Offline visualization of existing data. Raw reports remain unchanged. "
            "Hover timeline rectangles for exact kernel names and durations. "
            "No model execution, GPU synchronization, or profiling changes were introduced.</p>"
            f"<img alt='MoE kernel shares and request latency' src='data:image/png;base64,{png}'>"
            f"{svg_text_data}</main></html>")
    with (output / "report.html").open("x") as stream:
        stream.write(page)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("output_dir", type=Path, help="new directory; must not exist")
    parser.add_argument("--window-ms", type=float, default=2.0)
    parser.add_argument("--font-dir", type=Path, default=Path("/plot-fonts"))
    parser.add_argument("--source-revision", default="unrecorded; use plot_source_files_sha256",
                        help="host Git revision; Git is not required inside the SIF")
    args = parser.parse_args()
    args.output_dir.mkdir()  # Fail closed; never overwrite prior visualization.
    data = load_data(args.run_dir, args.window_ms)
    data["plot_source_revision"] = args.source_revision
    source_files = [Path(__file__), Path(__file__).with_name("nsys_moe_share.py")]
    data["plot_source_files_sha256"] = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files
    }
    with (args.output_dir / "plot-data.json").open("x") as stream:
        json.dump(data, stream, indent=2)
    render(data, args.output_dir, args.font_dir)
    print(f"Saved overview.png, kernel-timeline.png/.svg, report.html: {args.output_dir}")


if __name__ == "__main__":
    main()
