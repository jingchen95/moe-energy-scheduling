#!/usr/bin/env python3
"""Analyze post-placement, per-layer expert-token load without GPU execution.

The input profile is the frozen logical-expert routing profile used by the
Milestone H matrix. This tool deliberately uses only the Python standard
library so it can run on a login node without importing the CUDA environment.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import statistics
import struct
import sys
import zipfile
from pathlib import Path
from typing import Iterable, Sequence


PHASE_ARRAYS = {
    "total": "expert_counts",
    "prefill": "prefill_expert_counts",
    "decode": "decode_expert_counts",
}
PLACEMENT_IDS = {
    "upstream_eplb": "P1",
    "controlled_medium": "P2",
    "controlled_high": "P3",
    "controlled_very_high": "P4",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_repo_path(raw_path: str, repo_root: Path) -> Path:
    """Resolve absolute cluster paths after the repository has been relocated."""
    path = Path(raw_path)
    if path.exists():
        return path
    parts = path.parts
    if "moe-energy-scheduling" in parts:
        index = parts.index("moe-energy-scheduling")
        candidate = repo_root.joinpath(*parts[index + 1 :])
        if candidate.exists():
            return candidate
    raise FileNotFoundError(raw_path)


def _read_npy_integer_matrix(payload: bytes) -> list[list[int]]:
    """Read a C-contiguous 2-D integer NPY payload using the standard library."""
    if payload[:6] != b"\x93NUMPY":
        raise ValueError("invalid NPY magic")
    major = payload[6]
    if major == 1:
        header_length = struct.unpack("<H", payload[8:10])[0]
        offset = 10
    elif major in (2, 3):
        header_length = struct.unpack("<I", payload[8:12])[0]
        offset = 12
    else:
        raise ValueError(f"unsupported NPY version {major}")
    header = ast.literal_eval(
        payload[offset : offset + header_length].decode("latin1").strip()
    )
    shape = tuple(int(value) for value in header["shape"])
    if len(shape) != 2 or header["fortran_order"]:
        raise ValueError("expected a C-contiguous two-dimensional NPY array")

    descriptor = header["descr"]
    if len(descriptor) < 3 or descriptor[-2:] not in {"i1", "u1", "i2", "u2", "i4", "u4", "i8", "u8"}:
        raise ValueError(f"unsupported NPY dtype {descriptor!r}")
    kind, size = descriptor[-2], int(descriptor[-1])
    formats = {
        ("i", 1): "b", ("u", 1): "B",
        ("i", 2): "h", ("u", 2): "H",
        ("i", 4): "i", ("u", 4): "I",
        ("i", 8): "q", ("u", 8): "Q",
    }
    byte_order = descriptor[0]
    endian = ">" if byte_order == ">" else "<"
    fmt = endian + formats[(kind, size)]
    data = payload[offset + header_length :]
    expected = shape[0] * shape[1] * size
    if len(data) != expected:
        raise ValueError(f"NPY payload has {len(data)} bytes; expected {expected}")
    values = [item[0] for item in struct.iter_unpack(fmt, data)]
    return [values[row * shape[1] : (row + 1) * shape[1]] for row in range(shape[0])]


def load_profile(path: Path) -> dict[str, list[list[int]]]:
    result: dict[str, list[list[int]]] = {}
    with zipfile.ZipFile(path) as archive:
        members = set(archive.namelist())
        for phase, array_name in PHASE_ARRAYS.items():
            member = f"{array_name}.npy"
            if member not in members:
                raise ValueError(f"profile is missing {member}")
            result[phase] = _read_npy_integer_matrix(archive.read(member))
    validate_profile(result)
    return result


def validate_profile(profile: dict[str, list[list[int]]]) -> None:
    shapes = set()
    for phase in PHASE_ARRAYS:
        matrix = profile[phase]
        if not matrix or not matrix[0]:
            raise ValueError(f"{phase} profile is empty")
        width = len(matrix[0])
        if any(len(row) != width for row in matrix):
            raise ValueError(f"{phase} profile is ragged")
        if any(value < 0 for row in matrix for value in row):
            raise ValueError(f"{phase} profile contains negative counts")
        shapes.add((len(matrix), width))
    if len(shapes) != 1:
        raise ValueError(f"profile shapes differ: {sorted(shapes)}")
    total, prefill, decode = (
        profile["total"], profile["prefill"], profile["decode"]
    )
    for layer in range(len(total)):
        for expert in range(len(total[layer])):
            if total[layer][expert] != prefill[layer][expert] + decode[layer][expert]:
                raise ValueError(
                    f"phase counts do not conserve at layer {layer}, expert {expert}"
                )


def validate_mapping(
    mapping: Sequence[Sequence[int]],
    num_layers: int,
    num_experts: int,
    num_ranks: int = 4,
) -> None:
    if len(mapping) != num_layers or any(len(row) != num_experts for row in mapping):
        raise ValueError(
            f"mapping must have shape ({num_layers}, {num_experts})"
        )
    expected = num_experts // num_ranks
    if expected * num_ranks != num_experts:
        raise ValueError("expert count is not divisible by rank count")
    for layer, row in enumerate(mapping):
        if any(not isinstance(rank, int) or isinstance(rank, bool) for rank in row):
            raise ValueError(f"layer {layer} contains a non-integer rank")
        if any(rank < 0 or rank >= num_ranks for rank in row):
            raise ValueError(f"layer {layer} contains a rank outside [0, {num_ranks})")
        counts = [row.count(rank) for rank in range(num_ranks)]
        if counts != [expected] * num_ranks:
            raise ValueError(
                f"layer {layer} has rank cardinalities {counts}; expected {expected} each"
            )


def aggregate_layer_loads(
    counts: Sequence[Sequence[int]],
    mapping: Sequence[Sequence[int]],
    num_ranks: int = 4,
) -> list[list[int]]:
    """Map logical-expert demand C[layer, expert] to GPU load L[layer, rank]."""
    if not counts or not counts[0]:
        raise ValueError("counts must be non-empty")
    validate_mapping(mapping, len(counts), len(counts[0]), num_ranks)
    loads: list[list[int]] = []
    for layer, (count_row, rank_row) in enumerate(zip(counts, mapping)):
        if len(count_row) != len(counts[0]):
            raise ValueError(f"count row {layer} has an inconsistent width")
        rank_loads = [0] * num_ranks
        for count, rank in zip(count_row, rank_row):
            if count < 0:
                raise ValueError("counts must be nonnegative")
            rank_loads[rank] += int(count)
        loads.append(rank_loads)
    return loads


def normalize_layer_loads(loads: Sequence[Sequence[int]]) -> list[list[float]]:
    normalized = []
    for row in loads:
        total = sum(row)
        normalized.append([value / total for value in row] if total else [0.0] * len(row))
    return normalized


def adjacent_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Histogram intersection; equal to 1 - 0.5 * L1 for normalized vectors."""
    if len(left) != len(right):
        raise ValueError("vectors have different lengths")
    return sum(min(a, b) for a, b in zip(left, right))


def identity_runs(values: Sequence[int]) -> list[dict[str, int]]:
    if not values:
        return []
    runs: list[dict[str, int]] = []
    start = 0
    for index in range(1, len(values) + 1):
        if index == len(values) or values[index] != values[start]:
            runs.append({
                "run_id": len(runs),
                "gpu": int(values[start]),
                "start_layer": start,
                "end_layer": index - 1,
                "length": index - start,
            })
            start = index
    return runs


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(ordered[lower])
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def layer_diagnostics(loads: Sequence[Sequence[int]]) -> list[dict[str, object]]:
    fractions = normalize_layer_loads(loads)
    result: list[dict[str, object]] = []
    for layer, (absolute, normalized) in enumerate(zip(loads, fractions)):
        mean = statistics.fmean(absolute)
        maximum = max(absolute)
        minimum = min(absolute)
        most = min(index for index, value in enumerate(absolute) if value == maximum)
        least = min(index for index, value in enumerate(absolute) if value == minimum)
        cv = statistics.pstdev(absolute) / mean if mean else 0.0
        result.append({
            "layer": layer,
            "loads": list(absolute),
            "fractions": list(normalized),
            "layer_total": sum(absolute),
            "max_over_mean": maximum / mean if mean else 0.0,
            "coefficient_of_variation": cv,
            "most_loaded_gpu": most,
            "least_loaded_gpu": least,
            "most_loaded_tie_count": sum(value == maximum for value in absolute),
            "least_loaded_tie_count": sum(value == minimum for value in absolute),
            "adjacent_similarity": (
                None if layer == 0 else adjacent_similarity(fractions[layer - 1], normalized)
            ),
            "adjacent_l1_distance": (
                None if layer == 0 else sum(
                    abs(a - b) for a, b in zip(fractions[layer - 1], normalized)
                )
            ),
        })
    return result


def summarize_diagnostics(rows: Sequence[dict[str, object]]) -> dict[str, object]:
    max_mean = [float(row["max_over_mean"]) for row in rows]
    cvs = [float(row["coefficient_of_variation"]) for row in rows]
    similarities = [
        float(row["adjacent_similarity"])
        for row in rows if row["adjacent_similarity"] is not None
    ]
    most_values = [int(row["most_loaded_gpu"]) for row in rows]
    least_values = [int(row["least_loaded_gpu"]) for row in rows]
    most_runs = identity_runs(most_values)
    least_runs = identity_runs(least_values)

    def run_stats(runs: Sequence[dict[str, int]], prefix: str) -> dict[str, object]:
        lengths = [run["length"] for run in runs]
        return {
            f"{prefix}_run_count": len(runs),
            f"{prefix}_run_mean_length": statistics.fmean(lengths),
            f"{prefix}_run_median_length": statistics.median(lengths),
            f"{prefix}_run_max_length": max(lengths),
        }

    summary: dict[str, object] = {
        "layer_count": len(rows),
        "mean_max_over_mean": statistics.fmean(max_mean),
        "p50_max_over_mean": _percentile(max_mean, 0.50),
        "p90_max_over_mean": _percentile(max_mean, 0.90),
        "max_max_over_mean": max(max_mean),
        "mean_cv": statistics.fmean(cvs),
        "p90_cv": _percentile(cvs, 0.90),
        "max_cv": max(cvs),
        "mean_adjacent_similarity": statistics.fmean(similarities),
        "min_adjacent_similarity": min(similarities),
        "most_loaded_switches": sum(a != b for a, b in zip(most_values, most_values[1:])),
        "least_loaded_switches": sum(a != b for a, b in zip(least_values, least_values[1:])),
    }
    summary.update(run_stats(most_runs, "most"))
    summary.update(run_stats(least_runs, "least"))
    for gpu in range(4):
        summary[f"most_loaded_layers_gpu{gpu}"] = most_values.count(gpu)
        summary[f"least_loaded_layers_gpu{gpu}"] = least_values.count(gpu)
    return summary


def _run_lookup(runs: Sequence[dict[str, int]]) -> dict[int, dict[str, int]]:
    return {
        layer: run
        for run in runs
        for layer in range(run["start_layer"], run["end_layer"] + 1)
    }


def write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _blend(base: tuple[int, int, int], amount: float) -> str:
    rgb = tuple(round(255 + (channel - 255) * amount) for channel in base)
    return "#" + "".join(f"{channel:02x}" for channel in rgb)


def heat_color(value: float, scale: tuple[float, float]) -> str:
    deviation = value - 0.25
    low_extent, high_extent = scale
    denominator = high_extent if deviation > 0 else low_extent
    amount = min(abs(deviation) / denominator, 1.0) if denominator else 0.0
    return _blend((178, 24, 43) if deviation > 0 else (33, 102, 172), amount)


def _svg_header(width: int, height: int, title: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#222}.title{font-size:20px;font-weight:700}.panel{font-size:14px;font-weight:700}.axis{font-size:10px}.note{font-size:11px;fill:#444}</style>',
        f'<text x="{width / 2}" y="28" text-anchor="middle" class="title">{title}</text>',
    ]


def _draw_heatmap(
    svg: list[str], fractions: Sequence[Sequence[float]], x: int, y: int,
    cell_width: int, cell_height: int, scale: tuple[float, float], label_layers: bool,
) -> None:
    for layer, row in enumerate(fractions):
        for gpu, value in enumerate(row):
            svg.append(
                f'<rect x="{x + gpu * cell_width}" y="{y + layer * cell_height}" '
                f'width="{cell_width}" height="{cell_height}" fill="{heat_color(value, scale)}" '
                f'stroke="#fff" stroke-width="0.35"><title>Layer {layer}, GPU {gpu}: {value:.4f}</title></rect>'
            )
        if label_layers and layer % 4 == 0:
            svg.append(
                f'<text x="{x - 7}" y="{y + layer * cell_height + cell_height * 0.75}" '
                f'text-anchor="end" class="axis">{layer}</text>'
            )
    for gpu in range(4):
        svg.append(
            f'<text x="{x + (gpu + 0.5) * cell_width}" y="{y - 7}" '
            f'text-anchor="middle" class="axis">G{gpu}</text>'
        )


def _draw_legend(
    svg: list[str], x: int, y: int, width: int, scale: tuple[float, float]
) -> None:
    steps = 80
    low_extent, high_extent = scale
    minimum, maximum = 0.25 - low_extent, 0.25 + high_extent
    for index in range(steps):
        fraction = minimum + ((maximum - minimum) * index / (steps - 1))
        svg.append(
            f'<rect x="{x + index * width / steps:.2f}" y="{y}" '
            f'width="{width / steps + 0.5:.2f}" height="12" fill="{heat_color(fraction, scale)}"/>'
        )
    center_x = x + width * low_extent / (low_extent + high_extent)
    svg.extend([
        f'<text x="{x}" y="{y + 27}" text-anchor="middle" class="axis">{minimum:.2f}</text>',
        f'<text x="{center_x}" y="{y + 27}" text-anchor="middle" class="axis">0.25</text>',
        f'<text x="{x + width}" y="{y + 27}" text-anchor="middle" class="axis">{maximum:.2f}</text>',
    ])


def write_p1_figure(
    path: Path, analyses: dict[str, dict[str, object]], scale: tuple[float, float]
) -> None:
    width, height = 760, 625
    svg = _svg_header(width, height, "P1 per-layer post-placement GPU load")
    phases = ["total", "prefill", "decode"]
    for panel, phase in enumerate(phases):
        x, y = 82 + panel * 225, 78
        svg.append(f'<text x="{x + 72}" y="55" text-anchor="middle" class="panel">{phase.title()}</text>')
        _draw_heatmap(svg, analyses[f"P1:{phase}"]["fractions"], x, y, 36, 9, scale, panel == 0)
    svg.append('<text x="23" y="295" text-anchor="middle" class="axis" transform="rotate(-90 23 295)">MoE layer</text>')
    _draw_legend(svg, 280, 540, 200, scale)
    svg.append('<text x="380" y="600" text-anchor="middle" class="note">Each row sums to 1.0; white is balanced at 0.25 per GPU.</text>')
    svg.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(svg) + "\n", encoding="utf-8")


def write_comparison_figure(
    path: Path, analyses: dict[str, dict[str, object]], scale: tuple[float, float]
) -> None:
    width, height = 890, 1190
    svg = _svg_header(width, height, "Layer-wise GPU load across fixed placements")
    phases = ["total", "prefill", "decode"]
    placements = ["P1", "P2", "P3", "P4"]
    for column, phase in enumerate(phases):
        svg.append(f'<text x="{190 + column * 245}" y="58" text-anchor="middle" class="panel">{phase.title()}</text>')
    for row_index, placement in enumerate(placements):
        y = 92 + row_index * 250
        svg.append(f'<text x="36" y="{y + 117}" text-anchor="middle" class="panel" transform="rotate(-90 36 {y + 117})">{placement}</text>')
        for column, phase in enumerate(phases):
            x = 118 + column * 245
            entry = analyses[f"{placement}:{phase}"]
            _draw_heatmap(svg, entry["fractions"], x, y, 34, 4, scale, column == 0)
            summary = entry["summary"]
            svg.append(
                f'<text x="{x + 68}" y="{y + 215}" text-anchor="middle" class="note">'
                f'mean max/mean {summary["mean_max_over_mean"]:.3f}; longest heavy run {summary["most_run_max_length"]}</text>'
            )
    _draw_legend(svg, 345, 1100, 200, scale)
    svg.append('<text x="445" y="1162" text-anchor="middle" class="note">Normalized expert-token fractions; equal white share = 0.25. Scale shared across all panels.</text>')
    svg.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(svg) + "\n", encoding="utf-8")


def analyze(plan_path: Path, repo_root: Path) -> dict[str, object]:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    profile_path = resolve_repo_path(plan["phase_counts"], repo_root)
    if sha256_file(profile_path) != plan["phase_counts_sha256"]:
        raise ValueError("phase-count profile hash does not match the frozen plan")
    profile = load_profile(profile_path)
    num_layers, num_experts = len(profile["total"]), len(profile["total"][0])
    if (num_layers, num_experts) != (48, 128):
        raise ValueError(f"expected profile shape (48, 128), got {(num_layers, num_experts)}")

    placements: dict[str, dict[str, object]] = {}
    for item in plan["placements"]:
        placement_id = PLACEMENT_IDS.get(item["name"])
        if placement_id is None:
            continue
        path = resolve_repo_path(item["placement_file"], repo_root)
        actual_hash = sha256_file(path)
        if actual_hash != item["placement_sha256"]:
            raise ValueError(f"{placement_id} hash does not match the frozen plan")
        payload = json.loads(path.read_text(encoding="utf-8"))
        mapping = payload["logical_to_rank"]
        validate_mapping(mapping, num_layers, num_experts)
        placements[placement_id] = {
            "name": item["name"], "path": path, "sha256": actual_hash,
            "mapping": mapping,
        }
    if set(placements) != {"P1", "P2", "P3", "P4"}:
        raise ValueError(f"missing placements: {sorted(set(PLACEMENT_IDS.values()) - set(placements))}")

    analyses: dict[str, dict[str, object]] = {}
    detail_rows: list[dict[str, object]] = []
    run_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for placement_id in ["P1", "P2", "P3", "P4"]:
        placement = placements[placement_id]
        for phase in PHASE_ARRAYS:
            loads = aggregate_layer_loads(profile[phase], placement["mapping"])
            fractions = normalize_layer_loads(loads)
            diagnostics = layer_diagnostics(loads)
            summary = summarize_diagnostics(diagnostics)
            analyses[f"{placement_id}:{phase}"] = {
                "loads": loads, "fractions": fractions,
                "diagnostics": diagnostics, "summary": summary,
            }
            most_runs = identity_runs([int(row["most_loaded_gpu"]) for row in diagnostics])
            least_runs = identity_runs([int(row["least_loaded_gpu"]) for row in diagnostics])
            most_lookup, least_lookup = _run_lookup(most_runs), _run_lookup(least_runs)
            for kind, runs in (("most_loaded", most_runs), ("least_loaded", least_runs)):
                for run in runs:
                    run_rows.append({
                        "profile_sha256": plan["phase_counts_sha256"],
                        "placement": placement_id,
                        "placement_name": placement["name"],
                        "placement_sha256": placement["sha256"],
                        "phase": phase,
                        "identity": kind,
                        **run,
                    })
            for diagnostic in diagnostics:
                layer = int(diagnostic["layer"])
                row: dict[str, object] = {
                    "profile_sha256": plan["phase_counts_sha256"],
                    "placement": placement_id,
                    "placement_name": placement["name"],
                    "placement_sha256": placement["sha256"],
                    "phase": phase,
                    "layer": layer,
                }
                for gpu in range(4):
                    row[f"gpu{gpu}_load"] = diagnostic["loads"][gpu]
                    row[f"gpu{gpu}_fraction"] = f'{diagnostic["fractions"][gpu]:.9f}'
                for key in (
                    "layer_total", "max_over_mean", "coefficient_of_variation",
                    "most_loaded_gpu", "least_loaded_gpu", "most_loaded_tie_count",
                    "least_loaded_tie_count", "adjacent_similarity", "adjacent_l1_distance",
                ):
                    value = diagnostic[key]
                    row[key] = "" if value is None else value
                for prefix, lookup in (("most", most_lookup), ("least", least_lookup)):
                    row[f"{prefix}_run_id"] = lookup[layer]["run_id"]
                    row[f"{prefix}_run_length"] = lookup[layer]["length"]
                detail_rows.append(row)
            summary_rows.append({
                "profile_sha256": plan["phase_counts_sha256"],
                "placement": placement_id,
                "placement_name": placement["name"],
                "placement_sha256": placement["sha256"],
                "phase": phase,
                **summary,
            })
    return {
        "plan": plan, "profile_path": profile_path, "placements": placements,
        "analyses": analyses, "detail_rows": detail_rows,
        "run_rows": run_rows, "summary_rows": summary_rows,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan", type=Path,
        default=Path("results/raw/milestone-h-plan-20260831T100215Z-seeded/matrix-plan.json"),
        help="Milestone H matrix plan that freezes the profile and placements",
    )
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, default=Path("results/processed"))
    parser.add_argument("--figure-dir", type=Path, default=Path("results/figures"))
    args = parser.parse_args(argv)
    repo_root = args.repo_root.resolve()
    plan_path = args.plan if args.plan.is_absolute() else repo_root / args.plan
    output_dir = args.output_dir if args.output_dir.is_absolute() else repo_root / args.output_dir
    figure_dir = args.figure_dir if args.figure_dir.is_absolute() else repo_root / args.figure_dir

    result = analyze(plan_path, repo_root)
    write_csv(output_dir / "milestone_j_layer_loads.csv", result["detail_rows"])
    write_csv(output_dir / "milestone_j_persistence_runs.csv", result["run_rows"])
    write_csv(output_dir / "milestone_j_summary.csv", result["summary_rows"])
    all_fractions = [
        value for entry in result["analyses"].values()
        for row in entry["fractions"] for value in row
    ]
    p1_fractions = [
        value for key, entry in result["analyses"].items() if key.startswith("P1:")
        for row in entry["fractions"] for value in row
    ]
    p1_scale = (0.25 - min(p1_fractions), max(p1_fractions) - 0.25)
    comparison_scale = (0.25 - min(all_fractions), max(all_fractions) - 0.25)
    write_p1_figure(
        figure_dir / "milestone_j_p1_layer_load_heatmaps.svg",
        result["analyses"], p1_scale,
    )
    write_comparison_figure(
        figure_dir / "milestone_j_placement_comparison.svg",
        result["analyses"], comparison_scale,
    )
    manifest = {
        "schema_version": 1,
        "analysis": "descriptive_per_layer_post_placement_expert_token_load",
        "source_plan": str(args.plan),
        "profile": str(result["profile_path"]),
        "profile_sha256": result["plan"]["phase_counts_sha256"],
        "shape": [48, 128],
        "phases": list(PHASE_ARRAYS),
        "placements": {
            placement_id: {
                "name": entry["name"], "path": str(entry["path"]),
                "sha256": entry["sha256"], "experts_per_rank_per_layer": 32,
            }
            for placement_id, entry in result["placements"].items()
        },
        "normalization": "each layer's four post-placement GPU loads sum to 1.0",
        "adjacent_similarity": "sum_g min(v[layer-1,g], v[layer,g]); equal to 1 - L1/2",
        "selection_note": "expert loads are descriptive diagnostics, not execution-time models",
    }
    (output_dir / "milestone_j_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "detail_rows": len(result["detail_rows"]),
        "run_rows": len(result["run_rows"]),
        "summary_rows": len(result["summary_rows"]),
        "p1_heatmap_scale": p1_scale,
        "comparison_heatmap_scale": comparison_scale,
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
