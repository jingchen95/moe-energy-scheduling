#!/usr/bin/env python3
"""Offline P1 expert-group permutation versus heterogeneous-DVFS analysis."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.analyze_layerwise_loads import (  # noqa: E402
    PHASE_ARRAYS,
    aggregate_layer_loads,
    load_profile,
    resolve_repo_path,
    sha256_file,
    validate_mapping,
    write_csv,
)


H, M, L = 2520, 1635, 1005
FREQUENCIES_MHZ = (H, M, L)
IDENTITY = (0, 1, 2, 3)
ALL_GROUP_AT_GPU_PERMUTATIONS = tuple(itertools.permutations(range(4)))
FOCUS_MULTISETS = {"MHHH", "LHHH", "MMHH", "LMHH", "LLHH", "LLLH"}
FREQUENCY_SYMBOL = {H: "H", M: "M", L: "L"}


def frequency_tuples() -> list[tuple[int, int, int, int]]:
    return list(itertools.product(FREQUENCIES_MHZ, repeat=4))


def frequency_label(frequencies: Sequence[int]) -> str:
    return "".join(FREQUENCY_SYMBOL[value] for value in frequencies)


def frequency_multiset(frequencies: Sequence[int]) -> str:
    return "".join(
        FREQUENCY_SYMBOL[value] for value in sorted(frequencies)
    )


def normalized(values: Sequence[int]) -> tuple[float, ...]:
    total = sum(values)
    if total <= 0:
        raise ValueError("group loads must have a positive sum")
    return tuple(value / total for value in values)


def apply_group_permutation(
    group_values: Sequence[float | int], group_at_gpu: Sequence[int]
) -> tuple[float | int, ...]:
    """Return physical-GPU values; group_at_gpu[g] names GPU g's P1 group."""
    if sorted(group_at_gpu) != list(range(len(group_values))):
        raise ValueError("group_at_gpu must be a permutation of all groups")
    return tuple(group_values[group] for group in group_at_gpu)


def invert_group_at_gpu(group_at_gpu: Sequence[int]) -> tuple[int, ...]:
    group_to_gpu = [0] * len(group_at_gpu)
    for gpu, group in enumerate(group_at_gpu):
        group_to_gpu[group] = gpu
    return tuple(group_to_gpu)


def permute_mapping_row(
    original_group_for_expert: Sequence[int], group_to_gpu: Sequence[int]
) -> tuple[int, ...]:
    """Relabel intact P1 groups without changing any group's expert members."""
    if sorted(group_to_gpu) != list(range(len(group_to_gpu))):
        raise ValueError("group_to_gpu must be a permutation of all groups")
    if any(group < 0 or group >= len(group_to_gpu) for group in original_group_for_expert):
        raise ValueError("original mapping contains an unknown group")
    return tuple(group_to_gpu[group] for group in original_group_for_expert)


def pressure_max(load_fractions: Sequence[float], speeds: Sequence[float]) -> float:
    return max(load / speed for load, speed in zip(load_fractions, speeds))


def rank_order_inversions(
    physical_loads: Sequence[float], speeds: Sequence[float]
) -> int:
    """Count slower/faster pairs where the slower GPU receives more load."""
    inversions = 0
    for left in range(len(speeds)):
        for right in range(left + 1, len(speeds)):
            if speeds[left] < speeds[right] and physical_loads[left] > physical_loads[right]:
                inversions += 1
            elif speeds[right] < speeds[left] and physical_loads[right] > physical_loads[left]:
                inversions += 1
    return inversions


def choose_aligned_permutation(
    group_load_fractions: Sequence[float], frequencies: Sequence[int]
) -> dict[str, object]:
    """Exhaust all 24 assignments and choose the deterministic proxy optimum."""
    if len(group_load_fractions) != 4 or len(frequencies) != 4:
        raise ValueError("exactly four groups and four frequencies are required")
    speeds = tuple(frequency / H for frequency in frequencies)
    candidates = []
    for group_at_gpu in ALL_GROUP_AT_GPU_PERMUTATIONS:
        physical = tuple(
            float(value)
            for value in apply_group_permutation(group_load_fractions, group_at_gpu)
        )
        objective = pressure_max(physical, speeds)
        inversions = rank_order_inversions(physical, speeds)
        identity_distance = sum(
            group != gpu for gpu, group in enumerate(group_at_gpu)
        )
        candidates.append((
            objective, inversions, identity_distance, group_at_gpu, physical
        ))
    objective, inversions, identity_distance, group_at_gpu, physical = min(candidates)
    return {
        "group_at_gpu": group_at_gpu,
        "group_to_gpu": invert_group_at_gpu(group_at_gpu),
        "physical_load_fractions": physical,
        "max_pressure": objective,
        "rank_order_inversions": inversions,
        "identity_distance": identity_distance,
    }


def percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def analyze_phase_tuple(
    phase: str,
    group_loads: Sequence[Sequence[int]],
    frequencies: tuple[int, int, int, int],
    source: dict[str, str],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    speeds = tuple(frequency / H for frequency in frequencies)
    detail: list[dict[str, object]] = []
    selected_permutations: list[tuple[int, ...]] = []
    for layer, absolute_groups in enumerate(group_loads):
        group_fractions = normalized(absolute_groups)
        baseline_pressure = pressure_max(group_fractions, speeds)
        baseline_inversions = rank_order_inversions(group_fractions, speeds)
        aligned = choose_aligned_permutation(group_fractions, frequencies)
        group_at_gpu = tuple(aligned["group_at_gpu"])
        group_to_gpu = tuple(aligned["group_to_gpu"])
        physical_fractions = tuple(aligned["physical_load_fractions"])
        physical_absolute = apply_group_permutation(absolute_groups, group_at_gpu)
        aligned_pressure = float(aligned["max_pressure"])
        reduction_pct = 100.0 * (baseline_pressure - aligned_pressure) / baseline_pressure
        selected_permutations.append(group_at_gpu)
        row: dict[str, object] = {
            **source,
            "phase": phase,
            "frequency_label": frequency_label(frequencies),
            "frequency_multiset": frequency_multiset(frequencies),
            "focus_multiset": frequency_multiset(frequencies) in FOCUS_MULTISETS,
            "layer": layer,
        }
        for gpu in range(4):
            row[f"gpu{gpu}_frequency_mhz"] = frequencies[gpu]
            row[f"gpu{gpu}_speed_factor"] = speeds[gpu]
            row[f"group{gpu}_absolute_load"] = absolute_groups[gpu]
            row[f"group{gpu}_load_fraction"] = group_fractions[gpu]
            row[f"aligned_gpu{gpu}_group"] = group_at_gpu[gpu]
            row[f"aligned_gpu{gpu}_absolute_load"] = physical_absolute[gpu]
            row[f"aligned_gpu{gpu}_load_fraction"] = physical_fractions[gpu]
        row.update({
            "group_at_gpu": "|".join(str(value) for value in group_at_gpu),
            "group_to_gpu": "|".join(str(value) for value in group_to_gpu),
            "baseline_max_pressure": baseline_pressure,
            "aligned_max_pressure": aligned_pressure,
            "layer_pressure_reduction_pct": reduction_pct,
            "assignment_changed": group_at_gpu != IDENTITY,
            "identity_distance": aligned["identity_distance"],
            "baseline_rank_order_inversions": baseline_inversions,
            "aligned_rank_order_inversions": aligned["rank_order_inversions"],
        })
        detail.append(row)

    baseline_pressures = [float(row["baseline_max_pressure"]) for row in detail]
    aligned_pressures = [float(row["aligned_max_pressure"]) for row in detail]
    reductions = [float(row["layer_pressure_reduction_pct"]) for row in detail]
    baseline_mean = statistics.fmean(baseline_pressures)
    aligned_mean = statistics.fmean(aligned_pressures)
    permutation_changes = sum(
        left != right
        for left, right in zip(selected_permutations, selected_permutations[1:])
    )
    summary: dict[str, object] = {
        **source,
        "phase": phase,
        "frequency_label": frequency_label(frequencies),
        "frequency_multiset": frequency_multiset(frequencies),
        "focus_multiset": frequency_multiset(frequencies) in FOCUS_MULTISETS,
        "gpu0_frequency_mhz": frequencies[0],
        "gpu1_frequency_mhz": frequencies[1],
        "gpu2_frequency_mhz": frequencies[2],
        "gpu3_frequency_mhz": frequencies[3],
        "baseline_mean_max_pressure": baseline_mean,
        "aligned_mean_max_pressure": aligned_mean,
        "relative_mean_pressure_reduction_pct": (
            100.0 * (baseline_mean - aligned_mean) / baseline_mean
        ),
        "layer_reduction_p50_pct": percentile(reductions, 0.50),
        "layer_reduction_p90_pct": percentile(reductions, 0.90),
        "layer_reduction_max_pct": max(reductions),
        "layers_assignment_changed": sum(
            bool(row["assignment_changed"]) for row in detail
        ),
        "distinct_permutations": len(set(selected_permutations)),
        "adjacent_permutation_changes": permutation_changes,
        "baseline_zero_inversion_layers": sum(
            int(row["baseline_rank_order_inversions"]) == 0 for row in detail
        ),
        "aligned_zero_inversion_layers": sum(
            int(row["aligned_rank_order_inversions"]) == 0 for row in detail
        ),
    }
    for gpu in range(4):
        summary[f"baseline_gpu{gpu}_mean_load_fraction"] = statistics.fmean(
            float(row[f"group{gpu}_load_fraction"]) for row in detail
        )
        summary[f"aligned_gpu{gpu}_mean_load_fraction"] = statistics.fmean(
            float(row[f"aligned_gpu{gpu}_load_fraction"]) for row in detail
        )
        summary[f"aligned_gpu{gpu}_lightest_layer_count"] = sum(
            float(row[f"aligned_gpu{gpu}_load_fraction"])
            == min(float(row[f"aligned_gpu{rank}_load_fraction"]) for rank in range(4))
            for row in detail
        )
        summary[f"aligned_gpu{gpu}_heaviest_layer_count"] = sum(
            float(row[f"aligned_gpu{gpu}_load_fraction"])
            == max(float(row[f"aligned_gpu{rank}_load_fraction"]) for rank in range(4))
            for row in detail
        )
    return detail, summary


def load_frozen_inputs(plan_path: Path, repo_root: Path) -> dict[str, object]:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    profile_path = resolve_repo_path(plan["phase_counts"], repo_root)
    profile_hash = sha256_file(profile_path)
    if profile_hash != plan["phase_counts_sha256"]:
        raise ValueError("frozen phase-count profile hash mismatch")
    profile = load_profile(profile_path)
    if (len(profile["total"]), len(profile["total"][0])) != (48, 128):
        raise ValueError("expected a 48 x 128 frozen profile")
    p1_item = next(item for item in plan["placements"] if item["name"] == "upstream_eplb")
    p1_path = resolve_repo_path(p1_item["placement_file"], repo_root)
    p1_hash = sha256_file(p1_path)
    if p1_hash != p1_item["placement_sha256"]:
        raise ValueError("frozen P1 placement hash mismatch")
    p1 = json.loads(p1_path.read_text(encoding="utf-8"))
    mapping = p1["logical_to_rank"]
    validate_mapping(mapping, 48, 128, 4)
    group_loads = {
        phase: aggregate_layer_loads(profile[phase], mapping, 4)
        for phase in PHASE_ARRAYS
    }
    return {
        "plan": plan,
        "profile_path": profile_path,
        "profile_sha256": profile_hash,
        "p1_path": p1_path,
        "p1_sha256": p1_hash,
        "mapping": mapping,
        "group_loads": group_loads,
    }


def _blend(base: tuple[int, int, int], amount: float) -> str:
    values = tuple(round(255 + (channel - 255) * amount) for channel in base)
    return "#" + "".join(f"{value:02x}" for value in values)


def _load_color(value: float, low: float, high: float) -> str:
    deviation = value - 0.25
    extent = (0.25 - low) if deviation < 0 else (high - 0.25)
    amount = min(abs(deviation) / extent, 1.0) if extent else 0.0
    return _blend((33, 102, 172) if deviation < 0 else (178, 24, 43), amount)


def write_representative_figure(
    path: Path, detail_rows: Sequence[dict[str, object]]
) -> None:
    rows = [
        row for row in detail_rows
        if row["phase"] == "total" and row["frequency_label"] == "MMHH"
    ]
    if len(rows) != 48:
        raise ValueError("representative MMHH total rows are missing")
    fractions = [
        float(row[key]) for row in rows
        for gpu in range(4)
        for key in (f"group{gpu}_load_fraction", f"aligned_gpu{gpu}_load_fraction")
    ]
    low, high = min(fractions), max(fractions)
    width, height = 900, 640
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#222}.title{font-size:20px;font-weight:700}.panel{font-size:14px;font-weight:700}.axis{font-size:10px}.note{font-size:11px;fill:#444}</style>',
        '<text x="450" y="28" text-anchor="middle" class="title">P1 group alignment for MMHH (total profile)</text>',
    ]
    panel_x = [105, 365, 625]
    titles = ["Original physical loads", "Aligned physical loads", "Group at physical GPU"]
    group_colors = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759"]
    for panel, title in enumerate(titles):
        svg.append(f'<text x="{panel_x[panel] + 72}" y="58" text-anchor="middle" class="panel">{title}</text>')
        for gpu in range(4):
            freq = (M, M, H, H)[gpu]
            svg.append(f'<text x="{panel_x[panel] + gpu * 36 + 18}" y="79" text-anchor="middle" class="axis">G{gpu} {freq}</text>')
        for layer, row in enumerate(rows):
            for gpu in range(4):
                if panel == 0:
                    value = float(row[f"group{gpu}_load_fraction"])
                    color = _load_color(value, low, high)
                    tip = f"Layer {layer}, GPU {gpu}: {value:.4f}"
                elif panel == 1:
                    value = float(row[f"aligned_gpu{gpu}_load_fraction"])
                    color = _load_color(value, low, high)
                    tip = f"Layer {layer}, GPU {gpu}: {value:.4f}"
                else:
                    group = int(row[f"aligned_gpu{gpu}_group"])
                    color = group_colors[group]
                    tip = f"Layer {layer}, GPU {gpu}: original P1 group {group}"
                svg.append(
                    f'<rect x="{panel_x[panel] + gpu * 36}" y="{88 + layer * 9}" width="36" height="9" '
                    f'fill="{color}" stroke="#fff" stroke-width="0.35"><title>{tip}</title></rect>'
                )
            if panel == 0 and layer % 4 == 0:
                svg.append(f'<text x="{panel_x[panel] - 8}" y="{95 + layer * 9}" text-anchor="end" class="axis">{layer}</text>')
    svg.append('<text x="28" y="310" text-anchor="middle" class="axis" transform="rotate(-90 28 310)">MoE layer</text>')
    for group, color in enumerate(group_colors):
        x = 625 + group * 57
        svg.append(f'<rect x="{x}" y="548" width="12" height="12" fill="{color}"/>')
        svg.append(f'<text x="{x + 16}" y="558" class="axis">G{group}</text>')
    svg.append('<text x="450" y="605" text-anchor="middle" class="note">The first two physical GPUs are slower (M); colors in the load panels are centered at equal share 0.25.</text>')
    svg.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(svg) + "\n", encoding="utf-8")


def write_summary_figure(
    path: Path, summary_rows: Sequence[dict[str, object]]
) -> None:
    multisets = ["MHHH", "LHHH", "MMHH", "LMHH", "LLHH", "LLLH"]
    phases = ["total", "prefill", "decode"]
    colors = {"total": "#4e79a7", "prefill": "#59a14f", "decode": "#e15759"}
    aggregates: dict[tuple[str, str], tuple[float, float, float]] = {}
    for multiset in multisets:
        for phase in phases:
            values = [
                float(row["relative_mean_pressure_reduction_pct"])
                for row in summary_rows
                if row["frequency_multiset"] == multiset and row["phase"] == phase
            ]
            aggregates[(multiset, phase)] = (
                min(values), statistics.fmean(values), max(values)
            )
    ymax = max(value[2] for value in aggregates.values()) * 1.12
    width, height = 900, 500
    left, top, plot_width, plot_height = 75, 60, 770, 350
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#222}.title{font-size:20px;font-weight:700}.axis{font-size:11px}.note{font-size:11px;fill:#444}</style>',
        '<text x="450" y="28" text-anchor="middle" class="title">Pressure-proxy reduction across focused frequency multisets</text>',
    ]
    for tick in range(6):
        value = ymax * tick / 5
        y = top + plot_height - plot_height * value / ymax
        svg.append(f'<line x1="{left}" x2="{left + plot_width}" y1="{y}" y2="{y}" stroke="#ddd"/>')
        svg.append(f'<text x="{left - 8}" y="{y + 4}" text-anchor="end" class="axis">{value:.1f}</text>')
    group_width = plot_width / len(multisets)
    offsets = [-22, 0, 22]
    for index, multiset in enumerate(multisets):
        center = left + group_width * (index + 0.5)
        svg.append(f'<text x="{center}" y="{top + plot_height + 24}" text-anchor="middle" class="axis">{multiset}</text>')
        for phase_index, phase in enumerate(phases):
            low, mean, high = aggregates[(multiset, phase)]
            x = center + offsets[phase_index]
            y_low = top + plot_height - plot_height * low / ymax
            y_mean = top + plot_height - plot_height * mean / ymax
            y_high = top + plot_height - plot_height * high / ymax
            svg.append(f'<line x1="{x}" x2="{x}" y1="{y_high}" y2="{y_low}" stroke="{colors[phase]}" stroke-width="2"/>')
            svg.append(f'<circle cx="{x}" cy="{y_mean}" r="4" fill="{colors[phase]}"><title>{multiset} {phase}: mean {mean:.2f}%, range {low:.2f}-{high:.2f}%</title></circle>')
    svg.append('<text x="18" y="235" text-anchor="middle" class="axis" transform="rotate(-90 18 235)">Mean max-pressure reduction (%)</text>')
    for index, phase in enumerate(phases):
        x = 330 + index * 95
        svg.append(f'<circle cx="{x}" cy="470" r="5" fill="{colors[phase]}"/>')
        svg.append(f'<text x="{x + 10}" y="474" class="axis">{phase}</text>')
    svg.append('<text x="450" y="445" text-anchor="middle" class="note">Point: mean across physical permutations; line: min-max. Proxy only, not measured latency or energy.</text>')
    svg.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(svg) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan", type=Path,
        default=Path("results/raw/milestone-h-plan-20260831T100215Z-seeded/matrix-plan.json"),
    )
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output-dir", type=Path, default=Path("results/processed"))
    parser.add_argument("--figure-dir", type=Path, default=Path("results/figures"))
    args = parser.parse_args(argv)
    repo_root = args.repo_root.resolve()
    plan_path = args.plan if args.plan.is_absolute() else repo_root / args.plan
    output_dir = args.output_dir if args.output_dir.is_absolute() else repo_root / args.output_dir
    figure_dir = args.figure_dir if args.figure_dir.is_absolute() else repo_root / args.figure_dir
    frozen = load_frozen_inputs(plan_path, repo_root)
    source = {
        "profile_sha256": frozen["profile_sha256"],
        "p1_placement_sha256": frozen["p1_sha256"],
    }
    detail_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for phase in PHASE_ARRAYS:
        for frequencies in frequency_tuples():
            detail, summary = analyze_phase_tuple(
                phase, frozen["group_loads"][phase], frequencies, source
            )
            detail_rows.extend(detail)
            summary_rows.append(summary)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "milestone_k_alignment_layers.csv", detail_rows)
    write_csv(output_dir / "milestone_k_alignment_summary.csv", summary_rows)
    manifest = {
        "schema_version": 1,
        "analysis": "offline_P1_group_to_physical_rank_DVFS_alignment",
        "source_plan": str(args.plan),
        "source_plan_sha256": sha256_file(plan_path),
        "profile_path": str(frozen["profile_path"]),
        "profile_sha256": frozen["profile_sha256"],
        "p1_placement_path": str(frozen["p1_path"]),
        "p1_placement_sha256": frozen["p1_sha256"],
        "profile_shape": [48, 128],
        "experts_per_group_per_layer": 32,
        "phases": list(PHASE_ARRAYS),
        "frequencies_mhz": {"H": H, "M": M, "L": L},
        "frequency_tuple_count": len(frequency_tuples()),
        "permutations_per_layer": len(ALL_GROUP_AT_GPU_PERMUTATIONS),
        "permutation_semantics": "group_at_gpu[g] is the original P1 group assigned to physical GPU g",
        "primary_proxy": "max_g(expert_load_fraction[layer,g] / (frequency[g] / H))",
        "tie_breakers": [
            "fewest rank-order inversions",
            "minimum Hamming distance from original P1 physical assignment",
            "lexicographic group_at_gpu tuple",
        ],
        "caveat": "structural capacity proxy only; not execution time, latency, energy, or a service model",
    }
    (output_dir / "milestone_k_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_representative_figure(
        figure_dir / "milestone_k_mmhh_alignment.svg", detail_rows
    )
    write_summary_figure(
        figure_dir / "milestone_k_alignment_summary.svg", summary_rows
    )
    print(json.dumps({
        "detail_rows": len(detail_rows),
        "summary_rows": len(summary_rows),
        "frequency_tuples": len(frequency_tuples()),
        "permutations_per_layer": len(ALL_GROUP_AT_GPU_PERMUTATIONS),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
