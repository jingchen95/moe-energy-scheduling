#!/usr/bin/env python3
"""Prepare intact-P1-group placements and the Milestone L validation plan."""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.analyze_layerwise_loads import (
    aggregate_layer_loads,
    load_profile,
    resolve_repo_path,
    sha256_file,
    validate_mapping,
)
from tools.analyze_rank_permutation_dvfs_alignment import (
    H,
    M,
    choose_aligned_permutation,
    normalized,
    permute_mapping_row,
)


ORDER_SEED = 20260923
CONFIGURATIONS = (
    ("C1", "original_p1", (H, H, H, H)),
    ("C2", "p1_align_mhhh", (H, H, H, H)),
    ("C3", "p1_align_mmhh", (H, H, H, H)),
    ("C4", "original_p1", (M, H, H, H)),
    ("C5", "p1_align_mhhh", (M, H, H, H)),
    ("C6", "original_p1", (M, M, H, H)),
    ("C7", "p1_align_mmhh", (M, M, H, H)),
)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def validate_intact_group_permutation(
    original: Sequence[Sequence[int]], aligned: Sequence[Sequence[int]],
    *, experts_per_group: int = 32,
) -> list[list[int]]:
    """Return group-at-GPU rows after proving every original group is intact."""
    if len(original) != len(aligned):
        raise ValueError("original and aligned mappings have different layer counts")
    group_at_gpu_rows: list[list[int]] = []
    for layer, (original_row, aligned_row) in enumerate(zip(original, aligned)):
        if len(original_row) != len(aligned_row):
            raise ValueError(f"layer {layer} has different expert counts")
        original_groups = {
            group: {expert for expert, value in enumerate(original_row) if value == group}
            for group in range(4)
        }
        aligned_groups = {
            gpu: {expert for expert, value in enumerate(aligned_row) if value == gpu}
            for gpu in range(4)
        }
        if any(len(members) != experts_per_group for members in original_groups.values()):
            raise ValueError(f"layer {layer} original P1 group cardinality is invalid")
        if any(len(members) != experts_per_group for members in aligned_groups.values()):
            raise ValueError(f"layer {layer} aligned GPU cardinality is invalid")
        group_at_gpu = []
        for gpu in range(4):
            matches = [
                group for group in range(4)
                if aligned_groups[gpu] == original_groups[group]
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"layer {layer} GPU{gpu} does not contain one intact P1 group"
                )
            group_at_gpu.append(matches[0])
        if sorted(group_at_gpu) != list(range(4)):
            raise ValueError(f"layer {layer} does not preserve all four P1 groups")
        group_at_gpu_rows.append(group_at_gpu)
    return group_at_gpu_rows


def construct_aligned_placement(
    original_mapping: Sequence[Sequence[int]],
    total_counts: Sequence[Sequence[int]],
    frequencies: tuple[int, int, int, int],
    *, name: str,
    source_hashes: dict[str, str] | None = None,
) -> dict[str, Any]:
    validate_mapping(original_mapping, len(total_counts), len(total_counts[0]), 4)
    group_loads = aggregate_layer_loads(total_counts, original_mapping, 4)
    aligned_mapping: list[list[int]] = []
    chosen_group_at_gpu: list[list[int]] = []
    for original_row, loads in zip(original_mapping, group_loads):
        choice = choose_aligned_permutation(normalized(loads), frequencies)
        group_to_gpu = tuple(choice["group_to_gpu"])
        aligned_mapping.append(list(permute_mapping_row(original_row, group_to_gpu)))
        chosen_group_at_gpu.append(list(choice["group_at_gpu"]))
    experts_per_group = len(original_mapping[0]) // 4
    observed = validate_intact_group_permutation(
        original_mapping, aligned_mapping, experts_per_group=experts_per_group
    )
    if observed != chosen_group_at_gpu:
        raise AssertionError("constructed mapping does not match selected permutations")
    return {
        "schema_version": 1,
        "name": name,
        "num_layers": len(aligned_mapping),
        "num_experts": len(aligned_mapping[0]),
        "num_ranks": 4,
        "experts_per_rank": len(aligned_mapping[0]) // 4,
        "logical_to_rank": aligned_mapping,
        "metadata": {
            "algorithm": "Milestone K exhaustive intact-P1-group rank permutation",
            "alignment_phase": "total",
            "frequency_tuple_mhz": list(frequencies),
            "objective": "min max_g(load_fraction[layer,g] / (frequency[g] / 2520))",
            "group_membership_changed": False,
            "replication": False,
            "source_hashes": dict(source_hashes or {}),
            "group_at_gpu_per_layer": chosen_group_at_gpu,
        },
    }


def experiment_configurations(seed: int = ORDER_SEED) -> list[dict[str, Any]]:
    rows = [
        {"config_id": config_id, "placement_id": placement_id,
         "frequency_tuple": list(frequencies)}
        for config_id, placement_id, frequencies in CONFIGURATIONS
    ]
    rng = random.Random(seed)
    rng.shuffle(rows)
    for task_index, row in enumerate(rows):
        repetitions = [0, 1, 2]
        rng.shuffle(repetitions)
        row["task_index"] = task_index
        row["runs"] = [
            {"order": order, "repetition": repetition}
            for order, repetition in enumerate(repetitions)
        ]
    return rows


def _git_commit(repo_root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
    ).strip()


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = args.repo_root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    milestone_i = args.milestone_i_study.resolve()
    i_plan_path = milestone_i / "experiment-plan.json"
    i_plan = json.loads(i_plan_path.read_text(encoding="utf-8"))
    if i_plan.get("study") != "higher_load_repeatability":
        raise ValueError("expected the validated Milestone I experiment plan")
    if float(i_plan["request_rate"]) != 16.0 or int(i_plan["requests"]) != 300:
        raise ValueError("Milestone I trace is not the fixed 300-request, 16 req/s trace")

    k_manifest_path = args.milestone_k_manifest.resolve()
    k_manifest = json.loads(k_manifest_path.read_text(encoding="utf-8"))
    h_plan_path = resolve_repo_path(k_manifest["source_plan"], repo_root)
    h_plan = json.loads(h_plan_path.read_text(encoding="utf-8"))
    profile_path = resolve_repo_path(k_manifest["profile_path"], repo_root)
    p1_path = resolve_repo_path(k_manifest["p1_placement_path"], repo_root)
    trace_path = resolve_repo_path(i_plan["trace"], repo_root)
    model_config = resolve_repo_path(i_plan["model_config"], repo_root)

    expected_hashes = {
        "profile": k_manifest["profile_sha256"],
        "p1": k_manifest["p1_placement_sha256"],
        "trace": i_plan["trace_sha256"],
    }
    actual_hashes = {
        "profile": sha256_file(profile_path),
        "p1": sha256_file(p1_path),
        "trace": sha256_file(trace_path),
    }
    if actual_hashes != expected_hashes:
        raise ValueError(
            f"frozen input hash mismatch: expected {expected_hashes}, got {actual_hashes}"
        )
    if h_plan.get("phase_counts_sha256") != actual_hashes["profile"]:
        raise ValueError("Milestone H and K profile hashes differ")

    profile = load_profile(profile_path)
    p1 = json.loads(p1_path.read_text(encoding="utf-8"))
    original_mapping = p1["logical_to_rank"]
    validate_mapping(original_mapping, 48, 128, 4)
    placements_dir = output / "placements"
    placements_dir.mkdir()
    placement_payloads = {
        "original_p1": p1,
        "p1_align_mhhh": construct_aligned_placement(
            original_mapping, profile["total"], (M, H, H, H),
            name="p1_align_mhhh", source_hashes=actual_hashes,
        ),
        "p1_align_mmhh": construct_aligned_placement(
            original_mapping, profile["total"], (M, M, H, H),
            name="p1_align_mmhh", source_hashes=actual_hashes,
        ),
    }
    placement_specs: dict[str, dict[str, Any]] = {}
    for placement_id, payload in placement_payloads.items():
        path = placements_dir / f"{placement_id}.json"
        write_json(path, payload)
        group_at_gpu = validate_intact_group_permutation(
            original_mapping, payload["logical_to_rank"]
        )
        placement_specs[placement_id] = {
            "placement_id": placement_id,
            "runtime_placement_name": payload["name"],
            "placement_file": str(path),
            "placement_sha256": sha256_file(path),
            "source_p1_sha256": actual_hashes["p1"],
            "intact_group_validation": True,
            "experts_per_rank_per_layer": 32,
            "distinct_group_at_gpu_permutations": len({tuple(row) for row in group_at_gpu}),
        }

    configs = experiment_configurations(args.order_seed)
    for row in configs:
        row["placement"] = placement_specs[row["placement_id"]]
    plan = {
        "schema_version": 1,
        "study": "real_gpu_p1_group_alignment_validation",
        "git_commit": _git_commit(repo_root),
        "model": "Qwen/Qwen3-30B-A3B",
        "vllm_version": "0.15.1",
        "node": "uranus",
        "exclusive_node": True,
        "parallel": {"tensor": 1, "data": 4, "expert": 4},
        "all2all_backend": "allgather_reducescatter",
        "model_config": str(model_config),
        "model_config_sha256": sha256_file(model_config),
        "trace": str(trace_path),
        "trace_sha256": actual_hashes["trace"],
        "request_rate": 16.0,
        "requests": 300,
        "decoding": {"temperature": 0.0, "seed": 0, "max_output_tokens": 128},
        "phase_counts": str(profile_path),
        "phase_counts_sha256": actual_hashes["profile"],
        "source_p1": str(p1_path),
        "source_p1_sha256": actual_hashes["p1"],
        "source_milestone_i_plan": str(i_plan_path),
        "source_milestone_i_plan_sha256": sha256_file(i_plan_path),
        "source_milestone_k_manifest": str(k_manifest_path),
        "source_milestone_k_manifest_sha256": sha256_file(k_manifest_path),
        "frequencies_mhz": {"H": H, "M": M},
        "configuration_order_seed": args.order_seed,
        "randomization": "seven serialized configuration tasks in seeded order; three repetitions seeded within each task",
        "configuration_order": [row["config_id"] for row in configs],
        "configurations": configs,
        "configuration_count": 7,
        "repetitions": 3,
        "expected_runs": 21,
        "ttft_guard_ms": 500.0,
        "energy_source": "NVML cumulative total-energy counter; no idle subtraction",
        "fused_moe_timing_primary_runs": False,
        "fused_moe_timing_reason": (
            "existing timing requires extra vLLM patches and --enforce-eager; "
            "exclude it because overhead/equivalence is not established"
        ),
        "placement_specs": placement_specs,
    }
    write_json(output / "experiment-plan.json", plan)
    print(json.dumps({
        "output_dir": str(output),
        "configuration_order": plan["configuration_order"],
        "placement_hashes": {
            key: value["placement_sha256"] for key, value in placement_specs.items()
        },
        "expected_runs": 21,
    }, indent=2, sort_keys=True))
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--milestone-i-study", type=Path,
        default=Path("results/raw/milestone-i-repeatability-20260901T091212Z-r16-validated"),
    )
    parser.add_argument(
        "--milestone-k-manifest", type=Path,
        default=Path("results/processed/milestone_k_manifest.json"),
    )
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--order-seed", type=int, default=ORDER_SEED)
    args = parser.parse_args()
    prepare(args)


if __name__ == "__main__":
    main()
