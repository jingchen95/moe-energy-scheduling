#!/usr/bin/env bash
set -euo pipefail

usage() {
    printf 'usage: %s <apply|verify|revert> --root <site-packages-root>\n' "$0" >&2
    exit 2
}

[[ $# -eq 3 && "$2" == "--root" ]] || usage
action="$1"
root="$(cd "$3" && pwd)"
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[[ -f "${root}/vllm/__init__.py" ]] || {
    printf 'not a site-packages root: %s\n' "${root}" >&2
    exit 1
}

python_bin="${PYTHON:-python3}"
phase_state="$({ ROOT="${root}" PYTHON_BIN="${python_bin}" "${python_bin}" - <<'PY'
import os
from pathlib import Path

root = Path(os.environ["ROOT"])
markers = {
    root / "vllm/model_executor/layers/fused_moe/router/base_router.py":
        "eplb_map_to_physical_and_record_phased",
    root / "vllm/model_executor/layers/fused_moe/layer.py":
        "set_eplb_phase_state",
    root / "vllm/distributed/eplb/eplb_state.py":
        "phase_profile_enabled",
    root / "vllm/v1/worker/gpu_worker.py": "phase_loads",
}
states = {path.read_text(encoding="utf-8").count(marker) > 0
          for path, marker in markers.items()}
if states == {False}:
    print("before")
elif states == {True}:
    print("after")
else:
    print("mixed")
PY
})"

apply_phase_patches() {
    patch --batch --forward -d "${root}" -p1 \
        < "${project_root}/patches/vllm-0.15.1-phase-profile.patch"
    patch --batch --forward -d "${root}" -p1 \
        < "${project_root}/patches/vllm-0.15.1-phase-state.patch"
    patch --batch --forward -d "${root}" -p1 \
        < "${project_root}/patches/vllm-0.15.1-phase-worker.patch"
}

revert_phase_patches() {
    patch --batch --reverse -d "${root}" -p1 \
        < "${project_root}/patches/vllm-0.15.1-phase-worker.patch"
    patch --batch --reverse -d "${root}" -p1 \
        < "${project_root}/patches/vllm-0.15.1-phase-state.patch"
    patch --batch --reverse -d "${root}" -p1 \
        < "${project_root}/patches/vllm-0.15.1-phase-profile.patch"
}

case "${action}:${phase_state}" in
    verify:before) printf 'vLLM phase patch state: not applied\n' ;;
    verify:after) printf 'vLLM phase patch state: applied\n' ;;
    verify:mixed)
        printf 'vLLM phase patch state: mixed\n' >&2
        exit 1
        ;;
    apply:before)
        base_state="$(${project_root}/scripts/manage_vllm_patch.sh verify --root "${root}")"
        if [[ "${base_state}" == *'not applied'* ]]; then
            ${project_root}/scripts/manage_vllm_patch.sh apply --root "${root}"
        elif [[ "${base_state}" != *'applied'* ]]; then
            printf 'cannot apply phase patch from base state: %s\n' "${base_state}" >&2
            exit 1
        fi
        apply_phase_patches
        exec "$0" verify --root "${root}"
        ;;
    apply:after) printf 'vLLM phase patch already applied\n' ;;
    apply:mixed) printf 'refusing to apply from mixed phase state\n' >&2; exit 1 ;;
    revert:after)
        revert_phase_patches
        exec "$0" verify --root "${root}"
        ;;
    revert:before) printf 'vLLM phase patch already reverted\n' ;;
    revert:mixed) printf 'refusing to revert from mixed phase state\n' >&2; exit 1 ;;
    *) printf 'unsupported phase patch state: %s\n' "${phase_state}" >&2; exit 1 ;;
esac
