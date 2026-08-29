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
patch_file="${project_root}/patches/vllm-0.15.1-moe-timing.patch"
[[ -f "${root}/vllm/__init__.py" ]] || {
    printf 'not a site-packages root: %s\n' "${root}" >&2
    exit 1
}

state="$({ ROOT="${root}" python3 - <<'PY'
import os
from pathlib import Path

root = Path(os.environ["ROOT"])
markers = {
    root / "vllm/model_executor/layers/fused_moe/layer.py":
        "_moe_timing_enabled",
    root / "vllm/v1/worker/gpu_worker.py": "dump_moe_timing",
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

case "${action}:${state}" in
    verify:before) printf 'vLLM MoE timing patch state: not applied\n' ;;
    verify:after) printf 'vLLM MoE timing patch state: applied\n' ;;
    verify:mixed)
        printf 'vLLM MoE timing patch state: mixed\n' >&2
        exit 1
        ;;
    apply:before)
        phase_state="$("${project_root}/scripts/manage_phase_patch.sh" verify --root "${root}")"
        [[ "${phase_state}" == *'applied'* ]] || {
            printf 'MoE timing patch requires the Milestone F phase patch: %s\n' "${phase_state}" >&2
            exit 1
        }
        patch --batch --forward -d "${root}" -p1 < "${patch_file}"
        exec "$0" verify --root "${root}"
        ;;
    apply:after) printf 'vLLM MoE timing patch already applied\n' ;;
    apply:mixed) printf 'refusing to apply from mixed timing state\n' >&2; exit 1 ;;
    revert:after)
        patch --batch --reverse -d "${root}" -p1 < "${patch_file}"
        exec "$0" verify --root "${root}"
        ;;
    revert:before) printf 'vLLM MoE timing patch already reverted\n' ;;
    revert:mixed) printf 'refusing to revert from mixed timing state\n' >&2; exit 1 ;;
    *) printf 'unsupported timing patch state: %s\n' "${state}" >&2; exit 1 ;;
esac
