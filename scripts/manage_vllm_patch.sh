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
manifest="${project_root}/patches/vllm-0.15.1-placement-profile.json"
patch_file="${project_root}/patches/vllm-0.15.1-placement-profile.patch"
[[ -f "${root}/vllm/__init__.py" ]] || { printf 'not a site-packages root: %s\n' "${root}" >&2; exit 1; }

python_bin="${PYTHON:-python3}"
state="$({ ROOT="${root}" MANIFEST="${manifest}" "${python_bin}" - <<'PY'
import hashlib, json, os
from pathlib import Path

root = Path(os.environ["ROOT"])
manifest = json.loads(Path(os.environ["MANIFEST"]).read_text())
states = set()
for relative, hashes in manifest["files"].items():
    digest = hashlib.sha256((root / relative).read_bytes()).hexdigest()
    if digest == hashes["before"]:
        states.add("before")
    elif digest == hashes["after"]:
        states.add("after")
    else:
        states.add("unknown")
if states == {"before"}:
    print("before")
elif states == {"after"}:
    print("after")
else:
    print("mixed:" + ",".join(sorted(states)))
PY
} )"

case "${action}:${state}" in
    verify:before) printf 'vLLM patch state: not applied\n' ;;
    verify:after) printf 'vLLM patch state: applied\n' ;;
    apply:before)
        patch --batch --forward -d "${root}" -p1 < "${patch_file}"
        exec "$0" verify --root "${root}"
        ;;
    apply:after) printf 'vLLM patch already applied\n' ;;
    revert:after)
        patch --batch --reverse -d "${root}" -p1 < "${patch_file}"
        exec "$0" verify --root "${root}"
        ;;
    revert:before) printf 'vLLM patch already reverted\n' ;;
    *) printf 'refusing %s from unrecognized patch state: %s\n' "${action}" "${state}" >&2; exit 1 ;;
esac
