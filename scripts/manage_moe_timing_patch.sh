#!/usr/bin/env bash
set -euo pipefail

usage() {
    printf 'usage: %s <apply|check|verify|revert> --root <site-packages-root>\n' "$0" >&2
    exit 2
}

[[ $# -eq 3 && "$2" == "--root" ]] || usage
action="$1"
root="$(cd "$3" && pwd)"
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
manifest="${project_root}/patches/vllm-0.15.1-moe-timing.json"
patch_file="${project_root}/patches/vllm-0.15.1-moe-timing.patch"
python_bin="${PYTHON:-python3}"

[[ -f "${root}/vllm/__init__.py" ]] || {
    printf 'not a site-packages root: %s\n' "${root}" >&2
    exit 1
}

state="$({
    ROOT="${root}" MANIFEST="${manifest}" PATCH_FILE="${patch_file}" \
        "${python_bin}" - <<'PY'
import hashlib
import json
import os
from pathlib import Path

root = Path(os.environ["ROOT"])
manifest_path = Path(os.environ["MANIFEST"])
patch_path = Path(os.environ["PATCH_FILE"])
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

if manifest.get("patch_version") != 2:
    raise SystemExit("unsupported MoE timing patch manifest version")
if manifest.get("vllm_version") != "0.15.1":
    raise SystemExit("MoE timing patch manifest is not for vLLM 0.15.1")
if manifest.get("patch_file") != patch_path.name:
    raise SystemExit("MoE timing patch filename does not match its manifest")

patch_digest = hashlib.sha256(patch_path.read_bytes()).hexdigest()
if patch_digest != manifest.get("patch_sha256"):
    raise SystemExit("MoE timing patch SHA-256 does not match its manifest")

states = set()
for relative, hashes in manifest["files"].items():
    path = root / relative
    if not path.is_file():
        raise SystemExit(f"missing vLLM file: {relative}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest == hashes["before_sha256"]:
        states.add("before")
    elif digest == hashes["after_sha256"]:
        states.add("after")
    else:
        states.add("unknown")

if states == {"before"}:
    print("before")
elif states == {"after"}:
    print("after")
else:
    print("mixed")
PY
})"

case "${action}:${state}" in
    check:before|verify:before)
        printf 'vLLM MoE timing patch state: not applied\n'
        ;;
    check:after|verify:after)
        printf 'vLLM MoE timing patch state: applied\n'
        ;;
    check:mixed|verify:mixed)
        printf 'vLLM MoE timing patch state: mixed or unrecognized\n' >&2
        exit 1
        ;;
    apply:before)
        patch --batch --forward --dry-run -d "${root}" -p1 < "${patch_file}"
        patch --batch --forward -d "${root}" -p1 < "${patch_file}"
        exec "$0" check --root "${root}"
        ;;
    apply:after)
        printf 'vLLM MoE timing patch already applied\n'
        ;;
    apply:mixed)
        printf 'refusing to apply from mixed or unrecognized timing state\n' >&2
        exit 1
        ;;
    revert:after)
        patch --batch --reverse --dry-run -d "${root}" -p1 < "${patch_file}"
        patch --batch --reverse -d "${root}" -p1 < "${patch_file}"
        exec "$0" check --root "${root}"
        ;;
    revert:before)
        printf 'vLLM MoE timing patch already reverted\n'
        ;;
    revert:mixed)
        printf 'refusing to revert from mixed or unrecognized timing state\n' >&2
        exit 1
        ;;
    *)
        usage
        ;;
esac
