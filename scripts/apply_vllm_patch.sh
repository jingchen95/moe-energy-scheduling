#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
    printf 'usage: %s <vllm-source-root> <patch-file>\n' "$0" >&2
    exit 2
fi

source_root="$(cd "$1" && pwd)"
patch_file="$(cd "$(dirname "$2")" && pwd)/$(basename "$2")"
[[ -f "${source_root}/vllm/__init__.py" ]] || { printf 'not a vLLM source tree: %s\n' "${source_root}" >&2; exit 1; }
[[ -f "${patch_file}" ]] || { printf 'patch not found: %s\n' "${patch_file}" >&2; exit 1; }

git -C "${source_root}" apply --check "${patch_file}"
git -C "${source_root}" apply "${patch_file}"
