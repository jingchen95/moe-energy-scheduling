#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    printf 'usage: %s <site-packages-root>\n' "$0" >&2
    exit 2
fi
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${PROJECT_ROOT}/scripts/manage_vllm_patch.sh" apply --root "$1"
