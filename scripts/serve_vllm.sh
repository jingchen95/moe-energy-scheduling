#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_CONFIG="${MODEL_CONFIG:-${PROJECT_ROOT}/configs/model_qwen3_30b_a3b.yaml}"

if [[ "${1:-}" == "--config" ]]; then
    [[ $# -ge 2 ]] || { printf 'missing value for --config\n' >&2; exit 2; }
    MODEL_CONFIG="$2"
    shift 2
fi

cd "${PROJECT_ROOT}"
printf 'Resolved vLLM command:\n'
python -m src.config "${MODEL_CONFIG}" -- "$@"
exec python -m src.config --execute "${MODEL_CONFIG}" -- "$@"
