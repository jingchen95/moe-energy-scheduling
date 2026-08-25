#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
destination="${1:-${PROJECT_ROOT}/data/datasets}"
mkdir -p "${destination}"

fetch() {
    local url="$1"
    local output="$2"
    if [[ -e "${output}" ]]; then
        printf 'keeping existing %s\n' "${output}"
        return
    fi
    local temporary="${output}.partial.$$"
    trap 'rm -f "${temporary}"' RETURN
    curl --fail --location --retry 3 --output "${temporary}" "${url}"
    mv "${temporary}" "${output}"
    trap - RETURN
}

fetch \
  https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/main/databricks-dolly-15k.jsonl \
  "${destination}/databricks-dolly-15k.jsonl"
fetch \
  https://raw.githubusercontent.com/google-research/google-research/master/mbpp/mbpp.jsonl \
  "${destination}/mbpp.jsonl"
fetch \
  https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl \
  "${destination}/gsm8k-train.jsonl"

(cd "${destination}" && sha256sum databricks-dolly-15k.jsonl mbpp.jsonl gsm8k-train.jsonl)
