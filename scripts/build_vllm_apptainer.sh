#!/usr/bin/env bash
set -euo pipefail

usage() {
    printf 'usage: %s OUTPUT_SIF [BUILD_TMPDIR]\n' "$0" >&2
    exit 2
}

[[ $# -ge 1 && $# -le 2 ]] || usage

output="$1"
build_tmp="${2:-${SLURM_TMPDIR:-/tmp}}"
image_ref='docker://vllm/vllm-openai@sha256:06f9f0d5c7cb079504615c51dab70cd18abbf609d1358b940172181ac0a92efa'

command -v apptainer >/dev/null || {
    printf 'apptainer is not available in PATH\n' >&2
    exit 1
}
[[ -d "${build_tmp}" && -w "${build_tmp}" ]] || {
    printf 'build temporary directory is not writable: %s\n' "${build_tmp}" >&2
    exit 1
}
[[ ! -e "${output}" ]] || {
    printf 'refusing to overwrite existing image: %s\n' "${output}" >&2
    exit 1
}

mkdir -p "$(dirname "${output}")"

# Do not retain OCI layers as thousands of cache files on shared storage.
# Apptainer expands them only below BUILD_TMPDIR and emits one immutable SIF.
APPTAINER_TMPDIR="${build_tmp}" apptainer build \
    --arch amd64 \
    --disable-cache \
    "${output}" \
    "${image_ref}"

apptainer inspect "${output}"
