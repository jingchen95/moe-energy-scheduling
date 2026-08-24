#!/usr/bin/env bash
set -euo pipefail

NVIDIA_SMI="${NVIDIA_SMI:-nvidia-smi}"
SUDO="${SUDO:-sudo}"
gpu_csv="${1:-0,1,2,3}"
IFS=',' read -r -a gpu_ids <<< "${gpu_csv}"

for gpu_id in "${gpu_ids[@]}"; do
    [[ "${gpu_id}" =~ ^[0-9]+$ ]] || { printf 'invalid GPU index: %s\n' "${gpu_id}" >&2; exit 2; }
    "${SUDO}" "${NVIDIA_SMI}" -i "${gpu_id}" -rgc
done
"${NVIDIA_SMI}" --query-gpu=index,clocks.current.graphics,clocks.current.sm --format=csv,noheader
