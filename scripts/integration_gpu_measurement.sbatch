#!/usr/bin/env bash
#SBATCH --job-name=moe-gpu-integration
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:L40s:4
#SBATCH --time=00:10:00
#SBATCH --output=results/raw/slurm-gpu-integration-%j.out

set -euo pipefail
PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
cd "${PROJECT_ROOT}"
source "${HOME}/miniforge3/etc/profile.d/conda.sh"
conda activate cuda-env
allocated_gpus="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')"
[[ "${allocated_gpus}" -eq 4 ]] || { printf 'expected 4 allocated GPUs, found %s\n' "${allocated_gpus}" >&2; exit 1; }
run_dir="results/raw/gpu-integration-${SLURM_JOB_ID}-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "${run_dir}"

cleanup() {
    status=$?
    trap - EXIT INT TERM
    python -m src.gpu_control reset >> "${run_dir}/reset.log" 2>&1 || true
    exit "${status}"
}
trap cleanup EXIT INT TERM

python -m src.gpu_control list > "${run_dir}/supported-clocks.json"
python -m src.gpu_monitor --duration-s 3 --interval-s 0.2 \
    --output "${run_dir}/idle-gpu-samples.csv" > "${run_dir}/energy-sanity.json"
nonmax="$(python - <<'PY'
from src.gpu_control import supported_graphics_clocks
values = supported_graphics_clocks(0)
print(values[max(0, len(values) // 3)])
PY
)"
python -m src.gpu_control set-all --clocks "${nonmax},${nonmax},${nonmax},${nonmax}" \
    > "${run_dir}/nonmax-readback.json"
python -m src.gpu_control reset > "${run_dir}/reset-readback.json"
