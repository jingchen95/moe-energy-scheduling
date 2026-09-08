#!/usr/bin/env bash
#SBATCH --job-name=vllm-apptainer-check
#SBATCH --partition=debug
#SBATCH --nodelist=uranus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:L40s:1
#SBATCH --time=00:10:00
#SBATCH --output=results/slurm-apptainer-verify-%j.out

set -euo pipefail

project_root="${SLURM_SUBMIT_DIR:?submit this script from the repository root}"
cd "${project_root}"

case "$(hostname -s)" in
    uranus*|neptune*) ;;
    *) printf 'refusing to run outside uranus/neptune\n' >&2; exit 1 ;;
esac

image="${project_root}/containers/vllm-openai-v0.15.1-amd64.sif"
[[ -f "${image}" ]] || {
    printf 'missing Apptainer image: %s\n' "${image}" >&2
    exit 1
}

run_id="apptainer-verify-${SLURM_JOB_ID}-$(date -u +%Y%m%dT%H%M%SZ)"
run_dir="${project_root}/results/raw/${run_id}"
mkdir -p "${project_root}/results/raw"
mkdir "${run_dir}"
exec > >(tee -a "${run_dir}/job.log") 2>&1

status=1
finish() {
    printf '%s\n' "${status}" > "${run_dir}/exit-status.txt"
    if [[ "${status}" -eq 0 ]]; then
        printf 'PASS\n' > "${run_dir}/status.txt"
    else
        printf 'FAIL\n' > "${run_dir}/status.txt"
    fi
}
trap finish EXIT

apptainer --version | tee "${run_dir}/apptainer-version.txt"
sha256sum "${image}" | tee "${run_dir}/image-sha256.txt"
scontrol show job "${SLURM_JOB_ID}" > "${run_dir}/slurm-job.txt"
nvidia-smi -L | tee "${run_dir}/host-gpus.txt"

apptainer exec --nv "${image}" nvidia-smi -L \
    | tee "${run_dir}/container-gpus.txt"

apptainer exec --nv "${image}" python3 - <<'PY' \
    | tee "${run_dir}/container-python.txt"
import inspect
import platform
from pathlib import Path

import aiohttp
import torch
import vllm

assert vllm.__version__ == "0.15.1", vllm.__version__
assert torch.cuda.is_available(), "CUDA is not available inside Apptainer"
assert torch.cuda.device_count() >= 1, "no CUDA device visible inside Apptainer"

print("python", platform.python_version())
print("torch", torch.__version__)
print("torch_cuda", torch.version.cuda)
print("vllm", vllm.__version__)
print("aiohttp", aiohttp.__version__)
print("cuda_available", torch.cuda.is_available())
print("cuda_device_count", torch.cuda.device_count())
for index in range(torch.cuda.device_count()):
    print(f"cuda_device_{index}", torch.cuda.get_device_name(index))
print("site_root", Path(inspect.getfile(vllm)).resolve().parent.parent)
PY

site_root="$(apptainer exec "${image}" python3 - <<'PY'
import inspect
from pathlib import Path
import vllm

print(Path(inspect.getfile(vllm)).resolve().parent.parent)
PY
)"

apptainer exec \
    --bind "${project_root}:/workspace:ro" \
    --pwd /workspace \
    "${image}" \
    env PYTHON=python3 scripts/manage_moe_timing_patch.sh \
        verify --root "${site_root}" \
    | tee "${run_dir}/timing-patch-state.txt"

status=0
