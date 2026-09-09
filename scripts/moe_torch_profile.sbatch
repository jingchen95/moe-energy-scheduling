#!/usr/bin/env bash
# Short Torch Profiler diagnostic; CUDA Graph/compile remain enabled.
#SBATCH --job-name=moe-torch-profile
#SBATCH --partition=debug
#SBATCH --nodelist=uranus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --gres=gpu:L40s:4
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --signal=B:TERM@180
#SBATCH --output=results/raw/slurm-moe-torch-%j.out

set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"
job_id="${SLURM_JOB_ID:?run inside an existing four-GPU Slurm allocation}"
host="$(hostname -s)"
case "${host}" in
    uranus*|neptune*) run_scope="approved-study-node" ;;
    vera-*) run_scope="environment-smoke-only-not-comparable" ;;
    *) printf 'refusing to run on unsupported host: %s\n' "${host}" >&2; exit 1 ;;
esac
command -v apptainer >/dev/null
gpu_count="$(nvidia-smi -L | wc -l)"
[[ "${gpu_count}" -eq 4 ]] || {
    printf 'expected exactly four visible GPUs, found %s\n' "${gpu_count}" >&2
    exit 1
}

# No Conda, Nsight, site-packages patch, or host CUDA module is needed.
image="${project_root}/containers/vllm-openai-v0.15.1-amd64.sif"
expected_image_sha="9eef28a71be2c24668f3bf744e3f38f9cbb4f2be5c3467c07d4502b42b55b96e"
model_dir_input="${1:-${project_root}/containers/models/Qwen3-30B-A3B}"
[[ -f "${image}" && -d "${model_dir_input}" ]] || {
    printf 'missing pinned SIF or local model directory\n' >&2
    exit 1
}
printf 'Verifying pinned container image...\n'
actual_image_sha="$(sha256sum "${image}" | awk '{print $1}')"
[[ "${actual_image_sha}" == "${expected_image_sha}" ]] || {
    printf 'unexpected SIF SHA-256: %s\n' "${actual_image_sha}" >&2
    exit 1
}
model_dir="$(cd "${model_dir_input}" && pwd)"
for required in config.json tokenizer.json model.safetensors.index.json; do
    [[ -s "${model_dir}/${required}" ]] || {
        printf 'missing model file: %s\n' "${required}" >&2
        exit 1
    }
done
shard_count="$(find "${model_dir}" -maxdepth 1 -type f -name 'model-*.safetensors' | wc -l)"
[[ "${shard_count}" -eq 16 ]] || {
    printf 'expected 16 model shards, found %s\n' "${shard_count}" >&2
    exit 1
}

run_id="moe-torch-${job_id}-$(date -u +%Y%m%dT%H%M%SZ)"
run_dir="${project_root}/results/raw/${run_id}"
# Compute-node /tmp may be only 1 GiB. Keep all large caches and export
# temporaries on the same spacious filesystem as this unique result directory.
runtime_cache="${run_dir}/runtime-cache"
mkdir -p "${project_root}/results/raw"
mkdir "${run_dir}"
mkdir "${run_dir}/profile" "${run_dir}/tmp" "${runtime_cache}"
exec > >(tee -a "${run_dir}/job.log") 2>&1

server_pid=""
benchmark_pid=""
record_clocks() {
    nvidia-smi --query-gpu=index,uuid,driver_version,pstate,clocks.current.graphics,clocks.current.memory,memory.used \
        --format=csv
}
record_storage() {
    df -h "${run_dir}/profile" "${runtime_cache}" "${run_dir}/tmp" /tmp
    df -i "${run_dir}/profile" "${runtime_cache}" "${run_dir}/tmp" /tmp
}
cleanup() {
    local status=$?
    local cleanup_failed=0
    trap - EXIT INT TERM
    if [[ -n "${benchmark_pid}" ]]; then
        if scripts/stop_vllm.sh "${run_dir}/benchmark.pid"; then
            wait "${benchmark_pid}" 2>/dev/null || true
        else
            cleanup_failed=1
        fi
    fi
    if [[ -n "${server_pid}" ]]; then
        if scripts/stop_vllm.sh "${run_dir}/vllm.pid"; then
            wait "${server_pid}" 2>/dev/null || true
        else
            cleanup_failed=1
        fi
    fi
    record_clocks > "${run_dir}/clocks-end.csv" 2>&1 || true
    record_storage > "${run_dir}/storage-end.txt" 2>&1 || true
    if [[ "${status}" -eq 0 && "${cleanup_failed}" -ne 0 ]]; then status=1; fi
    printf '%s\n' "${status}" > "${run_dir}/exit-status.txt"
    if [[ "${status}" -eq 0 ]]; then
        printf 'PASS\n' > "${run_dir}/status.txt"
    else
        printf 'FAIL\n' > "${run_dir}/status.txt"
        printf 'FAILED. Raw traces/logs were retained in %s\n' "${run_dir}" >&2
        tail -n 25 "${run_dir}/benchmark.log" 2>/dev/null || true
        tail -n 35 "${run_dir}/vllm.log" 2>/dev/null || true
    fi
    exit "${status}"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

printf 'run scope: %s\nrun directory: %s\n' "${run_scope}" "${run_dir}"
printf 'Cache/temp files: %s/{runtime-cache,tmp}; trace compression disabled.\n' "${run_dir}"
record_storage > "${run_dir}/storage-start.txt" 2>&1
printf '%s\n' "${run_scope}" > "${run_dir}/run-scope.txt"
if [[ "${run_scope}" == environment-smoke-* ]]; then
    printf 'WARNING: Vera results are smoke-only; do not compare with Uranus/Neptune study data.\n'
fi
trace="data/traces/gsm8k-50-r0.5.jsonl"
expected_trace_sha="721638a6250b929de05c2c355c8bc9b20258db7b28fedb94acb5572136571cbc"
[[ "$(sha256sum "${trace}" | awk '{print $1}')" == "${expected_trace_sha}" ]] || {
    printf 'frozen trace hash mismatch\n' >&2; exit 1
}
cp "${trace}" "${run_dir}/trace.jsonl"
git rev-parse HEAD > "${run_dir}/git-commit.txt"
git status --short > "${run_dir}/git-status.txt"
reproducibility_paths=(
    scripts/moe_torch_profile.sbatch scripts/stop_vllm.sh
    src/torch_profile.py src/torch_moe_share.py src/nsys_profile.py
    src/nsys_moe_share.py src/benchmark.py src/metrics.py src/moe_timing.py
    src/workload.py "${trace}"
)
git status --short -- "${reproducibility_paths[@]}" > "${run_dir}/relevant-git-status.txt"
[[ ! -s "${run_dir}/relevant-git-status.txt" ]] || {
    printf 'refusing to profile with uncommitted relevant source files\n' >&2
    exit 1
}
git diff --no-ext-diff HEAD -- "${reproducibility_paths[@]}" > "${run_dir}/source-diff.patch"
scontrol show job "${job_id}" > "${run_dir}/slurm-job.txt"
nvidia-smi -L > "${run_dir}/gpus.txt"
nvidia-smi topo -m > "${run_dir}/gpu-topology.txt"
record_clocks > "${run_dir}/clocks-start.csv"
apptainer --version > "${run_dir}/apptainer-version.txt"
printf '%s  %s\n' "${actual_image_sha}" "${image}" > "${run_dir}/image-sha256.txt"
if type module >/dev/null 2>&1; then module -t list > "${run_dir}/modules.txt" 2>&1; fi

profiler_config="$(PYTHONDONTWRITEBYTECODE=1 python3 -m src.torch_profile --config)"
printf '%s\n' "${profiler_config}" > "${run_dir}/profiler-config.json"
container_command=(
    apptainer exec --cleanenv --nv
    --bind "${project_root}:/workspace:ro"
    --bind "${model_dir}:/model:ro"
    --bind "${runtime_cache}:/runtime-cache"
    --bind "${run_dir}:/run-output"
    --pwd /workspace
    --env CUDA_VISIBLE_DEVICES=0,1,2,3
    --env VLLM_SERVER_DEV_MODE=1
    --env VLLM_WORKER_MULTIPROC_METHOD=spawn
    --env PYTHONUNBUFFERED=1
    --env PYTHONDONTWRITEBYTECODE=1
    --env PYTHONFAULTHANDLER=1
    --env HF_HUB_OFFLINE=1
    --env TRANSFORMERS_OFFLINE=1
    --env XDG_CACHE_HOME=/runtime-cache
    --env TRITON_CACHE_DIR=/runtime-cache/triton
    --env TORCHINDUCTOR_CACHE_DIR=/runtime-cache/torchinductor
    --env TMPDIR=/run-output/tmp
    --env TMP=/run-output/tmp
    --env TEMP=/run-output/tmp
    "${image}" env -u BASH_ENV
)
printf 'Checking container cache/temp/trace storage (at least 5 GiB free)...\n'
"${container_command[@]}" python3 -m src.torch_profile --check-storage \
    > "${run_dir}/storage-preflight.json" 2> "${run_dir}/storage-preflight.log" || {
        tail -n 25 "${run_dir}/storage-preflight.log" >&2; exit 1
    }
# Verify profiler configuration/support using the actual container before loading weights.
"${container_command[@]}" python3 - "${profiler_config}" <<'PY' > "${run_dir}/versions.txt" 2>&1
import json
import platform
import sys
import torch
import vllm
from vllm.config import ProfilerConfig
print("python", platform.python_version())
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("vllm", vllm.__version__)
print("profiler_config", ProfilerConfig(**json.loads(sys.argv[1])))
assert torch.profiler.kineto_available(), "PyTorch Kineto profiler is unavailable"
assert torch.cuda.device_count() == 4, "container must see four GPUs"
PY

if curl --fail --silent --max-time 2 http://127.0.0.1:8000/health >/dev/null 2>&1; then
    printf 'refusing to reuse existing service on port 8000\n' >&2; exit 1
fi
vllm_command=(
    python3 -m vllm.entrypoints.openai.api_server
    --model /model --served-model-name qwen3-30b-a3b
    --host 127.0.0.1 --port 8000 --dtype auto --max-model-len 4096
    --gpu-memory-utilization 0.9 --seed 0
    --tensor-parallel-size 1 --data-parallel-size 4 --data-parallel-backend mp
    --enable-expert-parallel --expert-placement-strategy linear
    --all2all-backend allgather_reducescatter --no-enable-prefix-caching
    --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}'
    --profiler-config "${profiler_config}"
)
printf '%q ' "${vllm_command[@]}" > "${run_dir}/vllm-command.txt"
printf '\n' >> "${run_dir}/vllm-command.txt"
printf '%q ' "${container_command[@]}" "${vllm_command[@]}" > "${run_dir}/launch-command.txt"
printf '\n' >> "${run_dir}/launch-command.txt"
printf 'Starting model with Torch Profiler; CUDA Graph remains enabled...\n'
setsid "${container_command[@]}" "${vllm_command[@]}" > "${run_dir}/vllm.log" 2>&1 &
server_pid=$!
printf '%s\n' "${server_pid}" > "${run_dir}/vllm.pid"

ready=0
for attempt in $(seq 1 360); do
    kill -0 "${server_pid}" 2>/dev/null || {
        printf 'vLLM exited during startup\n' >&2; exit 1
    }
    if curl --fail --silent --max-time 2 http://127.0.0.1:8000/health >/dev/null 2>&1; then
        ready=1; break
    fi
    if (( attempt % 6 == 0 )); then
        printf 'Waiting for model startup (~%s seconds)...\n' "$((attempt * 10))"
        tail -n 2 "${run_dir}/vllm.log" || true
    fi
    sleep 10
done
[[ "${ready}" -eq 1 ]] || { printf 'model startup timed out\n' >&2; exit 1; }

printf 'Model ready. Warming up, then profiling three requests...\n'
setsid apptainer exec --cleanenv \
    --bind "${project_root}:/workspace:ro" --bind "${run_dir}:/run-output" \
    --env TMPDIR=/run-output/tmp --env TMP=/run-output/tmp --env TEMP=/run-output/tmp \
    --pwd /workspace "${image}" \
    env -u BASH_ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    MOE_PROFILE_RUN_SCOPE="${run_scope}" \
    python3 -m src.torch_profile /run-output/benchmark > "${run_dir}/benchmark.log" 2>&1 &
benchmark_pid=$!
printf '%s\n' "${benchmark_pid}" > "${run_dir}/benchmark.pid"
attempt=0
while kill -0 "${benchmark_pid}" 2>/dev/null; do
    sleep 5
    attempt=$((attempt + 1))
    if (( attempt % 6 == 0 )); then
        printf 'Client/profiler still running; latest progress:\n'
        tail -n 2 "${run_dir}/benchmark.log" || true
    fi
done
benchmark_status=0
wait "${benchmark_pid}" || benchmark_status=$?
benchmark_pid=""
printf '%s\n' "${benchmark_status}" > "${run_dir}/benchmark-exit-status.txt"
[[ "${benchmark_status}" -eq 0 ]] || exit "${benchmark_status}"

# stop_profile returns after in-worker trace export. Never stop the model before it.
if grep -Eq 'Failed to (start|stop) profiler|Failed to (open|write to) log file' "${run_dir}/vllm.log"; then
    printf 'Profiler start/stop/export failed despite HTTP status; inspect vllm.log and storage-*.txt. Raw output retained.\n' >&2
    exit 1
fi
scripts/stop_vllm.sh "${run_dir}/vllm.pid"
wait "${server_pid}" 2>/dev/null || true
server_pid=""
printf 'Checking all four GPU traces and calculating approximate MoE kernel share...\n'
PYTHONDONTWRITEBYTECODE=1 python3 -m src.torch_moe_share "${run_dir}" \
    > "${run_dir}/moe-kernel-share.log" 2>&1 || {
        tail -n 30 "${run_dir}/moe-kernel-share.log" >&2; exit 1
    }
python3 - "${run_dir}/moe-kernel-share.json" <<'PY'
import json
import sys
with open(sys.argv[1]) as stream:
    result = json.load(stream)
print(f"Approximate MoE GPU kernel-time share: {result['conservative_moe_cuda_kernel_time_share_percent']:.3f}%")
print(f"Classified MoE: {result['classified_moe_kernel_time_ms']:.3f} GPU-ms; all kernels: {result['total_cuda_kernel_time_ms']:.3f} GPU-ms")
print("This is NOT an E2E wall-time fraction. Inspect classification.largest_unmatched_kernels.")
print(f"Results: {sys.argv[1]}")
PY
