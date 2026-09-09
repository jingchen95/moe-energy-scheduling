#!/usr/bin/env bash

# A short CUDA Graph diagnostic run. It profiles three fixed requests with
# Nsight Systems and estimates the fraction of CUDA kernel time related to MoE.
# This is not the primary E2E benchmark.
#SBATCH --job-name=moe-nsys-profile
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
#SBATCH --output=results/raw/slurm-moe-nsys-%j.out

set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"

job_id="${SLURM_JOB_ID:?run this script inside a four-GPU Slurm allocation}"
host="$(hostname -s)"
case "${host}" in
    uranus*|neptune*) run_scope="approved-study-node" ;;
    vera-*) run_scope="environment-smoke-only-not-comparable" ;;
    *) printf 'refusing to run on unsupported host: %s\n' "${host}" >&2; exit 1 ;;
esac

gpu_count="$(nvidia-smi -L | wc -l)"
[[ "${gpu_count}" -eq 4 ]] || {
    printf 'expected exactly four visible GPUs, found %s\n' "${gpu_count}" >&2
    exit 1
}

# CUDA is loaded only to provide the host-side nsys executable. The model
# runtime, CUDA user-space libraries, PyTorch, and vLLM all come from the SIF.
module load CUDA/12.9.1
command -v nsys >/dev/null || {
    printf 'nsys is unavailable after loading CUDA/12.9.1\n' >&2
    exit 1
}
command -v apptainer >/dev/null || {
    printf 'apptainer is not available\n' >&2
    exit 1
}
# Run nsys inside the clean container so injection reaches the CUDA workers.
nsys_install="${EBROOTCUDA}/nsight-systems-2025.1.3"
[[ -x "${nsys_install}/target-linux-x64/nsys" ]] || {
    printf 'missing Nsight installation: %s\n' "${nsys_install}" >&2
    exit 1
}

image="${project_root}/containers/vllm-openai-v0.15.1-amd64.sif"
expected_image_sha="9eef28a71be2c24668f3bf744e3f38f9cbb4f2be5c3467c07d4502b42b55b96e"
model_dir_input="${1:-${project_root}/containers/models/Qwen3-30B-A3B}"
[[ -f "${image}" ]] || {
    printf 'missing Apptainer image: %s\n' "${image}" >&2
    exit 1
}
actual_image_sha="$(sha256sum "${image}" | awk '{print $1}')"
[[ "${actual_image_sha}" == "${expected_image_sha}" ]] || {
    printf 'unexpected Apptainer image SHA-256: %s\n' "${actual_image_sha}" >&2
    exit 1
}
[[ -d "${model_dir_input}" ]] || {
    printf 'missing model directory: %s\n' "${model_dir_input}" >&2
    exit 1
}
model_dir="$(cd "${model_dir_input}" && pwd)"
for required in config.json tokenizer.json model.safetensors.index.json; do
    [[ -s "${model_dir}/${required}" ]] || {
        printf 'missing model file: %s\n' "${model_dir}/${required}" >&2
        exit 1
    }
done
shard_count="$(find "${model_dir}" -maxdepth 1 -type f \
    -name 'model-*.safetensors' | wc -l)"
[[ "${shard_count}" -eq 16 ]] || {
    printf 'expected 16 model shards, found %s\n' "${shard_count}" >&2
    exit 1
}

output_root="${project_root}/results/raw"
run_id="moe-nsys-${job_id}-$(date -u +%Y%m%dT%H%M%SZ)"
run_dir="${output_root}/${run_id}"
container_benchmark_dir="/workspace/results/raw/${run_id}/benchmark"
profile_dir="${run_dir}/profile"
pid_file="${run_dir}/vllm.pid"
temporary_root="${SLURM_TMPDIR:-/tmp}"
runtime_cache="${temporary_root}/vllm-nsys-cache-${job_id}-${run_id}"
mkdir -p "${output_root}"
mkdir "${run_dir}"
mkdir "${profile_dir}"
mkdir "${runtime_cache}"

exec > >(tee -a "${run_dir}/job.log") 2>&1

server_pid=""
profiler_pid=""
benchmark_pid=""

cleanup() {
    local status=$?
    local cleanup_failed=0
    trap - EXIT INT TERM

    if [[ -n "${benchmark_pid}" ]] && kill -0 "${benchmark_pid}" 2>/dev/null; then
        kill -TERM "${benchmark_pid}" 2>/dev/null || true
        for _ in {1..10}; do
            kill -0 "${benchmark_pid}" 2>/dev/null || break
            sleep 1
        done
        kill -KILL "${benchmark_pid}" 2>/dev/null || true
    fi
    if [[ -n "${server_pid}" ]]; then
        scripts/stop_vllm.sh "${pid_file}" || cleanup_failed=1
    fi
    if [[ -n "${profiler_pid}" ]]; then
        scripts/stop_vllm.sh "${run_dir}/nsys.pid" || cleanup_failed=1
    fi
    if [[ "${status}" -eq 0 && "${cleanup_failed}" -ne 0 ]]; then
        status=1
    fi
    printf '%s\n' "${status}" > "${run_dir}/exit-status.txt"
    if [[ "${status}" -eq 0 ]]; then
        printf 'PASS\n' > "${run_dir}/status.txt"
    else
        printf 'FAIL\n' > "${run_dir}/status.txt"
    fi
    exit "${status}"
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

printf 'run scope: %s\n' "${run_scope}"
printf 'run directory: %s\n' "${run_dir}"
printf '%s\n' "${run_scope}" > "${run_dir}/run-scope.txt"
if [[ "${run_scope}" == environment-smoke-* ]]; then
    printf 'WARNING: Vera results are smoke-only and must not be compared with Uranus/Neptune study data.\n'
fi

trace="data/traces/gsm8k-50-r0.5.jsonl"
expected_trace_sha="721638a6250b929de05c2c355c8bc9b20258db7b28fedb94acb5572136571cbc"
actual_trace_sha="$(sha256sum "${trace}" | awk '{print $1}')"
[[ "${actual_trace_sha}" == "${expected_trace_sha}" ]] || {
    printf 'unexpected trace hash: %s\n' "${actual_trace_sha}" >&2
    exit 1
}

cp "${trace}" "${run_dir}/trace.jsonl"
git rev-parse HEAD > "${run_dir}/git-commit.txt"
git status --short > "${run_dir}/git-status.txt"
reproducibility_paths=(
    scripts/moe_nsys_profile.sbatch
    scripts/stop_vllm.sh
    src/nsys_profile.py
    src/nsys_moe_share.py
    src/nsys_target.py
    src/nsys_probe.py
    src/benchmark.py
    src/metrics.py
    src/workload.py
    "${trace}"
)
git status --short -- "${reproducibility_paths[@]}" \
    > "${run_dir}/relevant-git-status.txt"
[[ ! -s "${run_dir}/relevant-git-status.txt" ]] || {
    printf 'refusing to profile with uncommitted relevant source files\n' >&2
    exit 1
}
git diff --no-ext-diff HEAD -- "${reproducibility_paths[@]}" \
    > "${run_dir}/source-diff.patch"
scontrol show job "${job_id}" > "${run_dir}/slurm-job.txt"
nvidia-smi -L > "${run_dir}/gpus.txt"
nvidia-smi topo -m > "${run_dir}/gpu-topology.txt"
nsys --version > "${run_dir}/nsys-version.txt"
apptainer --version > "${run_dir}/apptainer-version.txt"
sha256sum "${image}" > "${run_dir}/image-sha256.txt"
module -t list > "${run_dir}/modules.txt" 2>&1
apptainer exec \
    --cleanenv \
    "${image}" \
    python3 - <<'PY' > "${run_dir}/versions.txt"
import platform
import torch
import vllm

print("python", platform.python_version())
print("torch", torch.__version__)
print("torch_cuda", torch.version.cuda)
print("vllm", vllm.__version__)
PY

if curl --fail --silent --max-time 2 http://127.0.0.1:8000/health \
    >/dev/null 2>&1; then
    printf 'refusing to reuse an existing service on 127.0.0.1:8000\n' >&2
    exit 1
fi

container_command=(
    apptainer exec
    --cleanenv
    --nv
    --bind "${project_root}:/workspace:ro"
    --bind "${model_dir}:/model:ro"
    --bind "${runtime_cache}:/runtime-cache"
    --bind "${run_dir}:/run-output"
    --bind "${nsys_install}:/opt/moe-nsys:ro"
    --pwd /workspace
    --env CUDA_VISIBLE_DEVICES=0,1,2,3
    --env VLLM_SERVER_DEV_MODE=1
    --env VLLM_WORKER_MULTIPROC_METHOD=spawn
    --env PYTHONUNBUFFERED=1
    --env PYTHONDONTWRITEBYTECODE=1
    --env HF_HUB_OFFLINE=1
    --env TRANSFORMERS_OFFLINE=1
    --env XDG_CACHE_HOME=/runtime-cache
    --env TRITON_CACHE_DIR=/runtime-cache/triton
    --env TORCHINDUCTOR_CACHE_DIR=/runtime-cache/torchinductor
    "${image}"
    env -u BASH_ENV
)

vllm_command=(
    python3 -m vllm.entrypoints.openai.api_server
    --model /model
    --served-model-name qwen3-30b-a3b
    --host 127.0.0.1
    --port 8000
    --dtype auto
    --max-model-len 4096
    --gpu-memory-utilization 0.9
    --seed 0
    --tensor-parallel-size 1
    --data-parallel-size 4
    --data-parallel-backend mp
    --enable-expert-parallel
    --expert-placement-strategy linear
    --all2all-backend allgather_reducescatter
    --no-enable-prefix-caching
    --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}'
    --profiler-config '{"profiler":"cuda"}'
)

nsys_command=(
    /opt/moe-nsys/target-linux-x64/nsys profile
    --trace=cuda,nvtx
    --sample=none
    --cpuctxsw=none
    --trace-fork-before-exec=true
    --cuda-graph-trace=node
    --cuda-event-trace=false
    --capture-range=cudaProfilerApi
    --capture-range-end=stop
    --kill=none
    --wait=all
    --force-overwrite=false
)

printf '%q ' "${vllm_command[@]}" > "${run_dir}/vllm-command.txt"
printf '\n' >> "${run_dir}/vllm-command.txt"
printf '%q ' "${nsys_command[@]}" > "${run_dir}/nsys-command.txt"
printf '\n' >> "${run_dir}/nsys-command.txt"
printf '%q ' "${container_command[@]}" > "${run_dir}/container-command.txt"
printf '\n' >> "${run_dir}/container-command.txt"
printf '%q ' "${container_command[@]}" "${nsys_command[@]}" \
    --output=/run-output/profile/moe-cudagraph \
    python3 -m src.nsys_target /run-output/vllm.pid /run-output/vllm.log \
    "${vllm_command[@]}" > "${run_dir}/launch-command.txt"
printf '\n' >> "${run_dir}/launch-command.txt"

# Fail before loading weights if CUDA Graph tracing/export is unavailable.
printf 'Checking Nsight capture with four small CUDA Graph workers...\n'
setsid "${container_command[@]}" "${nsys_command[@]}" \
    --output=/run-output/profile/probe python3 -m src.nsys_probe \
    > "${run_dir}/nsys-probe.log" 2>&1 &
profiler_pid=$!
printf '%s\n' "${profiler_pid}" > "${run_dir}/nsys.pid"
probe_status=0
wait "${profiler_pid}" || probe_status=$?
profiler_pid=""
if [[ "${probe_status}" -ne 0 || ! -s "${profile_dir}/probe.nsys-rep" ]]; then
    tail -n 50 "${run_dir}/nsys-probe.log" >&2
    printf 'Nsight probe failed before loading model weights\n' >&2
    exit 1
fi
nsys export --type sqlite --output "${profile_dir}/probe.sqlite" \
    "${profile_dir}/probe.nsys-rep" > "${run_dir}/nsys-probe-export.log" 2>&1

verify_devices() {
    python3 - "$1" <<'PY'
import sqlite3
import sys
from pathlib import Path

with sqlite3.connect(Path(sys.argv[1]).resolve().as_uri() + "?mode=ro", uri=True) as db:
    rows = db.execute(
        "SELECT deviceId, count(*) FROM CUPTI_ACTIVITY_KIND_KERNEL GROUP BY deviceId"
    ).fetchall()
if len(rows) != 4:
    raise SystemExit(f"expected CUDA kernels from 4 GPUs, observed {rows}")
print(f"CUDA kernel capture verified: {rows}")
PY
}
verify_devices "${profile_dir}/probe.sqlite"

printf 'Starting model; progress is in %s/vllm.log\n' "${run_dir}"
setsid "${container_command[@]}" "${nsys_command[@]}" \
    --output=/run-output/profile/moe-cudagraph \
    python3 -m src.nsys_target /run-output/vllm.pid /run-output/vllm.log \
    "${vllm_command[@]}" > "${run_dir}/nsys.log" 2>&1 &
profiler_pid=$!
printf '%s\n' "${profiler_pid}" > "${run_dir}/nsys.pid"
server_pid="pending"

ready=0
for attempt in $(seq 1 360); do
    if ! kill -0 "${profiler_pid}" 2>/dev/null; then
        printf 'profiled vLLM exited before becoming ready\n' >&2
        tail -n 50 "${run_dir}/nsys.log" >&2 || true
        tail -n 120 "${run_dir}/vllm.log" >&2 || true
        exit 1
    fi
    if curl --fail --silent --show-error --max-time 2 \
        http://127.0.0.1:8000/health >/dev/null 2>&1; then
        ready=1
        break
    fi
    if (( attempt % 6 == 0 )); then
        printf 'Waiting for model startup (%s seconds)...\n' "$((attempt * 10))"
        tail -n 2 "${run_dir}/vllm.log" || true
    fi
    sleep 10
done
[[ "${ready}" -eq 1 ]] || {
    printf 'timed out waiting for vLLM\n' >&2
    exit 1
}

apptainer exec \
    --cleanenv \
    --bind "${project_root}:/workspace" \
    --pwd /workspace \
    "${image}" \
    env -u BASH_ENV PYTHONDONTWRITEBYTECODE=1 \
    MOE_PROFILE_RUN_SCOPE="${run_scope}" \
    python3 -m src.nsys_profile "${container_benchmark_dir}" \
    > "${run_dir}/benchmark.log" 2>&1 &
benchmark_pid=$!
printf '%s\n' "${benchmark_pid}" > "${run_dir}/benchmark.pid"
benchmark_status=0
wait "${benchmark_pid}" || benchmark_status=$?
benchmark_pid=""
[[ "${benchmark_status}" -eq 0 ]] || exit "${benchmark_status}"

# Stop only the model group. Nsight survives to finish exporting the report.
scripts/stop_vllm.sh "${pid_file}"
server_pid=""
printf 'Requests complete; waiting for Nsight report export...\n'
for attempt in $(seq 1 180); do
    kill -0 "${profiler_pid}" 2>/dev/null || break
    sleep 1
done
if kill -0 "${profiler_pid}" 2>/dev/null; then
    printf 'Nsight export timed out; inspect %s/nsys.log\n' "${run_dir}" >&2
    exit 1
fi
profiler_status=0
wait "${profiler_pid}" || profiler_status=$?
profiler_pid=""
printf '%s\n' "${profiler_status}" > "${run_dir}/nsys-exit-status.txt"
[[ "${profiler_status}" -eq 0 ]] || {
    tail -n 50 "${run_dir}/nsys.log" >&2
    exit "${profiler_status}"
}

mapfile -t reports < <(
    find "${profile_dir}" -maxdepth 1 -type f -name 'moe-cudagraph*.nsys-rep' -print | sort
)
[[ "${#reports[@]}" -gt 0 ]] || {
    printf 'nsys produced no .nsys-rep file\n' >&2
    tail -n 50 "${run_dir}/nsys.log" >&2
    exit 1
}

stats_files=()
for report in "${reports[@]}"; do
    report_name="$(basename "${report}" .nsys-rep)"
    stats_path="${profile_dir}/${report_name}-cuda-kernels.csv"
    nsys stats --report cuda_gpu_kern_sum --format csv "${report}" \
        > "${stats_path}" \
        2> "${profile_dir}/${report_name}-stats.stderr"
    verify_devices "${profile_dir}/${report_name}.sqlite"
    stats_files+=("${stats_path}")
done

MOE_PROFILE_RUN_SCOPE="${run_scope}" python3 -m src.nsys_moe_share \
    "${run_dir}/moe-kernel-share.json" "${stats_files[@]}" \
    > "${run_dir}/moe-kernel-share.log"

printf '\n=== CONSERVATIVE MOE CUDA KERNEL-TIME SHARE ===\n'
python3 - "${run_dir}/moe-kernel-share.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    result = json.load(stream)
print(f"{result['conservative_moe_cuda_kernel_time_share_percent']:.3f}%")
print("See moe-kernel-share.json and inspect largest_unmatched_kernels.")
PY
