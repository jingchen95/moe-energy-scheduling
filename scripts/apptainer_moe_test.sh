#!/usr/bin/env bash
set -euo pipefail

usage() {
    printf 'usage: %s [MODEL_DIRECTORY]\n' "$0" >&2
    exit 2
}

[[ $# -le 1 ]] || usage

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"

image="${project_root}/containers/vllm-openai-v0.15.1-amd64.sif"
model_dir_input="${1:-${project_root}/containers/models/Qwen3-30B-A3B}"
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

[[ -f "${image}" ]] || {
    printf 'missing Apptainer image: %s\n' "${image}" >&2
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

if curl --fail --silent --max-time 2 http://127.0.0.1:8000/health \
    >/dev/null 2>&1; then
    printf 'refusing to reuse an existing service on 127.0.0.1:8000\n' >&2
    exit 1
fi

started_at="$(date -u +%Y%m%dT%H%M%SZ)"
run_id="apptainer-moe-test-${job_id}-${started_at}"
run_dir="${project_root}/results/raw/${run_id}"
container_run_dir="/workspace/results/raw/${run_id}"
benchmark_dir="${run_dir}/benchmark"
container_benchmark_dir="${container_run_dir}/benchmark"
pid_file="${run_dir}/vllm.pid"

temporary_root="${SLURM_TMPDIR:-/tmp}"
overlay="${temporary_root}/moe-overlay-${job_id}-${started_at}.ext3"
runtime_cache="${temporary_root}/vllm-cache-${job_id}-${started_at}"

mkdir -p "${project_root}/results/raw" "${runtime_cache}"
mkdir "${run_dir}"
exec > >(tee -a "${run_dir}/job.log") 2>&1

server_pid=""
patch_applied=0
cleanup_failed=0

patch_manager() {
    apptainer exec \
        --cleanenv \
        --overlay "${overlay}" \
        --bind "${project_root}:/workspace:ro" \
        --pwd /workspace \
        "${image}" \
        env -u BASH_ENV PYTHON=python3 \
        scripts/manage_moe_timing_patch.sh "$@"
}

cleanup() {
    local original_status=$?
    local final_status="${original_status}"
    local server_stopped=1
    trap - EXIT INT TERM

    if [[ -n "${server_pid}" ]]; then
        if scripts/stop_vllm.sh "${pid_file}"; then
            server_pid=""
        else
            server_stopped=0
            cleanup_failed=1
        fi
    fi

    if [[ "${patch_applied}" -eq 1 && "${server_stopped}" -eq 1 ]]; then
        if patch_manager revert --root "${site_root}"; then
            patch_applied=0
        else
            cleanup_failed=1
        fi
    elif [[ "${patch_applied}" -eq 1 ]]; then
        printf 'patch not reverted because the vLLM process group may still be running\n' >&2
    fi

    if [[ "${final_status}" -eq 0 && "${cleanup_failed}" -ne 0 ]]; then
        final_status=1
    fi
    printf '%s\n' "${final_status}" > "${run_dir}/exit-status.txt"
    if [[ "${final_status}" -eq 0 ]]; then
        printf 'PASS\n' > "${run_dir}/status.txt"
    else
        printf 'FAIL\n' > "${run_dir}/status.txt"
    fi
    exit "${final_status}"
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

printf 'run scope: %s\n' "${run_scope}"
printf 'run directory: %s\n' "${run_dir}"
if [[ "${run_scope}" == environment-smoke-* ]]; then
    printf 'WARNING: Vera results are environment smoke data and are not comparable with the Uranus/Neptune study.\n'
fi

apptainer --version > "${run_dir}/apptainer-version.txt"
sha256sum "${image}" > "${run_dir}/image-sha256.txt"
git rev-parse HEAD > "${run_dir}/git-commit.txt"
git status --short > "${run_dir}/git-status.txt"
git diff --no-ext-diff > "${run_dir}/source-diff.patch"
scontrol show job "${job_id}" > "${run_dir}/slurm-job.txt"
nvidia-smi -L > "${run_dir}/gpus.txt"
nvidia-smi topo -m > "${run_dir}/gpu-topology.txt"

apptainer overlay create --size 256 "${overlay}"

site_root="$(apptainer exec \
    --cleanenv \
    --overlay "${overlay}" \
    "${image}" \
    python3 -c 'import inspect; from pathlib import Path; import vllm; print(Path(inspect.getfile(vllm)).resolve().parent.parent)')"
printf '%s\n' "${site_root}" > "${run_dir}/site-packages-root.txt"

patch_state="$(patch_manager verify --root "${site_root}")"
printf '%s\n' "${patch_state}" > "${run_dir}/timing-patch-state-before.txt"
[[ "${patch_state}" == *"not applied"* ]] || {
    printf 'refusing to start from patch state: %s\n' "${patch_state}" >&2
    exit 1
}

patch_applied=1
patch_manager apply --root "${site_root}" \
    > "${run_dir}/timing-patch-apply.log" 2>&1
patch_manager verify --root "${site_root}" \
    > "${run_dir}/timing-patch-state-active.txt"

vllm_command=(
    apptainer exec
    --cleanenv
    --nv
    --overlay "${overlay}"
    --bind "${project_root}:/workspace"
    --bind "${model_dir}:/model:ro"
    --bind "${runtime_cache}:/runtime-cache"
    --pwd /workspace
    --env CUDA_VISIBLE_DEVICES=0,1,2,3
    --env VLLM_SERVER_DEV_MODE=1
    --env VLLM_MOE_TIMING=1
    --env PYTHONUNBUFFERED=1
    --env PYTHONDONTWRITEBYTECODE=1
    --env HF_HUB_OFFLINE=1
    --env TRANSFORMERS_OFFLINE=1
    --env XDG_CACHE_HOME=/runtime-cache
    --env TRITON_CACHE_DIR=/runtime-cache/triton
    --env TORCHINDUCTOR_CACHE_DIR=/runtime-cache/torchinductor
    "${image}"
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
    --enforce-eager
)
printf '%q ' "${vllm_command[@]}" > "${run_dir}/vllm-command.txt"
printf '\n' >> "${run_dir}/vllm-command.txt"

# This script is non-interactive, so job control is disabled and setsid does
# not fork an untracked runtime parent. The captured PID is also the PGID.
setsid "${vllm_command[@]}" > "${run_dir}/vllm.log" 2>&1 &
server_pid=$!
printf '%s\n' "${server_pid}" > "${pid_file}"

ready=0
for attempt in $(seq 1 180); do
    if ! kill -0 "${server_pid}" 2>/dev/null; then
        printf 'vLLM exited before becoming ready\n' >&2
        tail -n 120 "${run_dir}/vllm.log" >&2 || true
        exit 1
    fi
    if curl --fail --silent --max-time 2 http://127.0.0.1:8000/health \
        >/dev/null 2>&1; then
        ready=1
        break
    fi
    if (( attempt % 6 == 0 )); then
        printf 'waiting for vLLM: %s seconds\n' "$((attempt * 10))"
        tail -n 5 "${run_dir}/vllm.log" || true
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
    python3 -m src.benchmark "${container_benchmark_dir}" \
    > "${run_dir}/benchmark.log" 2>&1

scripts/stop_vllm.sh "${pid_file}"
server_pid=""
patch_manager revert --root "${site_root}" \
    > "${run_dir}/timing-patch-revert.log" 2>&1
patch_applied=0

python3 - "${benchmark_dir}/metrics.json" "${run_dir}/summary.json" \
    "${run_scope}" "${run_dir}" <<'PY'
import json
import sys
from pathlib import Path

metrics_path = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
scope = sys.argv[3]
run_directory = sys.argv[4]
metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
summary = {
    "status": metrics["status"],
    "scope": scope,
    "run_directory": run_directory,
    "submitted_requests": metrics["submitted_requests"],
    "completed_requests": metrics["completed_requests"],
    "failed_requests": metrics["failed_requests"],
    "active_request_wall_time_ms": metrics["active_request_wall_time_ms"],
    "fused_moe_max_rank_time_ms": metrics["fused_moe_max_rank_time_ms"],
    "fused_moe_over_active_request_wall_time_percent": metrics[
        "fused_moe_over_active_request_wall_time_percent"
    ],
    "moe_timing_record_count": metrics["moe_timing_record_count"],
    "moe_timing_per_rank": metrics["moe_timing_per_rank"],
}
summary_path.write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print("\n=== FINAL SUMMARY ===")
print(json.dumps(summary, indent=2, sort_keys=True))
PY

cleanup_failed=0
