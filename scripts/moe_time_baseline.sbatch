#!/usr/bin/env bash
#SBATCH --job-name=moe-time-baseline
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
#SBATCH --output=results/raw/slurm-moe-time-%j.out

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:?submit this script with sbatch from the repository root}"
cd "${PROJECT_ROOT}"

source "${HOME}/miniforge3/etc/profile.d/conda.sh"
conda activate cuda-env

case "$(hostname -s)" in
    uranus*|neptune*) ;;
    *) printf 'refusing to run outside uranus/neptune\n' >&2; exit 1 ;;
esac

output_root="${PROJECT_ROOT}/results/raw"
run_id="moe-time-${SLURM_JOB_ID}-$(date -u +%Y%m%dT%H%M%SZ)"
run_dir="${output_root}/${run_id}"
benchmark_dir="${run_dir}/benchmark"
pid_file="${run_dir}/vllm.pid"
mkdir -p "${output_root}"
mkdir "${run_dir}"

# Keep the controller's stdout/stderr with the rest of the run artifacts.
exec > >(tee -a "${run_dir}/job.log") 2>&1

server_pid=""
clock_monitor_pid=""
timing_patch_applied=0

record_clocks() {
    local destination="$1"
    nvidia-smi \
        --query-gpu=index,uuid,name,pstate,clocks.current.graphics,clocks.current.sm,clocks.current.memory,clocks.max.graphics \
        --format=csv > "${destination}"
}

stop_clock_monitor() {
    if [[ -n "${clock_monitor_pid}" ]]; then
        kill -TERM "${clock_monitor_pid}" 2>/dev/null || true
        wait "${clock_monitor_pid}" 2>/dev/null || true
        clock_monitor_pid=""
    fi
}

cleanup() {
    local status=$?
    local cleanup_failed=0
    trap - EXIT INT TERM

    stop_clock_monitor
    if [[ -n "${server_pid}" ]]; then
        scripts/stop_vllm.sh "${pid_file}" || cleanup_failed=1
    fi
    record_clocks "${run_dir}/clocks-after.csv" || cleanup_failed=1
    if [[ "${timing_patch_applied}" -eq 1 ]]; then
        scripts/manage_moe_timing_patch.sh revert --root "${site_root}" \
            > "${run_dir}/timing-patch-revert.log" 2>&1 || cleanup_failed=1
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

trace="data/traces/gsm8k-50-r0.5.jsonl"
expected_trace_sha="721638a6250b929de05c2c355c8bc9b20258db7b28fedb94acb5572136571cbc"
actual_trace_sha="$(sha256sum "${trace}" | awk '{print $1}')"
[[ "${actual_trace_sha}" == "${expected_trace_sha}" ]] || {
    printf 'unexpected trace hash: %s\n' "${actual_trace_sha}" >&2
    exit 1
}

cp configs/model.yaml "${run_dir}/model.yaml"
cp "${trace}" "${run_dir}/trace.jsonl"
cp patches/vllm-0.15.1-moe-timing.patch "${run_dir}/moe-timing.patch"
cp patches/vllm-0.15.1-moe-timing.json "${run_dir}/moe-timing-manifest.json"
sha256sum configs/model.yaml "${trace}" patches/vllm-0.15.1-moe-timing.* \
    > "${run_dir}/input-sha256.txt"
git rev-parse HEAD > "${run_dir}/git-commit.txt"
git status --short > "${run_dir}/git-status.txt"
[[ ! -s "${run_dir}/git-status.txt" ]] || {
    printf 'refusing to measure from a dirty Git worktree\n' >&2
    exit 1
}
git diff --no-ext-diff > "${run_dir}/source-diff.patch"
scontrol show job "${SLURM_JOB_ID}" > "${run_dir}/slurm-job.txt"

python - <<'PY' > "${run_dir}/versions.txt"
import platform
import torch
import vllm

print("python", platform.python_version())
print("torch", torch.__version__)
print("torch_cuda", torch.version.cuda)
print("vllm", vllm.__version__)
PY
nvidia-smi topo -m > "${run_dir}/gpu-topology.txt"
record_clocks "${run_dir}/clocks-before.csv"

site_root="$(python - <<'PY'
import inspect
from pathlib import Path
import vllm

print(Path(inspect.getfile(vllm)).resolve().parent.parent)
PY
)"
printf '%s\n' "${site_root}" > "${run_dir}/site-packages-root.txt"

patch_state="$(scripts/manage_moe_timing_patch.sh verify --root "${site_root}")"
printf '%s\n' "${patch_state}" > "${run_dir}/timing-patch-state-before.txt"
[[ "${patch_state}" == *"not applied"* ]] || {
    printf 'refusing to start from a non-clean timing patch state: %s\n' "${patch_state}" >&2
    exit 1
}
timing_patch_applied=1
scripts/manage_moe_timing_patch.sh apply --root "${site_root}" \
    > "${run_dir}/timing-patch-apply.log" 2>&1
scripts/manage_moe_timing_patch.sh verify --root "${site_root}" \
    > "${run_dir}/timing-patch-state-active.txt"

export CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_SERVER_DEV_MODE=1
export VLLM_MOE_TIMING=1
unset VLLM_EXPERT_PHASE_PROFILE

python -m src.config configs/model.yaml -- --enforce-eager \
    > "${run_dir}/vllm-command.txt"
setsid scripts/serve_vllm.sh --config configs/model.yaml -- --enforce-eager \
    > "${run_dir}/vllm.log" 2>&1 &
server_pid=$!
printf '%s\n' "${server_pid}" > "${pid_file}"

ready=0
for _ in $(seq 1 360); do
    if ! kill -0 "${server_pid}" 2>/dev/null; then
        printf 'vLLM exited before becoming ready\n' >&2
        tail -n 120 "${run_dir}/vllm.log" >&2 || true
        exit 1
    fi
    if curl --fail --silent --show-error http://127.0.0.1:8000/health \
        >/dev/null 2>&1; then
        ready=1
        break
    fi
    sleep 10
done
[[ "${ready}" -eq 1 ]] || {
    printf 'timed out waiting for vLLM\n' >&2
    exit 1
}

nvidia-smi \
    --query-gpu=timestamp,index,uuid,pstate,clocks.current.graphics,clocks.current.sm \
    --format=csv,noheader,nounits --loop=1 \
    > "${run_dir}/clock-samples.csv" 2>&1 &
clock_monitor_pid=$!
printf '%s\n' "${clock_monitor_pid}" > "${run_dir}/clock-monitor.pid"

python -m src.benchmark \
    --trace "${trace}" \
    --base-url http://127.0.0.1:8000 \
    --model qwen3-30b-a3b \
    --output-dir "${benchmark_dir}" \
    --expected-ranks 4 \
    --warmup-requests 8 \
    --settle-seconds 5 \
    --max-concurrency 256 \
    --request-timeout-seconds 900 \
    2>&1 | tee "${run_dir}/benchmark.log"

stop_clock_monitor
[[ -s "${run_dir}/clock-samples.csv" ]] || {
    printf 'clock monitor produced no samples\n' >&2
    exit 1
}
record_clocks "${run_dir}/clocks-after-benchmark.csv"
