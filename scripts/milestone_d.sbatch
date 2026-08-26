#!/usr/bin/env bash
#SBATCH --job-name=moe-milestone-d
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --gres=gpu:L40s:4
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --output=results/raw/slurm-milestone-d-%j.out

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
cd "${PROJECT_ROOT}"
source "${HOME}/miniforge3/etc/profile.d/conda.sh"
conda activate cuda-env
export CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_SERVER_DEV_MODE=1

controller="${PROJECT_ROOT}/results/raw/milestone-d-${SLURM_JOB_ID}-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "${controller}/servers" "${controller}/placements"
manifest="${controller}/runs.tsv"
printf 'label\tplacement\tclocks\trun_dir\n' > "${manifest}"
server_pid=""
pid_file=""
patch_applied_by_job=0
site_root="$(python - <<'PY'
import inspect
from pathlib import Path
import vllm
print(Path(inspect.getfile(vllm)).resolve().parent.parent)
PY
)"

cleanup() {
    status=$?
    trap - EXIT INT TERM
    if [[ -n "${server_pid}" && -n "${pid_file}" ]]; then
        scripts/stop_vllm.sh "${pid_file}" || true
    fi
    python -m src.gpu_control reset >> "${controller}/clock-reset.log" 2>&1 || true
    if [[ "${patch_applied_by_job}" -eq 1 ]]; then
        scripts/manage_vllm_patch.sh revert --root "${site_root}" \
            >> "${controller}/patch-revert.log" 2>&1 || true
    fi
    printf '%s\n' "${status}" > "${controller}/exit-status.txt"
    exit "${status}"
}
trap cleanup EXIT INT TERM

allocated_gpus="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')"
[[ "${allocated_gpus}" -eq 4 ]] || { printf 'expected 4 allocated GPUs, found %s\n' "${allocated_gpus}" >&2; exit 1; }

scripts/check_environment.sh > "${controller}/environment.txt" 2>&1
initial_patch_state="$(scripts/manage_vllm_patch.sh verify --root "${site_root}")"
printf '%s\n' "${initial_patch_state}" > "${controller}/patch-state-before.txt"
if [[ "${initial_patch_state}" == *'not applied'* ]]; then
    scripts/manage_vllm_patch.sh apply --root "${site_root}" > "${controller}/patch-apply.log"
    patch_applied_by_job=1
fi
scripts/manage_vllm_patch.sh verify --root "${site_root}" > "${controller}/patch-state-active.txt"
sha256sum patches/vllm-0.15.1-placement-profile.* > "${controller}/patch-hashes.txt"

python -m tools.rescale_trace \
    --input data/traces/math-pool-r1.jsonl \
    --output "${controller}/gsm8k-50-r0.5.jsonl" \
    --request-rate 0.5 --limit 50
python -m tools.mark_trace_outputs \
    --input data/traces/math-pool-r1.jsonl \
    --output "${controller}/gsm8k-validation-15-r0.5.jsonl" \
    --count 15 --request-rate 0.5

stop_server() {
    if [[ -n "${server_pid}" ]]; then
        scripts/stop_vllm.sh "${pid_file}"
        server_pid=""
        pid_file=""
    fi
}

launch_server() {
    local placement_name="$1"
    local placement_file="${2:-}"
    local server_dir="${controller}/servers/${placement_name}"
    mkdir -p "${server_dir}"
    pid_file="${server_dir}/vllm.pid"
    local extra=()
    if [[ -n "${placement_file}" ]]; then
        extra=(--expert-placement-file "${placement_file}")
    fi
    python -m src.config configs/model_qwen3_30b_a3b_profile.yaml -- "${extra[@]}" \
        > "${server_dir}/vllm-command.txt"
    setsid scripts/serve_vllm.sh --config configs/model_qwen3_30b_a3b_profile.yaml \
        -- "${extra[@]}" > "${server_dir}/vllm.log" 2>&1 &
    server_pid=$!
    printf '%s\n' "${server_pid}" > "${pid_file}"
    local ready=0
    for _ in $(seq 1 270); do
        if ! kill -0 "${server_pid}" 2>/dev/null; then
            tail -n 160 "${server_dir}/vllm.log" >&2 || true
            return 1
        fi
        if curl --fail --silent http://127.0.0.1:8000/health >/dev/null; then
            ready=1
            break
        fi
        sleep 10
    done
    [[ "${ready}" -eq 1 ]] || { printf 'timed out waiting for vLLM\n' >&2; return 1; }
}

set_clocks() {
    local csv="$1"
    python -m src.gpu_control set-all --clocks "${csv}" \
        > "${controller}/clocks-$(date -u +%Y%m%dT%H%M%S.%NZ).json"
}

run_one() {
    local label="$1"
    local placement_name="$2"
    local placement_file="$3"
    local trace="$4"
    local clocks="$5"
    local collect_profile="${6:-0}"
    local args=(
        --trace "${trace}"
        --request-rate 0.5
        --local-files-only
        --run-label "${label}"
        --placement-name "${placement_name}"
        --vllm-patch-manifest patches/vllm-0.15.1-placement-profile.json
        --target-clocks ${clocks//,/ }
        --server-pid-file "${pid_file}"
        --server-log "${controller}/servers/${placement_name}/vllm.log"
    )
    if [[ -n "${placement_file}" ]]; then
        args+=(--placement-file "${placement_file}")
    fi
    if [[ "${collect_profile}" -eq 1 ]]; then
        args+=(--expert-profile)
    fi
    local output="${controller}/${label}.json"
    python -m src.benchmark "${args[@]}" > "${output}" 2>&1
    local run_dir
    run_dir="$(python -c 'import json,sys; print(json.loads(open(sys.argv[1]).read().splitlines()[-1])["run_dir"])' "${output}")"
    if [[ -n "${frozen_counts:-}" && "${collect_profile}" -eq 0 ]]; then
        cp "${frozen_counts}" "${run_dir}/expert_counts.npz"
        if [[ -n "${placement_file}" ]]; then
            cp "${placement_file}" "${run_dir}/expert_mapping.json"
        else
            cp "${frozen_default_mapping}" "${run_dir}/expert_mapping.json"
        fi
        sha256sum "${frozen_counts}" \
            "${run_dir}/expert_mapping.json" > "${run_dir}/frozen-profile-hashes.txt"
        printf '%s\n' "${p0_profile}" > "${run_dir}/frozen-profile-source.txt"
    fi
    printf '%s\t%s\t%s\t%s\n' "${label}" "${placement_name}" "${clocks}" "${run_dir}" >> "${manifest}"
    printf '%s\n' "${run_dir}"
}

uniform_runs() {
    local placement_name="$1"
    local placement_file="$2"
    for specification in high:2520,2520,2520,2520 medium:1635,1635,1635,1635 low:1005,1005,1005,1005; do
        local level="${specification%%:*}"
        local clocks="${specification#*:}"
        set_clocks "${clocks}"
        run_one "${placement_name}-uniform-${level}" "${placement_name}" \
            "${placement_file}" "${controller}/gsm8k-50-r0.5.jsonl" "${clocks}" >/dev/null
    done
}

single_gpu_runs() {
    local placement_name="$1"
    local placement_file="$2"
    for gpu in 0 1 2 3; do
        local clocks=(2520 2520 2520 2520)
        clocks[${gpu}]=1635
        local csv="${clocks[0]},${clocks[1]},${clocks[2]},${clocks[3]}"
        set_clocks "${csv}"
        run_one "${placement_name}-gpu${gpu}-medium" "${placement_name}" \
            "${placement_file}" "${controller}/gsm8k-50-r0.5.jsonl" "${csv}" >/dev/null
    done
}

repeat_key_runs() {
    local placement_name="$1"
    local placement_file="$2"
    for repetition in 2 3; do
        set_clocks '2520,2520,2520,2520'
        run_one "${placement_name}-uniform-high-rep${repetition}" "${placement_name}" \
            "${placement_file}" "${controller}/gsm8k-50-r0.5.jsonl" \
            '2520,2520,2520,2520' >/dev/null
        set_clocks '1635,2520,2520,2520'
        run_one "${placement_name}-gpu0-medium-rep${repetition}" "${placement_name}" \
            "${placement_file}" "${controller}/gsm8k-50-r0.5.jsonl" \
            '1635,2520,2520,2520' >/dev/null
    done
}

resume_controller="${MILESTONE_D_RESUME_CONTROLLER:-}"
if [[ -n "${resume_controller}" ]]; then
    resume_controller="$(realpath "${resume_controller}")"
    [[ -f "${resume_controller}/runs.tsv" ]] || {
        printf 'resume controller has no run manifest: %s\n' "${resume_controller}" >&2
        exit 1
    }
    cp -a "${resume_controller}/placements/." "${controller}/placements/"
    cp "${resume_controller}/placement-generation.json" "${controller}/"
    awk -F '\t' 'NR > 1 && $2 == "default_linear"' \
        "${resume_controller}/runs.tsv" >> "${manifest}"
    p0_validation="$(awk -F '\t' '$1 == "p0-live-validation" {print $4}' \
        "${resume_controller}/runs.tsv")"
    p0_profile="$(awk -F '\t' '$1 == "p0-frozen-profile" {print $4}' \
        "${resume_controller}/runs.tsv")"
    [[ -f "${p0_validation}/requests.jsonl" ]] || exit 1
    [[ -f "${p0_profile}/expert_counts.npz" ]] || exit 1
    cp "${resume_controller}/gsm8k-50-r0.5.jsonl" "${controller}/"
    cp "${resume_controller}/gsm8k-validation-15-r0.5.jsonl" "${controller}/"
    printf '%s\n' "${resume_controller}" > "${controller}/resumed-from.txt"
else
    launch_server default_linear
    set_clocks '2520,2520,2520,2520'
    p0_validation="$(run_one p0-live-validation default_linear '' \
        "${controller}/gsm8k-validation-15-r0.5.jsonl" '2520,2520,2520,2520')"
    python -m tools.validate_live_run "${p0_validation}" > "${controller}/live-validation.json"
    p0_profile="$(run_one p0-frozen-profile default_linear '' \
        "${controller}/gsm8k-50-r0.5.jsonl" '2520,2520,2520,2520' 1)"
    python -m tools.validate_live_run "${p0_profile}" --require-expert-profile \
        > "${controller}/p0-profile-validation.json"
    python -m src.placement --counts "${p0_profile}/expert_counts.npz" \
        --output-dir "${controller}/placements" --num-groups 1 \
        > "${controller}/placement-generation.json"
    uniform_runs default_linear "${controller}/placements/default_linear.json"
    stop_server
fi

frozen_counts="${p0_profile}/expert_counts.npz"
frozen_default_mapping="${p0_profile}/expert_mapping.json"

for placement_name in upstream_default_eplb diagnostic_load_concentrated; do
    placement_file="${controller}/placements/${placement_name}.json"
    launch_server "${placement_name}" "${placement_file}"
    set_clocks '2520,2520,2520,2520'
    validation="$(run_one "${placement_name}-semantic-validation" "${placement_name}" \
        "${placement_file}" "${controller}/gsm8k-validation-15-r0.5.jsonl" \
        '2520,2520,2520,2520')"
    python -m tools.validate_live_run "${validation}" \
        > "${controller}/${placement_name}-live-validation.json"
    python -m tools.compare_validation_outputs "${p0_validation}" "${validation}" \
        --minimum-common-prefix 8 \
        > "${controller}/${placement_name}-semantic-comparison.json"
    uniform_runs "${placement_name}" "${placement_file}"
    single_gpu_runs "${placement_name}" "${placement_file}"
    repeat_key_runs "${placement_name}" "${placement_file}"
    stop_server
done

python -m src.gpu_control reset > "${controller}/final-clock-reset.json"
