#!/usr/bin/env bash
# One-allocation routing and CUDA-event characterization for the AFD FFN pool.
# Submit with MILESTONE_N_STUDY=<prepared-dir> sbatch <this-file>.
#SBATCH --job-name=moe-n-afd-cost
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --nodelist=uranus
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --gres=gpu:L40s:4
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --output=results/raw/slurm-milestone-n-afd-cost-%j.out

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
cd "${PROJECT_ROOT}"
export AFD_PROJECT_ROOT="${PROJECT_ROOT}"
source scripts/afd_common.sh

study="${MILESTONE_N_STUDY:?set MILESTONE_N_STUDY}"
study="$(cd "${study}" && pwd)"
plan="${study}/manifest.json"
trace="${study}/trace.jsonl"
[[ -f "${plan}" && -f "${trace}" ]] || { printf 'missing prepared study\n' >&2; exit 1; }
expected_commit="$("${AFD_PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["git_commit"])' "${plan}")"
actual_commit="$(git rev-parse HEAD)"
[[ "${expected_commit}" == "${actual_commit}" ]] || {
    printf 'repository commit drift: manifest=%s runtime=%s\n' "${expected_commit}" "${actual_commit}" >&2
    exit 1
}
git diff --quiet && git diff --cached --quiet || {
    printf 'refusing modified tracked worktree\n' >&2; exit 1;
}
[[ ! -e "${study}/slurm-job-id.txt" ]] || { printf 'study already submitted\n' >&2; exit 1; }
printf '%s\n' "${SLURM_JOB_ID}" > "${study}/slurm-job-id.txt"

attention_pid=""
ffn_pid=""
patch_applied=0
cleanup() {
    status=$?
    trap - EXIT INT TERM
    afd_stop_process_group "${attention_pid}" || true
    afd_stop_process_group "${ffn_pid}" || true
    afd_reset_clocks > "${study}/clock-reset.json" 2> "${study}/clock-reset.err" || status=1
    nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv \
        > "${study}/compute-apps-after.csv" 2>&1 || true
    if [[ "${patch_applied}" -eq 1 ]]; then
        scripts/manage_afd_cost_model_patch.sh revert --root "${AFD_PLUGIN_ROOT}" \
            > "${study}/patch-revert.txt" 2>&1 || status=1
    fi
    scripts/manage_afd_cost_model_patch.sh verify --root "${AFD_PLUGIN_ROOT}" \
        > "${study}/patch-state-after.txt" 2>&1 || status=1
    "${AFD_PYTHON}" tools/check_milestone_m_environments.py \
        --old-python "${HOME}/miniforge3/envs/cuda-env/bin/python" \
        --afd-python "${AFD_PYTHON}" --plugin-root "${AFD_PLUGIN_ROOT}" \
        --uv "${PROJECT_ROOT}/.tools/uv-bootstrap/bin/uv" \
        --output "${study}/environment-after.json" || status=1
    printf '%s\n' "${status}" > "${study}/exit-status.txt"
    exit "${status}"
}
trap cleanup EXIT INT TERM

afd_require_runtime
afd_check_allocation > "${study}/slurm-allocation.txt"
nvidia-smi topo -m > "${study}/topology.txt"
git rev-parse HEAD > "${study}/git-commit.txt"
git -C "${AFD_PLUGIN_ROOT}" rev-parse HEAD > "${study}/afd-plugin-commit.txt"
sha256sum "${plan}" "${trace}" data/placements/milestone_m/P_balanced.json \
    data/placements/milestone_m_target_splits/P_split_40_slow_rank0.json \
    data/placements/milestone_m_target_splits/P_split_40_slow_rank1.json \
    patches/afd-plugin-0.26.0-cost-model.* > "${study}/immutable-input-hashes.txt"
scripts/manage_afd_cost_model_patch.sh verify --root "${AFD_PLUGIN_ROOT}" \
    > "${study}/patch-state-before.txt"
scripts/manage_afd_cost_model_patch.sh apply --root "${AFD_PLUGIN_ROOT}" \
    > "${study}/patch-apply.txt"
patch_applied=1

api_port=18560
ffn_api_port=18561
afd_port=6310
trace_sha="$("${AFD_PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["trace_sha256"])' "${plan}")"
balanced="${PROJECT_ROOT}/data/placements/milestone_m/P_balanced.json"

stop_server() {
    afd_stop_process_group "${attention_pid}"
    attention_pid=""
    afd_stop_process_group "${ffn_pid}"
    ffn_pid=""
    sleep 5
    local apps_file="$1"
    nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader > "${apps_file}"
    [[ ! -s "${apps_file}" ]] || { printf 'GPU processes remain after shutdown\n' >&2; exit 1; }
}

validate_ownership() {
    local placement="$1" evidence="$2" output="$3"
    "${AFD_PYTHON}" -m tools.validate_milestone_m_runtime_ownership \
        --placement-file "${placement}" --evidence-dir "${evidence}" \
        --output "${output}" > "${output%.json}.stdout"
}

run_conditioning() {
    local output_root="$1" model_name="$2" placement="$3" clocks_csv="$4" log="$5"
    IFS=',' read -r -a clocks_array <<< "${clocks_csv}"
    "${AFD_PYTHON}" -m src.benchmark \
        --trace "${trace}" --base-url "http://127.0.0.1:${api_port}" \
        --model "${model_name}-attention" --tokenizer Qwen/Qwen3-30B-A3B \
        --request-rate 8 --override-arrival-rate --output-root "${output_root}" \
        --warmup-requests 8 --settle-before-s 5 --settle-after-s 5 \
        --monitor-interval-s 0.2 --max-concurrency 256 --request-timeout-s 900 \
        --local-files-only --placement-file "${placement}" \
        --vllm-patch-manifest patches/afd-plugin-0.26.0-cost-model.json \
        --target-clocks "${clocks_array[@]}" --run-label conditioning \
        --placement-name conditioning --server-log "${log}" \
        > "${output_root}.log" 2>&1
    local count
    count="$(find "${output_root}" -mindepth 1 -maxdepth 1 -type d | wc -l)"
    [[ "${count}" -eq 1 ]] || { printf 'conditioning run count mismatch\n' >&2; exit 1; }
}

printf 'stage\torder\tconfig_id\tlifecycle\tfrequency_mhz\tstatus\trun_dir\n' > "${study}/runs.tsv"

# Stage A: one routing-only lifecycle. Profiling is enabled only for the measured trace.
routing_dir="${study}/stage-a-routing"
mkdir -p "${routing_dir}/server" "${routing_dir}/ownership" \
    "${routing_dir}/profile-control" "${routing_dir}/conditioning" "${routing_dir}/runs"
"${AFD_PYTHON}" -c \
    'from pathlib import Path; from src.afd_routing_profile import write_profile_control; import sys; write_profile_control(Path(sys.argv[1]), "startup", False)' \
    "${routing_dir}/profile-control"
SUDO="${SUDO:-sudo}" "${AFD_PYTHON}" -m src.gpu_control set-all \
    --clocks 2520,2520,2520,2520 > "${routing_dir}/clock-set.json"
model_name="qwen3-moe-n-routing"
afd_start_role ffn 2,3 "${ffn_api_port}" Qwen/Qwen3-30B-A3B "${model_name}" "${afd_port}" \
    "${routing_dir}/server/ffn.log" "${routing_dir}/server/ffn-command.txt" \
    "${balanced}" "${routing_dir}/ownership" "${routing_dir}/profile-control" full_decode_only
ffn_pid="${AFD_LAST_PID}"
sleep 5
afd_start_role attention 0,1 "${api_port}" Qwen/Qwen3-30B-A3B "${model_name}" "${afd_port}" \
    "${routing_dir}/server/attention.log" "${routing_dir}/server/attention-command.txt" \
    "${balanced}" "${routing_dir}/ownership" "" full_decode_only
attention_pid="${AFD_LAST_PID}"
afd_wait_attention_api "${attention_pid}" "${ffn_pid}" 2400 \
    "http://127.0.0.1:${api_port}/v1/models" || exit 1
validate_ownership "${balanced}" "${routing_dir}/ownership" "${routing_dir}/runtime-ownership.json"
run_conditioning "${routing_dir}/conditioning" "${model_name}" "${balanced}" \
    2520,2520,2520,2520 "${routing_dir}/server/attention.log"
"${AFD_PYTHON}" -m src.benchmark \
    --trace "${trace}" --base-url "http://127.0.0.1:${api_port}" \
    --model "${model_name}-attention" --tokenizer Qwen/Qwen3-30B-A3B \
    --request-rate 8 --override-arrival-rate --output-root "${routing_dir}/runs" \
    --warmup-requests 8 --settle-before-s 5 --settle-after-s 5 \
    --monitor-interval-s 0.2 --max-concurrency 256 --request-timeout-s 900 \
    --local-files-only --placement-file "${balanced}" \
    --vllm-patch-manifest patches/afd-plugin-0.26.0-cost-model.json \
    --afd-profile-control-dir "${routing_dir}/profile-control" \
    --target-clocks 2520 2520 2520 2520 --store-output-token-ids \
    --run-label routing-trace --placement-name P_balanced \
    --server-log "${routing_dir}/server/attention.log" > "${routing_dir}/benchmark.log" 2>&1
routing_run="$(find "${routing_dir}/runs" -mindepth 1 -maxdepth 1 -type d | head -1)"
[[ -f "${routing_run}/routing_trace.npz" ]] || { printf 'routing trace missing\n' >&2; exit 1; }
"${AFD_PYTHON}" - "${routing_run}/routing_trace.npz" <<'PY'
import sys
from pathlib import Path
from src.afd_cost_model import load_routing_trace
r = load_routing_trace(Path(sys.argv[1]))
assert set(r["layers"].tolist()) == set(range(48))
assert len(r["layers"]) % 48 == 0
PY
printf 'routing\t0\tP_balanced-hh\trouting\t2520\tcomplete\t%s\n' "${routing_run}" >> "${study}/runs.tsv"
printf '%s\n' "${routing_run}" > "${study}/routing-run-dir.txt"
stop_server "${routing_dir}/compute-apps-after.csv"

# Stages B/C: each timing configuration receives a fresh server and conditioning lifecycle.
mkdir -p "${study}/timing-configurations"
while IFS=$'\t' read -r order config_id stage lifecycle frequency placement_path placement_sha mapping_sha; do
    config_dir="${study}/timing-configurations/$(printf '%02d' "${order}")-${config_id}"
    mkdir -p "${config_dir}/server" "${config_dir}/ownership" \
        "${config_dir}/timing-control" "${config_dir}/conditioning" "${config_dir}/runs"
    placement="${PROJECT_ROOT}/${placement_path}"
    [[ "$(sha256sum "${placement}" | awk '{print $1}')" == "${placement_sha}" ]] || {
        printf 'placement hash drift: %s\n' "${placement}" >&2; exit 1;
    }
    clocks="2520,2520,${frequency},${frequency}"
    "${AFD_PYTHON}" -c \
        'from pathlib import Path; from src.afd_routing_profile import write_profile_control; import sys; write_profile_control(Path(sys.argv[1]), "startup", False)' \
        "${config_dir}/timing-control"
    SUDO="${SUDO:-sudo}" "${AFD_PYTHON}" -m src.gpu_control set-all \
        --clocks "${clocks}" > "${config_dir}/clock-set.json"
    "${AFD_PYTHON}" -m src.gpu_control list > "${config_dir}/clock-readback.json"
    sleep 5
    nvidia-smi --query-gpu=index,clocks.current.sm --format=csv,noheader,nounits \
        > "${config_dir}/actual-clocks-before-conditioning.csv"
    model_name="qwen3-moe-n-${order}"
    afd_start_role ffn 2,3 "${ffn_api_port}" Qwen/Qwen3-30B-A3B "${model_name}" "${afd_port}" \
        "${config_dir}/server/ffn.log" "${config_dir}/server/ffn-command.txt" \
        "${placement}" "${config_dir}/ownership" "" full_decode_only \
        "${config_dir}/timing-control" "${frequency},${frequency}"
    ffn_pid="${AFD_LAST_PID}"
    sleep 5
    afd_start_role attention 0,1 "${api_port}" Qwen/Qwen3-30B-A3B "${model_name}" "${afd_port}" \
        "${config_dir}/server/attention.log" "${config_dir}/server/attention-command.txt" \
        "${placement}" "${config_dir}/ownership" "" full_decode_only
    attention_pid="${AFD_LAST_PID}"
    afd_wait_attention_api "${attention_pid}" "${ffn_pid}" 2400 \
        "http://127.0.0.1:${api_port}/v1/models" || exit 1
    validate_ownership "${placement}" "${config_dir}/ownership" "${config_dir}/runtime-ownership.json"
    actual_mapping_sha="$("${AFD_PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["actual_mapping_sha256"])' "${config_dir}/runtime-ownership.json")"
    [[ "${actual_mapping_sha}" == "${mapping_sha}" ]] || { printf 'ownership mismatch\n' >&2; exit 1; }
    run_conditioning "${config_dir}/conditioning" "${model_name}" "${placement}" \
        "${clocks}" "${config_dir}/server/attention.log"
    IFS=',' read -r -a target_array <<< "${clocks}"
    "${AFD_PYTHON}" -m src.benchmark \
        --trace "${trace}" --base-url "http://127.0.0.1:${api_port}" \
        --model "${model_name}-attention" --tokenizer Qwen/Qwen3-30B-A3B \
        --request-rate 8 --override-arrival-rate --output-root "${config_dir}/runs" \
        --warmup-requests 8 --settle-before-s 5 --settle-after-s 5 \
        --monitor-interval-s 0.2 --max-concurrency 256 --request-timeout-s 900 \
        --local-files-only --placement-file "${placement}" \
        --vllm-patch-manifest patches/afd-plugin-0.26.0-cost-model.json \
        --afd-timing-control-dir "${config_dir}/timing-control" \
        --target-clocks "${target_array[@]}" --store-output-token-ids \
        --run-label "${config_id}" --placement-name "${config_id}" \
        --server-log "${config_dir}/server/attention.log" > "${config_dir}/benchmark.log" 2>&1
    run_dir="$(find "${config_dir}/runs" -mindepth 1 -maxdepth 1 -type d | head -1)"
    "${AFD_PYTHON}" -m tools.validate_live_run "${run_dir}" > "${run_dir}/live-validation.json"
    "${AFD_PYTHON}" -m tools.validate_milestone_n_cost_timing \
        --routing-trace "${routing_run}/routing_trace.npz" --timing-run "${run_dir}" \
        --placement "${placement}" --config-id "${config_id}" \
        --lifecycle "${lifecycle}" --frequency-mhz "${frequency}" \
        > "${run_dir}/cost-timing-validation.json"
    printf 'timing\t%s\t%s\t%s\t%s\tcomplete\t%s\n' \
        "${order}" "${config_id}" "${lifecycle}" "${frequency}" "${run_dir}" >> "${study}/runs.tsv"
    stop_server "${config_dir}/compute-apps-after.csv"
done < <("${AFD_PYTHON}" - "${plan}" <<'PY'
import json, sys
plan = json.load(open(sys.argv[1]))
for row in sorted(plan["configurations"], key=lambda item: item["order"]):
    placement = row["placement"]
    print(row["order"], row["config_id"], row["stage"], row["lifecycle"],
          row["frequency_mhz"], placement["path"], placement["sha256"],
          placement["canonical_mapping_sha256"], sep="\t")
PY
)

complete="$(awk -F '\t' '$1 == "timing" && $6 == "complete" {n++} END {print n+0}' "${study}/runs.tsv")"
expected="$("${AFD_PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["expected_timing_runs"])' "${plan}")"
[[ "${complete}" -eq "${expected}" ]] || { printf 'timing completion mismatch\n' >&2; exit 1; }
"${AFD_PYTHON}" - "${study}/result.json" "${complete}" <<'PY'
import json, sys
json.dump({"status": "complete", "routing_runs": 1, "timing_runs": int(sys.argv[2])},
          open(sys.argv[1], "w"), indent=2)
PY
printf 'PASS\n' > "${study}/status.txt"
