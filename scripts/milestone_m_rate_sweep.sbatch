#!/usr/bin/env bash
# One-server AFD request-rate sweep for a stationary Milestone M operating point.
# Submit with MILESTONE_M_RATE_STUDY=<prepared-dir> sbatch <this-file>.
#SBATCH --job-name=moe-m-afd-rate
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --nodelist=uranus
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --gres=gpu:L40s:4
#SBATCH --exclusive
#SBATCH --time=03:00:00
#SBATCH --output=results/raw/slurm-milestone-m-rate-%j.out

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
cd "${PROJECT_ROOT}"
export AFD_PROJECT_ROOT="${PROJECT_ROOT}"
source scripts/afd_common.sh

study="${MILESTONE_M_RATE_STUDY:?set MILESTONE_M_RATE_STUDY}"
study="$(cd "${study}" && pwd)"
plan="${study}/manifest.json"
trace="${study}/trace.jsonl"
[[ -f "${plan}" && -f "${trace}" ]] || { printf 'missing rate-sweep inputs\n' >&2; exit 1; }
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
        scripts/manage_afd_static_placement_patch.sh revert --root "${AFD_PLUGIN_ROOT}" \
            > "${study}/patch-revert.txt" 2>&1 || status=1
    fi
    scripts/manage_afd_static_placement_patch.sh verify --root "${AFD_PLUGIN_ROOT}" \
        > "${study}/patch-state-after.txt" 2>&1 || status=1
    "${AFD_PYTHON}" tools/check_milestone_m_environments.py \
        --old-python "${HOME}/miniforge3/envs/cuda-env/bin/python" \
        --afd-python "${AFD_PYTHON}" \
        --plugin-root "${AFD_PLUGIN_ROOT}" \
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
sha256sum "${plan}" "${trace}" data/placements/milestone_m/*.json \
    > "${study}/immutable-input-hashes.txt"
cp configs/milestone_m_afd_fixed_frequency.yaml "${study}/resolved-config.yaml"
scripts/manage_afd_static_placement_patch.sh verify --root "${AFD_PLUGIN_ROOT}" \
    > "${study}/patch-state-before.txt"
scripts/manage_afd_static_placement_patch.sh apply --root "${AFD_PLUGIN_ROOT}" \
    > "${study}/patch-apply.txt"
patch_applied=1

placement_path="$("${AFD_PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["placement"]["path"])' "${plan}")"
placement_sha="$("${AFD_PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["placement"]["sha256"])' "${plan}")"
mapping_sha="$("${AFD_PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["placement"]["canonical_mapping_sha256"])' "${plan}")"
placement_file="${PROJECT_ROOT}/${placement_path}"
clocks="2520,2520,2520,2520"
target_array=(2520 2520 2520 2520)

mkdir -p "${study}/server" "${study}/ownership" "${study}/windows"
printf 'order\trequest_rate\trepetition\tstatus\trun_dir\n' > "${study}/runs.tsv"
sha256sum "${placement_file}" > "${study}/placement.sha256"
[[ "$(sha256sum "${placement_file}" | awk '{print $1}')" == "${placement_sha}" ]] || {
    printf 'P_balanced placement hash drift\n' >&2
    exit 1
}

SUDO="${SUDO:-sudo}" "${AFD_PYTHON}" -m src.gpu_control set-all \
    --clocks "${clocks}" > "${study}/clock-set.json"
"${AFD_PYTHON}" -m src.gpu_control list > "${study}/clock-readback.json"
sleep 2

api_port=18440
ffn_api_port=18441
afd_port=6280
served_name="qwen3-moe-m-rate"
afd_start_role ffn 2,3 "${ffn_api_port}" Qwen/Qwen3-30B-A3B "${served_name}" "${afd_port}" \
    "${study}/server/ffn.log" "${study}/server/ffn-command.txt" \
    "${placement_file}" "${study}/ownership" "" full_decode_only
ffn_pid="${AFD_LAST_PID}"
printf '%s\n' "${ffn_pid}" > "${study}/server/ffn.pid"
sleep 5
afd_start_role attention 0,1 "${api_port}" Qwen/Qwen3-30B-A3B "${served_name}" "${afd_port}" \
    "${study}/server/attention.log" "${study}/server/attention-command.txt" \
    "${placement_file}" "${study}/ownership" "" full_decode_only
attention_pid="${AFD_LAST_PID}"
printf '%s\n' "${attention_pid}" > "${study}/server/attention.pid"

if ! afd_wait_attention_api "${attention_pid}" "${ffn_pid}" 2400 \
        "http://127.0.0.1:${api_port}/v1/models"; then
    tail -n 200 "${study}/server/attention.log" >&2 || true
    tail -n 200 "${study}/server/ffn.log" >&2 || true
    exit 1
fi
grep -m1 'AFD FFN EngineCore started; workers run connector loop.' \
    "${study}/server/ffn.log" > "${study}/ffn-ready.txt"
nvidia-smi --query-gpu=index,clocks.current.sm --format=csv,noheader,nounits \
    > "${study}/actual-clocks-before-runs.csv"
"${AFD_PYTHON}" -m tools.validate_milestone_m_runtime_ownership \
    --placement-file "${placement_file}" \
    --evidence-dir "${study}/ownership" \
    --output "${study}/actual-ownership.json" \
    > "${study}/ownership-validation.stdout"
actual_mapping_sha="$("${AFD_PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["actual_mapping_sha256"])' "${study}/actual-ownership.json")"
[[ "${actual_mapping_sha}" == "${mapping_sha}" ]] || {
    printf 'runtime ownership hash mismatch\n' >&2
    exit 1
}

while IFS=$'\t' read -r order rate repetition; do
    label="rate-${rate}-rep-${repetition}"
    window_dir="${study}/windows/$(printf '%02d' "${order}")-${label}"
    rep_root="${window_dir}/runs"
    mkdir -p "${rep_root}"
    benchmark_log="${window_dir}/benchmark.log"
    "${AFD_PYTHON}" -m src.benchmark \
        --trace "${trace}" \
        --base-url "http://127.0.0.1:${api_port}" \
        --model "${served_name}-attention" \
        --tokenizer Qwen/Qwen3-30B-A3B \
        --request-rate "${rate}" \
        --override-arrival-rate \
        --output-root "${rep_root}" \
        --warmup-requests 8 \
        --settle-before-s 5 \
        --settle-after-s 5 \
        --monitor-interval-s 0.2 \
        --max-concurrency 256 \
        --request-timeout-s 900 \
        --local-files-only \
        --placement-file "${placement_file}" \
        --vllm-patch-manifest patches/afd-plugin-0.26.0-static-placement.json \
        --target-clocks "${target_array[@]}" \
        --store-output-token-ids \
        --run-label "${label}" \
        --placement-name P_balanced \
        --server-log "${study}/server/attention.log" \
        > "${benchmark_log}" 2>&1
    mapfile -t run_dirs < <(find "${rep_root}" -mindepth 1 -maxdepth 1 -type d | sort)
    [[ "${#run_dirs[@]}" -eq 1 ]] || { printf 'expected one run directory\n' >&2; exit 1; }
    run_dir="${run_dirs[0]}"
    cp "${benchmark_log}" "${run_dir}/stdout-stderr.log"
    cp "${study}/actual-ownership.json" "${run_dir}/runtime-ownership.json"
    "${AFD_PYTHON}" -m tools.record_milestone_m_run_metadata \
        --plan "${plan}" --ownership "${study}/actual-ownership.json" \
        --config-id "rate-${rate}" --repetition "${repetition}" \
        --output "${run_dir}/experiment_metadata.json"
    "${AFD_PYTHON}" -m tools.validate_live_run "${run_dir}" \
        > "${run_dir}/live-validation.json"
    "${AFD_PYTHON}" -m tools.validate_milestone_m_measured_run \
        --run-dir "${run_dir}" \
        --trace-sha256 "$("${AFD_PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["trace_sha256"])' "${plan}")" \
        --placement-sha256 "${placement_sha}" \
        --mapping-sha256 "${mapping_sha}" \
        --target-clocks "${target_array[@]}" \
        > "${run_dir}/milestone-m-validation.stdout"
    printf '%s\t%s\t%s\tcomplete\t%s\n' \
        "${order}" "${rate}" "${repetition}" "${run_dir}" >> "${study}/runs.tsv"
done < <("${AFD_PYTHON}" - "${plan}" <<'PY'
import json, sys
plan = json.load(open(sys.argv[1]))
for window in sorted(plan["windows"], key=lambda row: row["order"]):
    print(window["order"], window["request_rate"], window["repetition"], sep="\t")
PY
)

expected="$("${AFD_PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["expected_runs"])' "${plan}")"
complete_count="$(awk -F '\t' '$4 == "complete" {n++} END {print n+0}' "${study}/runs.tsv")"
[[ "${complete_count}" -eq "${expected}" ]] || {
    printf 'expected %s runs, got %s\n' "${expected}" "${complete_count}" >&2
    exit 1
}
"${AFD_PYTHON}" - "${study}/result.json" "${complete_count}" <<'PY'
import json, sys
json.dump({"status": "complete", "completed_runs": int(sys.argv[2])}, open(sys.argv[1], "w"), indent=2)
PY
printf 'PASS\n' > "${study}/status.txt"
