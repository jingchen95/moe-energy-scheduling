#!/usr/bin/env bash

# =============================================================================
# 1. Slurm 资源申请
#    默认在 Uranus 的一个节点上申请 4 张 L40S，最长运行 2 小时。
#    下面的 #SBATCH 行由 Slurm 读取，不是普通的 Shell 命令。
# =============================================================================
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

# =============================================================================
# 2. 准备运行环境
# =============================================================================

# 任一命令失败、变量未定义或管道失败时立即退出。
set -euo pipefail

# sbatch 必须从仓库根目录提交；随后进入这个目录执行所有命令。
PROJECT_ROOT="${SLURM_SUBMIT_DIR:?submit this script with sbatch from the repository root}"
cd "${PROJECT_ROOT}"

source "${HOME}/miniforge3/etc/profile.d/conda.sh"
conda activate cuda-env

# 只允许使用本实验指定的单个计算节点。
case "$(hostname -s)" in
    uranus*|neptune*) ;;
    *) printf 'refusing to run outside uranus/neptune\n' >&2; exit 1 ;;
esac

# =============================================================================
# 3. 为本次实验创建独立结果目录
#    目录名同时包含 Slurm job ID 和 UTC 时间，因此不会覆盖旧结果。
# =============================================================================

output_root="${PROJECT_ROOT}/results/raw"
run_id="moe-time-${SLURM_JOB_ID}-$(date -u +%Y%m%dT%H%M%SZ)"
run_dir="${output_root}/${run_id}"
benchmark_dir="${run_dir}/benchmark"
pid_file="${run_dir}/vllm.pid"
mkdir -p "${output_root}"
mkdir "${run_dir}"

# 把这个控制脚本的 stdout/stderr 同时保存到本次实验目录。
exec > >(tee -a "${run_dir}/job.log") 2>&1

server_pid=""
benchmark_pid=""
timing_patch_applied=0

# =============================================================================
# 4. 定义退出清理
#    无论正常结束、报错还是收到 Slurm 信号，都会自动执行 cleanup。
# =============================================================================

cleanup() {
    local status=$?
    local cleanup_failed=0
    local server_stopped=1
    trap - EXIT INT TERM

    # 清理顺序：停止 benchmark -> 停止 vLLM -> 还原补丁。
    if [[ -n "${benchmark_pid}" ]] && kill -0 "${benchmark_pid}" 2>/dev/null; then
        kill -TERM "${benchmark_pid}" 2>/dev/null || true
        for _ in {1..10}; do
            kill -0 "${benchmark_pid}" 2>/dev/null || break
            sleep 1
        done
        kill -KILL "${benchmark_pid}" 2>/dev/null || true
    fi
    if [[ -n "${server_pid}" ]]; then
        if ! scripts/stop_vllm.sh "${pid_file}"; then
            cleanup_failed=1
            server_stopped=0
        fi
    fi
    if [[ "${timing_patch_applied}" -eq 1 && "${server_stopped}" -eq 1 ]]; then
        scripts/manage_moe_timing_patch.sh revert --root "${site_root}" \
            > "${run_dir}/timing-patch-revert.log" 2>&1 || cleanup_failed=1
    elif [[ "${timing_patch_applied}" -eq 1 ]]; then
        printf 'patch not reverted because the vLLM process group is still running\n' \
            > "${run_dir}/timing-patch-revert.log"
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

# EXIT 负责所有退出路径；INT/TERM 先转换为明确的退出状态。
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# =============================================================================
# 5. 检查固定输入并记录可复现实验信息
# =============================================================================

# 固定 trace 哈希，防止不同运行误用不同请求或到达时间。
trace="data/traces/gsm8k-50-r0.5.jsonl"
expected_trace_sha="721638a6250b929de05c2c355c8bc9b20258db7b28fedb94acb5572136571cbc"
actual_trace_sha="$(sha256sum "${trace}" | awk '{print $1}')"
[[ "${actual_trace_sha}" == "${expected_trace_sha}" ]] || {
    printf 'unexpected trace hash: %s\n' "${actual_trace_sha}" >&2
    exit 1
}

cp "${trace}" "${run_dir}/trace.jsonl"
cp patches/vllm-0.15.1-moe-timing.patch "${run_dir}/moe-timing.patch"
cp patches/vllm-0.15.1-moe-timing.json "${run_dir}/moe-timing-manifest.json"
sha256sum "${trace}" patches/vllm-0.15.1-moe-timing.* \
    > "${run_dir}/input-sha256.txt"
git rev-parse HEAD > "${run_dir}/git-commit.txt"
git status --short > "${run_dir}/git-status.txt"
# 只允许从干净的 Git 工作区测量，保证记录的 commit 对应实际代码。
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

# 如果 8000 端口已有 vLLM 服务，直接退出，不复用也不停止别人的服务。
if curl --fail --silent --max-time 2 http://127.0.0.1:8000/health \
    >/dev/null 2>&1; then
    printf 'refusing to reuse an existing service on 127.0.0.1:8000\n' >&2
    exit 1
fi

# =============================================================================
# 6. 临时应用 vLLM 0.15.1 的 MoE 计时补丁
#    补丁修改当前 cuda-env 中的 vLLM；cleanup 会在服务停止后还原它。
# =============================================================================

# 找到包含 vllm/ 目录的 site-packages 根目录。
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

# =============================================================================
# 7. 启动 vLLM 并等待服务就绪
# =============================================================================

# 开启开发 RPC 和本项目补丁中的 FusedMoE 计时器。
export CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_SERVER_DEV_MODE=1
export VLLM_MOE_TIMING=1

# vLLM 的模型与并行配置只在这里写一次。
vllm_command=(
    python -m vllm.entrypoints.openai.api_server
    --model Qwen/Qwen3-30B-A3B
    --served-model-name qwen3-30b-a3b
    --host 127.0.0.1
    --port 8000
    --dtype auto
    --max-model-len 4096
    --gpu-memory-utilization 0.9
    --seed 0
    # TP=1、DP=4，并让这 4 个 rank 同时组成 EP 组。
    --tensor-parallel-size 1
    --data-parallel-size 4
    --data-parallel-backend mp
    --enable-expert-parallel
    --expert-placement-strategy linear
    --all2all-backend allgather_reducescatter
    # 避免请求间 prefix cache 干扰；eager 模式便于 CUDA Event 计时。
    --no-enable-prefix-caching
    --enforce-eager
)
printf '%q ' "${vllm_command[@]}" > "${run_dir}/vllm-command.txt"
printf '\n' >> "${run_dir}/vllm-command.txt"
# setsid 创建独立进程组，使 cleanup 只停止本作业启动的 vLLM 进程。
setsid "${vllm_command[@]}" \
    > "${run_dir}/vllm.log" 2>&1 &
server_pid=$!
printf '%s\n' "${server_pid}" > "${pid_file}"

# 循环检查 /health；进程提前退出时打印 vLLM 日志末尾帮助定位问题。
ready=0
for _ in $(seq 1 360); do
    if ! kill -0 "${server_pid}" 2>/dev/null; then
        printf 'vLLM exited before becoming ready\n' >&2
        tail -n 120 "${run_dir}/vllm.log" >&2 || true
        exit 1
    fi
    if curl --fail --silent --show-error --max-time 2 \
        http://127.0.0.1:8000/health \
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

# =============================================================================
# 8. 运行固定 benchmark 并保存结果
#    Python 会完成预热、清空计时器、发送 50 个请求和计算 MoE 时间占比。
# =============================================================================

# 放到后台再 wait，使 Slurm TERM 信号可以立即触发前面的 cleanup。
python -m src.benchmark "${benchmark_dir}" \
    > "${run_dir}/benchmark.log" 2>&1 &
benchmark_pid=$!
printf '%s\n' "${benchmark_pid}" > "${run_dir}/benchmark.pid"
benchmark_status=0
wait "${benchmark_pid}" || benchmark_status=$?
benchmark_pid=""
if [[ "${benchmark_status}" -ne 0 ]]; then
    exit "${benchmark_status}"
fi

# 到达文件末尾后正常退出，并自动触发前面注册的 cleanup。
