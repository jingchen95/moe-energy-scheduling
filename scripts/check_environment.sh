#!/usr/bin/env bash
set -u

section() {
    printf '\n===== %s =====\n' "$1"
}

PYTHON_BIN="${PYTHON:-python}"

section "hostname"
hostname || true

section "operating system"
if [[ -r /etc/os-release ]]; then
    cat /etc/os-release
else
    uname -a || true
fi

section "python pytorch cuda and vllm"
"${PYTHON_BIN}" - <<'PY' || true
import importlib.metadata as metadata
import importlib.util
from pathlib import Path
import sys

print("python_version", sys.version.replace("\n", " "))
for package in ("torch", "vllm"):
    spec = importlib.util.find_spec(package)
    print(f"{package}_location", spec.origin if spec else "NOT_INSTALLED")
try:
    import torch
    print("torch_version", torch.__version__)
    print("cuda_runtime", torch.version.cuda)
    print("cuda_available", torch.cuda.is_available())
except Exception as exc:
    print("torch_error", type(exc).__name__, str(exc))
try:
    import vllm
    print("vllm_version", vllm.__version__)
    print("vllm_package_root", Path(vllm.__file__).resolve().parent)
    distribution = metadata.distribution("vllm")
    direct_url = distribution.read_text("direct_url.json")
    print("vllm_install_type", "editable/direct-url" if direct_url else "site-packages wheel")
    print("vllm_direct_url", direct_url.strip() if direct_url else "NONE")
except Exception as exc:
    print("vllm_error", type(exc).__name__, str(exc))
for distribution_name in ("nvidia-ml-py", "pynvml"):
    try:
        distribution = metadata.distribution(distribution_name)
        print(distribution_name, distribution.version, distribution.locate_file(""))
    except metadata.PackageNotFoundError:
        print(distribution_name, "NOT_INSTALLED")
PY

section "cuda compiler"
if command -v nvcc >/dev/null 2>&1; then
    nvcc --version || true
else
    printf 'nvcc NOT_FOUND\n'
fi

section "cuda driver and gpu inventory"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,memory.total,driver_version,power.limit,power.min_limit,power.max_limit,clocks.current.graphics,clocks.current.sm,clocks.current.memory,clocks.max.graphics,clocks.max.sm,clocks.max.memory --format=csv,noheader || true

    section "gpu supported clocks gpu0"
    nvidia-smi -i 0 -q -d SUPPORTED_CLOCKS,CLOCK,POWER || true

    section "gpu topology"
    nvidia-smi topo -m || true
else
    printf 'nvidia-smi NOT_FOUND\n'
fi

section "nccl environment"
env | LC_ALL=C sort | awk -F= '$1 ~ /^NCCL_/ {print}' || true

section "vllm and hugging face environment"
env | LC_ALL=C sort | awk -F= '$1 ~ /^(VLLM_|HF_|HUGGINGFACE_|TRANSFORMERS_)/ {print}' || true

section "qwen3 model cache"
cache_found=0
for cache_root in "${HF_HOME:-}" "${HUGGINGFACE_HUB_CACHE:-}" "${HOME}/.cache/huggingface/hub"; do
    [[ -n "${cache_root}" ]] || continue
    while IFS= read -r cached_model; do
        printf '%s\n' "${cached_model}"
        cache_found=1
    done < <(find "${cache_root}" -maxdepth 1 -type d -iname 'models--Qwen--Qwen3-30B-A3B*' -print 2>/dev/null)
done
[[ "${cache_found}" -eq 1 ]] || printf 'NOT_FOUND_IN_CONFIGURED_CACHES\n'

section "nvml total energy support"
"${PYTHON_BIN}" - <<'PY' || true
try:
    import pynvml
    pynvml.nvmlInit()
    for index in range(pynvml.nvmlDeviceGetCount()):
        handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        try:
            energy_mj = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
            print(index, "SUPPORTED", energy_mj, "mJ")
        except pynvml.NVMLError as exc:
            print(index, "UNSUPPORTED", type(exc).__name__, str(exc))
    pynvml.nvmlShutdown()
except Exception as exc:
    print("NVML_PROBE_ERROR", type(exc).__name__, str(exc))
PY
