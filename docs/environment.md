# Environment

Inspection date: 2026-08-24

## Access and installation

- Access path: SSH to `minerva`, then a single-node Slurm allocation on
  `uranus` or `neptune`.
- Milestone A ran only on `uranus` (Slurm job `254959`).
- Compute-node OS: Ubuntu 24.04.4 LTS.
- Python: 3.10.17 in the `cuda-env` Conda environment.
- PyTorch: 2.9.1+cu128; PyTorch CUDA runtime: 12.8.
- CUDA compiler: 12.4 (`nvcc` 12.4.131).
- NVIDIA driver: 595.71.05; `nvidia-smi` reports CUDA compatibility 13.2.
- vLLM: 0.15.1 at
  `/data/users/chjing/miniforge3/envs/cuda-env/lib/python3.10/site-packages/vllm`.
- vLLM is a regular site-packages wheel installation. Its distribution has no
  `direct_url.json`; it is not an editable source checkout.
- NVML Python distribution: `nvidia-ml-py` 13.590.48, exposing the import name
  `pynvml`. There is no separate `pynvml` distribution metadata.
- No NCCL-specific environment variables were set in the smoke allocation.
  vLLM reported NCCL 2.27.5 at runtime.

No CUDA, driver, PyTorch, vLLM, or Python package was installed or upgraded.

## Model cache

`Qwen/Qwen3-30B-A3B` was not present in the configured/default Hugging Face
cache when job `254959` began. vLLM downloaded it during the smoke test in
148.8 seconds. It is now cached at
`/data/users/chjing/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B`
(approximately 57 GiB). Downloaded model data remains outside Git.

## L40S hardware

All four devices are NVIDIA L40S GPUs with the following identical reported
limits:

| Property | Value |
| --- | ---: |
| Memory | 46,068 MiB |
| Default/current power limit | 350 W |
| Minimum/maximum power limit | 100 W / 350 W |
| Maximum graphics/SM clock | 2,520 MHz |
| Maximum memory clock | 9,001 MHz |

The supported graphics-clock table is exposed by `nvidia-smi` rather than
assumed. It descends in 15 MHz steps from 2,520 MHz (the complete captured
table is in the ignored smoke artifact `environment.txt`).

`nvidia-smi topo -m` reported no NVLink connections. GPUs 0 and 1 are attached
to NUMA node 0 and have `NODE` connectivity to each other; GPUs 2 and 3 are on
NUMA node 1 and likewise have `NODE` connectivity. Cross-pair paths are `SYS`.

NVML `nvmlDeviceGetTotalEnergyConsumption` succeeded on GPUs 0, 1, 2, and 3.
The L40S setup therefore supports the preferred cumulative energy counter;
sampled-power trapezoidal integration remains the required fallback and
cross-check for Milestone B.

## vLLM EP/EPLB inspection

The installed vLLM exposes `--enable-expert-parallel`,
`--data-parallel-size`, `--enable-eplb`, `--expert-placement-strategy`, and
`--all2all-backend`. `vllm.distributed.eplb.policy.default.DefaultEplbPolicy`
is present.

Existing static placement accepts only built-in `linear` and `round_robin`
strategies. No placement-file option or arbitrary per-layer logical
expert-to-rank interface was found. Custom static placement therefore requires
a minimal, reproducible patch in Milestone E. No patch or site-packages edit
was made in Milestone A.

## Expert-parallel smoke test

Slurm job `254959` completed successfully on `uranus` in 23:01. Its exact vLLM
command was:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-30B-A3B \
  --served-model-name qwen3-30b-a3b \
  --host 127.0.0.1 \
  --port 8000 \
  --dtype auto \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.9 \
  --seed 0 \
  --tensor-parallel-size 1 \
  --data-parallel-size 4 \
  --data-parallel-backend mp \
  --all2all-backend allgather_reducescatter \
  --enable-expert-parallel
```

Runtime rank assignment confirmed TP rank 0 on every worker, DP ranks 0-3,
and EP ranks 0-3. Each EP rank received 32 of Qwen3's 128 experts under linear
placement. vLLM selected `AgRsAll2AllManager`, confirming the configured
all-gather/reduce-scatter all-to-all backend. The unquantized MoE backend was
Triton.

Model weights occupied 16.38 GiB per GPU, so TP=1/DP=4/EP=4 fits. Cold model
loading took 1,236.3 seconds, dominated by shared-storage reads after the first
download. This startup time is outside all future warm-up/measurement windows.
The resulting KV cache held 236,576 tokens per rank and vLLM reported maximum
concurrency 57.76x at the configured 4,096-token context length.

The OpenAI-compatible endpoint reached application-ready state and returned
HTTP 200 for all three deterministic, content-based smoke prompts (general,
code, and math). The job wrote `PASS`, exited 0, and its recorded server process
group was terminated. vLLM 0.15.1 emitted semaphore/NCCL finalizer warnings
when the first version of the stop script signaled the entire group; the script
now signals the recorded parent first and escalates only that exact group if it
does not exit within 60 seconds.

Raw evidence is retained locally and on Minerva under the ignored directory:

```text
results/raw/smoke-254959-20260824T114641Z/
```
