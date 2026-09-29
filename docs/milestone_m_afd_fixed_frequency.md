# Milestone M: AFD Fixed-Frequency Expert Placement

## Research question

Milestone L found negligible serving-level benefit from aligning intact expert
groups with heterogeneous whole-GPU frequencies in the colocated runtime.
Milestone M asks whether separating Attention from the FFN pool exposes a
measurable individual-expert placement by fixed FFN-frequency interaction.

No per-layer or in-run frequency switching is part of this milestone.

## Stage 1: isolated runtime

The protected `cuda-env` remains the vLLM 0.15.1 environment used by the
colocated study. The AFD runtime is separate and project-local:

- repository: `https://github.com/vllm-project/afd-plugin.git`
- pinned commit: `4c2541a53ec8ad507508073eac81ac9feecc6c9a`
- plugin version: `0.26.0rc2.dev61+g4c2541a53`
- isolated vLLM: `0.26.0`
- isolated PyTorch: `2.11.0`
- bootstrap: project-local `uv 0.12.20`

The pinned plugin advertises generic `2A2F` runner scenarios, but its Qwen3
MoE E2E module currently exercises only `2A1F`. Therefore Qwen3 2A2F on the
four L40S GPUs is an explicit smoke-test gate, not an assumed working state.

The source-controlled environment record is
`configs/milestone_m_environment.json`. The large plugin checkout, virtual
environment, bootstrap environment, and package cache are ignored by Git.

Validation results:

- repository CPU suite in the protected vLLM 0.15.1 environment: `68 passed`;
- pinned plugin `tests/unit` CPU suite, excluding GPU/E2E/slow/NPU markers:
  passed with exit status 0;
- `git diff --check`: passed.

## Planned topology

- GPU0/GPU1: Attention ranks, fixed at 2520 MHz.
- GPU2/GPU3: FFN ranks, tested at HH, MH, and HM.
- Synchronous `P2pNcclAFDConnector` with gate computation on the FFN role.
- Qwen3-30B-A3B, BF16, TP=1, PP=1, no EPLB or expert replication.

## Required stage gates

1. Unmodified eager Qwen3 2A2F smoke on one exclusive L40S node.
2. Static individual-expert placement correctness at HH.
3. AFD-native frozen routing profile collection.
4. One MH aligned-placement smoke.
5. Only then, the randomized 18-run primary matrix.

## Current status

Environment preparation is complete. The unmodified-plugin smoke launcher is
prepared as `scripts/milestone_m_afd_validation.sbatch` and intentionally does
not contain placement or profiling changes. It launches two owned process
groups using the isolated interpreter:

- Attention API: GPUs 0--1, port 18100;
- FFN service: GPUs 2--3, port 18101;
- synchronous connector: port 6249.

The job sends the same three fixed greedy prompts twice to the Attention API,
requires exact generated-content/token-count equality, and samples all four
GPUs to confirm each role used its assigned devices. Cleanup stops only the
recorded process groups and resets graphics clocks. The current project CPU
suite passes `71` tests.

Real-GPU smoke validation, placement instrumentation, profiling, and primary
measurements are pending.

### First smoke attempt

Slurm job `269841` failed before either API became healthy. Both FFN ranks
reached their first Triton FusedMoE kernel invocation, then Triton's CUDA
driver helper compilation failed because the system-Python virtual environment
looked for `/usr/include/python3.12/Python.h`, which was absent on Uranus. This
was an environment construction problem, not evidence about Qwen3 2A2F,
placement, NCCL, latency, or energy.

The job exited 1 after 9 minutes 29 seconds. Cleanup reset all four graphics
clocks, left no GPU compute processes, and did not dirty the plugin checkout.
The protected environment remained vLLM 0.15.1.

The isolated runtime was rebuilt with a fixed uv-managed CPython 3.12.3 stored
under `.tools/uv-python`. Its `Python.h` is also on shared project storage, so
the compute node no longer depends on host `/usr/include`. The environment
guard now records and requires the Python include directory and header.

### Second smoke attempt

Slurm job `269912` also failed before either API became healthy, but it
confirmed that the project-local Python-header fix worked: both roles loaded
far enough to exercise their CUDA paths, and the FFN role passed the prior
Triton helper build. The Attention role then selected FlashInfer for top-k and
top-p sampling during vLLM's memory-profile pass. FlashInfer attempted a JIT
build and failed because Uranus has no discoverable `nvcc` or
`/usr/local/cuda`. The FFN ranks subsequently timed out in a synchronous NCCL
receive after 30 minutes; that timeout is a consequence of the Attention
failure, not the initiating error.

The job exited 1 after 35 minutes 5 seconds. Cleanup reset all graphics clocks
and left no GPU compute processes. No request was served, so this remains a
runtime-preparation failure rather than a 2A2F result.

vLLM 0.26 exposes `VLLM_USE_FLASHINFER_SAMPLER=0` specifically to use its
native sampling path. The project launcher now sets this only in owned AFD
role processes. This avoids an undeclared CUDA-toolkit dependency without
installing CUDA, modifying the pinned plugin/vLLM packages, or changing greedy
decoding parameters.
