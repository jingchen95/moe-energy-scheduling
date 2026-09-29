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

Environment preparation is complete. GPU smoke validation, placement
instrumentation, profiling, and primary measurements are pending.
