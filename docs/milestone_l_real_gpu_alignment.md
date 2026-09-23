# Milestone L: Real-GPU P1-group alignment validation

## Status

Submitted as serialized Slurm array `267080_[0-6%1]`. The immutable study is
`results/raw/milestone-l-alignment-20260923T141055Z`. This document will be
updated with measured results after the array completes.

## Registered question

At identical heterogeneous frequency tuples, does permuting each layer's four
intact upstream-EPLB groups onto physical GPUs improve measured latency or
energy relative to original P1?

Milestone K's 15--18% pressure-proxy reductions are structural evidence only.
They are not treated as measured performance improvements.

## Fixed method

- Qwen3-30B-A3B, vLLM 0.15.1, TP=1/DP=4/EP=4;
- one exclusive four-L40S Uranus allocation per configuration;
- validated Milestone I 300-request GSM8K trace at 16 req/s;
- deterministic greedy decoding with 128 output tokens;
- three independent measured windows per configuration;
- cumulative NVML GPU energy with no idle subtraction;
- H=2520 MHz and M=1635 MHz only.

The seven registered configurations are C1 original/HHHH, C2 aligned-MHHH/
HHHH, C3 aligned-MMHH/HHHH, C4 original/MHHH, C5 aligned-MHHH/MHHH, C6
original/MMHH, and C7 aligned-MMHH/MMHH. Their serialized task order and
within-task repetition order are randomized with seed `20260923`.

The aligned files preserve the original P1 expert membership of every group;
only whole-group physical-rank labels change. Construction uses the frozen
total expert counts and the exact Milestone K exhaustive objective and
tie-breakers.

## Instrumentation boundary

Primary runs do not enable FusedMoE CUDA timing. The existing timing path
requires extra vLLM patches and `--enforce-eager`, and its overhead/equivalence
is not established. Enabling it would contaminate the primary Milestone I
runtime comparison. A separate diagnostic pass may be considered only after
the primary results are valid.

## Submission record

The study was generated from preparation commit
`c63b9f3a176d4d6656a010f504563c55d5398a72`. The recorded configuration order
is `C4, C6, C1, C5, C7, C2, C3`; array concurrency is limited to one task.

```bash
squeue -j 267080
```
