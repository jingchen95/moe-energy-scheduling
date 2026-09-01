# Higher-load Placement × DVFS Repeatability Check

## Status

Prepared for P1/HHHH offered-load calibration. This document contains no
Milestone I measurement result yet.

## Research question

Did Milestone H's weak 0–0.1% Joint Oracle gap result from a lightly loaded
0.5 req/s trace and noisy 50-request p99 estimates?

This is a final measured motivation check. It does not fit a service model,
build a scheduler, change a placement, or use expert-load diagnostics for
selection.

## Fixed workload

- Model/runtime: Qwen3-30B-A3B, vLLM 0.15.1, TP=1/DP=4/EP=4.
- Hardware: one exclusive Uranus node with four L40S GPUs.
- Trace seed: `20260901`.
- Requests: 300 unique GSM8K prompts, deterministic greedy generation, 128
  output tokens.
- Prompt lengths: all 225 available 128–256-token prompts plus 75 samples from
  the adjacent 120–127-token range; mean/p50/p90 is 139.49/135/161 tokens.

## Offered-load calibration

P1 upstream EPLB at HHHH is measured at the fixed offered-rate set
0.5/1.0/1.5/2.0/2.5/3.0/3.5/4.0 req/s using 100 requests per rate. Seed
`20260902` randomizes rate order. Saturation and the approximately 65% target
are selected before any P2/P3/P4 higher-load result is observed.

The initial sweep reached 3.675 req/s at 4.0 req/s offered load, or 91.9% of
offered load, so it did not bracket saturation under the preregistered 90%
criterion. Before observing any P2/P3/P4 result, a fixed extension at
4.5/5.0/5.5/6.0 req/s was added with order seed `20260905`.

The extension reached 5.177 req/s at 6.0 offered and still had a substantial
positive throughput slope. A final plateau check therefore uses 300 requests
at 7/8/9/10 req/s with seed `20260906`. Saturation is accepted only when the
three highest-rate throughputs are all within 95% of the measured peak and the
highest offered rate is at least 1.2× that peak. This rule was fixed before any
P2/P3/P4 result was observed.

## Fixed measured matrix

After calibration, the exact same 300-request trace is used for P1/P2/P3/P4.
Each placement measures HHHH, MMMM, LLLL, all four MMML permutations, and all
four LLLM permutations. Every one of the 44 configurations receives three
independent measurement windows, for 132 runs total. Placement block order and
within-block run order use recorded fixed seeds `20260903` and `20260904`.

Analysis reports repeated mean and sample standard deviation, the complete
measured energy–TPOT Pareto frontier, preregistered 12/13/15 ms references, and
a 0.1 ms exploratory SLO grid across the measured frontier. TTFT p99 remains a
500 ms hard guard. The result will be classified A (strong), B (weak), or C
(not promising) without forcing a positive interpretation.
