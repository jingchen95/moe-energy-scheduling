# Milestone L: Real-GPU P1-group alignment validation

## Research question

At identical heterogeneous frequency tuples, does permuting each layer's four
intact upstream-EPLB groups onto physical GPUs improve measured latency or
energy relative to original P1?

Milestone K's 15--18% pressure-proxy reductions are structural evidence only.
This milestone tests their runtime consequence; it does not treat the proxy as
measured performance.

## Method and validation

- Qwen3-30B-A3B, vLLM 0.15.1, TP=1/DP=4/EP=4;
- one exclusive four-L40S Uranus allocation per configuration;
- Milestone I's deterministic 300-request GSM8K trace at 16 req/s;
- greedy decoding, seed 0, and 128 output tokens;
- three measurement windows per configuration;
- cumulative NVML GPU energy without idle subtraction;
- H=2520 MHz and M=1635 MHz.

Slurm array `267080` completed all seven serialized tasks with exit status 0.
All 21 runs completed 300/300 requests with zero failures, one trace hash,
cumulative NVML energy, and target-matching average SM clocks. Client send
delay p99 was at most 9.693 ms versus a 62.5 ms inter-arrival interval. Every
controller executed four-GPU clock reset and reverted the vLLM patch; the final
global patch state is `not applied`.

The aligned mappings preserve every original P1 group exactly: all 128 experts
appear once, every GPU receives 32 experts per layer, and only whole-group rank
labels change. Placement SHA-256 values are:

- original P1: `0912c7744e87dc35b223ee8d145d6a6ac8bc6f3274a642b5cbb20d4a011454e7`;
- aligned MHHH: `1ff6613f61a18772049e6fb743e22ff2fd84789e1b0e77e529f8dafb7488cae1`;
- aligned MMHH: `b69101216e21d7274d3564ad8a14d7cb498c47d183fdc6e5135516b4da7b7410`.

Primary runs did not enable FusedMoE CUDA timing. The available timing path
requires extra patches and `--enforce-eager`; using it would change the primary
Milestone I runtime configuration and has unquantified overhead.

## Aggregate results

Values are mean +/- sample standard deviation across three repetitions.

| ID | Placement / clocks | TTFT p99 (ms) | TPOT p99 (ms) | GPU energy (J) | Throughput (req/s) |
|---|---|---:|---:|---:|---:|
| C1 | original / HHHH | 234.678 +/- 12.670 | 64.602 +/- 0.313 | 21,213.773 +/- 187.056 | 12.794 +/- 0.044 |
| C2 | align-MHHH / HHHH | 234.682 +/- 9.454 | 64.932 +/- 0.703 | 21,147.132 +/- 370.404 | 12.793 +/- 0.051 |
| C3 | align-MMHH / HHHH | 232.208 +/- 8.092 | 64.505 +/- 0.506 | 21,200.918 +/- 249.533 | 12.783 +/- 0.003 |
| C4 | original / MHHH | 243.274 +/- 3.079 | 68.032 +/- 0.813 | 19,765.370 +/- 209.735 | 12.624 +/- 0.022 |
| C5 | align-MHHH / MHHH | 250.637 +/- 3.235 | 67.740 +/- 0.232 | 19,818.202 +/- 205.604 | 12.659 +/- 0.039 |
| C6 | original / MMHH | 247.933 +/- 3.404 | 68.599 +/- 0.288 | 18,667.898 +/- 90.671 | 12.586 +/- 0.013 |
| C7 | align-MMHH / MMHH | 250.762 +/- 5.916 | 68.671 +/- 0.356 | 18,667.292 +/- 131.046 | 12.587 +/- 0.048 |

All configurations satisfy the 500 ms TTFT guard. As in Milestone I at this
load, none satisfies the historical 12/13/15 ms TPOT reference thresholds;
alignment does not change SLO feasibility.

## Primary comparisons

Signed changes below are `(aligned - original) / original`; negative values
are improvements for latency and energy, while positive is better for
throughput.

| Comparison | TTFT p99 | TPOT p99 | Energy | Throughput |
|---|---:|---:|---:|---:|
| MHHH: C5 vs C4 | +3.027% | -0.428% | +0.267% | +0.278% |
| MMHH: C7 vs C6 | +1.141% | +0.106% | -0.003% | +0.010% |
| HHHH: C2 vs C1 | +0.002% | +0.511% | -0.314% | -0.008% |
| HHHH: C3 vs C1 | -1.052% | -0.150% | -0.061% | -0.081% |

The heterogeneous differences are smaller than, or comparable to, the
three-run spread and are internally mixed: MHHH slightly improves TPOT but
worsens TTFT and energy; MMHH is effectively identical. No statistical
significance claim is made from three repetitions. The HHHH controls are also
approximately neutral, so the result is not classified as a physical-rank
control failure.

## Individual runs

| ID / rep | TTFT p99 (ms) | TPOT p99 (ms) | Energy (J) | Throughput (req/s) |
|---|---:|---:|---:|---:|
| C1 / 0 | 227.416 | 64.584 | 21,321.322 | 12.743 |
| C1 / 1 | 249.308 | 64.924 | 20,997.780 | 12.818 |
| C1 / 2 | 227.310 | 64.298 | 21,322.217 | 12.820 |
| C2 / 0 | 231.802 | 64.635 | 21,447.606 | 12.741 |
| C2 / 1 | 227.003 | 64.427 | 21,260.496 | 12.794 |
| C2 / 2 | 245.240 | 65.735 | 20,733.294 | 12.844 |
| C3 / 0 | 229.151 | 64.248 | 21,385.755 | 12.784 |
| C3 / 1 | 241.383 | 65.088 | 20,917.076 | 12.780 |
| C3 / 2 | 226.089 | 64.178 | 21,299.923 | 12.786 |
| C4 / 0 | 241.703 | 67.408 | 19,799.581 | 12.648 |
| C4 / 1 | 241.298 | 67.736 | 19,955.896 | 12.615 |
| C4 / 2 | 246.821 | 68.951 | 19,540.633 | 12.608 |
| C5 / 0 | 248.912 | 67.747 | 19,864.648 | 12.658 |
| C5 / 1 | 254.370 | 67.969 | 19,593.348 | 12.698 |
| C5 / 2 | 248.630 | 67.505 | 19,996.610 | 12.620 |
| C6 / 0 | 244.593 | 68.414 | 18,751.246 | 12.592 |
| C6 / 1 | 251.397 | 68.930 | 18,571.351 | 12.571 |
| C6 / 2 | 247.809 | 68.451 | 18,681.098 | 12.594 |
| C7 / 0 | 257.558 | 68.564 | 18,782.427 | 12.544 |
| C7 / 1 | 246.767 | 68.381 | 18,694.761 | 12.639 |
| C7 / 2 | 247.960 | 69.068 | 18,524.689 | 12.579 |

## Load and per-GPU diagnostics

The frozen profile confirms that the intended structural intervention occurred.
For align-MHHH, GPU0's aggregate expert-token fraction is 0.199 versus
0.265/0.268/0.268 on GPUs 1--3. For align-MMHH, slow GPUs 0--1 receive
0.222/0.213 versus 0.282/0.284 on fast GPUs 2--3. This holds per layer by
construction, with similar prefill/decode aggregate fractions.

The intervention redistributes energy between ranks but not total energy. For
MHHH, alignment changes mean per-GPU energy from
`[4017.9, 5160.5, 5247.7, 5339.3]` J to
`[3992.7, 5187.0, 5277.7, 5360.8]` J: the slow GPU decreases by about 25 J,
while the fast GPUs increase enough that total energy rises by 53 J. For MMHH,
per-GPU energy changes from `[4022.6, 4045.1, 5248.5, 5351.7]` J to
`[3998.8, 4026.5, 5282.8, 5359.2]` J, nearly cancelling in total. Coarse SM
utilization remains 98.3--98.4% on every rank and cannot expose critical-path
wait time. FusedMoE timing was not collected, so no claim is made about which
rank dominates the MoE critical path.

## Decision: C -- no measured effect

The strong Milestone K alignment-proxy improvement does not produce a
practically meaningful measured latency, energy, throughput, or SLO benefit in
this controlled experiment. The maximum favorable heterogeneous change is a
0.428% TPOT reduction for MHHH, accompanied by 0.267% higher energy and 3.027%
worse TTFT. MMHH is indistinguishable at the reported scale.

Expert-token load alone is therefore insufficient to predict the real serving
critical path under vLLM 0.15.1 on this model and hardware. A broader joint
placement/DVFS scheduler is not justified by this mechanism. Any further work
should first isolate communication, synchronization/waiting, Attention/shared
computation, and nonlinear frequency sensitivity with non-perturbing timing,
rather than expanding the placement or DVFS search.

## Artifacts

- Raw study: `results/raw/milestone-l-alignment-20260923T141055Z`
- Per-run metrics: `results/processed/milestone_l_runs.csv`
- Three-repeat aggregates: `results/processed/milestone_l_aggregate.csv`
- Comparisons: `results/processed/milestone_l_comparisons.csv`
- Machine-readable summary: `results/processed/milestone_l_summary.json`
- Figure: `results/figures/milestone_l_alignment_validation.svg`
