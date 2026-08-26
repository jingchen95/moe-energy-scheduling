# Milestone D: Placement-DVFS Coupling Pilot

## Research question

Does expert placement change per-GPU DVFS sensitivity?

## Established premises

Prior work establishes expert-load skew, workload-dependent activation, and
performance benefits from workload- or communication-aware placement. This
project does not claim those observations as new. Milestone D tests only the
causal chain from placement to per-GPU work and DVFS sensitivity.

## Method

- Hardware: one exclusive Uranus node with 4 NVIDIA L40S GPUs.
- Model/runtime: Qwen/Qwen3-30B-A3B, vLLM 0.15.1, TP=1, DP=4, EP=4,
  `allgather_reducescatter`, no expert replication.
- Workload: fixed 50-request GSM8K trace at 0.5 requests/s. Input tokens are
  128-191 (mean 142.98, p50 140, p90 160.1); every request generates 128
  tokens with temperature 0 and seed 0.
- Window: warm-up and settling precede cumulative-NVML energy start; the end
  counter is read after all measured requests finish. Model loading and
  profiling are outside compared windows.
- Frequencies: high 2520 MHz, medium 1635 MHz, low 1005 MHz. Memory clocks were
  not changed. Measured average SM clocks matched every target.
- P0 is vLLM linear placement. P1 is the frozen mapping returned by upstream
  `DefaultEplbPolicy`. P2 deterministically puts the highest-load experts on
  lower-numbered ranks while retaining exactly 32 experts per rank per layer.
- Frozen activation counts came from default placement at 2520 MHz. Static
  placement uses the recorded vLLM 0.15.1 patch and upstream expert-map and
  weight-transfer machinery.

The complete custom-placement campaign is Slurm job 255131. It resumes the
validated frozen profile and P0 curve from job 255129. The executable benchmark
path is unchanged between those revisions; the later commits affect validation
and resume orchestration. All 29 manifest rows completed without request
failures, used the NVML total-energy counter, and had at most 2.61 ms p99 client
send delay.

## Placement statistics

| Placement | GPU0 load | GPU1 load | GPU2 load | GPU3 load | Max/mean |
|---|---:|---:|---:|---:|---:|
| P0 default linear | 5,887,013 | 4,868,805 | 5,177,445 | 5,051,569 | 1.1221 |
| P1 upstream EPLB | 5,246,175 | 5,246,332 | 5,246,164 | 5,246,161 | 1.00002 |
| P2 concentrated | 15,387,711 | 3,481,809 | 1,642,426 | 472,886 | 2.9331 |

These are predicted routed-token loads from the one frozen profile, not direct
per-GPU execution-time measurements.

## Uniform-frequency results

| Placement | MHz | TTFT p99 (ms) | TPOT p99 (ms) | E2E p99 (ms) | Energy (kJ) | J/request |
|---|---:|---:|---:|---:|---:|---:|
| P0 | 2520 | 90.46 | 11.463 | 1542.8 | 67.743 | 1354.85 |
| P0 | 1635 | 301.54 | 12.922 | 1921.2 | 52.433 | 1048.66 |
| P0 | 1005 | 128.27 | 15.748 | 2117.2 | 47.494 | 949.87 |
| P1 | 2520 | 92.07 | 11.195 | 1507.3 | 66.898 | 1337.97 |
| P1 | 1635 | 85.48 | 12.478 | 1666.7 | 52.548 | 1050.96 |
| P1 | 1005 | 338.37 | 15.225 | 2259.2 | 47.949 | 958.98 |
| P2 | 2520 | 233.69 | 15.734 | 2217.9 | 64.609 | 1292.19 |
| P2 | 1635 | 126.66 | 17.587 | 2350.6 | 48.266 | 965.31 |
| P2 | 1005 | 138.41 | 19.997 | 2666.7 | 44.443 | 888.86 |

TTFT p99 contains non-monotonic outliers and is not used alone to infer DVFS
sensitivity. TPOT and energy are substantially more repeatable.

## Single-GPU perturbation results

Each row compares one GPU at 1635 MHz, with the other three at 2520 MHz,
against the same placement at all-high frequency. Negative energy values are
savings; no idle-energy subtraction is used.

| Placement | GPU | Predicted load | Delta TPOT p99 (ms) | Delta E2E p99 (ms) | Delta total energy (kJ) | Delta GPU energy (kJ) |
|---|---:|---:|---:|---:|---:|---:|
| P1 | 0 | 5,246,175 | +0.942 | +123.35 | -3.164 | -3.625 |
| P1 | 1 | 5,246,332 | +0.895 | +114.82 | -2.772 | -3.458 |
| P1 | 2 | 5,246,164 | +0.889 | +117.15 | -2.782 | -3.507 |
| P1 | 3 | 5,246,161 | +0.724 | +89.37 | -3.147 | -3.872 |
| P2 | 0 | 15,387,711 | +1.719 | +110.25 | -6.416 | -5.217 |
| P2 | 1 | 3,481,809 | +0.802 | -8.94 | -4.390 | -3.631 |
| P2 | 2 | 1,642,426 | +0.844 | -8.51 | -4.220 | -3.528 |
| P2 | 3 | 472,886 | +0.913 | +5.35 | -4.211 | -3.607 |

Under P2, GPU0's TPOT penalty is about 2.0x the median penalty of GPUs 1-3.
Across the four P2 ranks, predicted load has Pearson correlation 0.960 with
TPOT penalty and 0.994 with total energy saved. With only four ranks and one
dominant point, these correlations are descriptive, not population estimates.

The same GPU0 perturbation changes from +0.942 ms under P1 to +1.719 ms under
P2, an 82% larger p99 TPOT penalty. The repeated means show the same result:
P1 changes from 11.115 to 12.075 ms (+0.960 ms), while P2 changes from 15.746
to 17.403 ms (+1.657 ms). TPOT p99 standard deviations across the three trials
are 0.115-0.124 ms for P1 and 0.021-0.043 ms for P2. Energy standard deviations
are 0.14-0.42 kJ.

At all-high frequency, P2 also exposes the intended rank-level intervention:
GPU0 averages 212.5 W and 21.36 kJ, while GPU3 averages 135.6 W and 13.60 kJ.
P1 is much flatter at 163.7-170.4 W and 16.32-17.33 kJ. SM utilization is not
a useful imbalance proxy here: P2 reports about 99.5% on every rank because EP
collectives synchronize the ranks even when their expert work differs.

## H1 assessment

**Supported for this pilot.** The same GPU0 frequency perturbation has a
reproducibly larger latency penalty and energy effect after placement moves a
large share of predicted expert work onto GPU0. The hot rank also separates
from the three cool ranks within P2. This is material placement-DVFS coupling,
but the result is demonstrated using an intentionally strong diagnostic
placement and does not establish the size of the gap between practical
placement policies.

## Preliminary H2 evidence

No positive H2 conclusion is warranted. At relative p99 TPOT budgets of 1.05x,
1.10x, and 1.20x versus each placement's all-high reference, all three
placements select the same frequency levels: high, high, and medium,
respectively. H2 remains open for heterogeneous DVFS and common absolute SLOs.

## Limitations

- One model, one GSM8K trace, one L40S node, and one request rate.
- No expert replication, migration, communication-aware placement, or Joint
  optimizer.
- P2 is a causal diagnostic, not a competitive policy.
- Exact greedy outputs are stable across independent P0 launches, but custom
  placement changes fused-MoE numerical ordering. P1 and P2 share at least the
  first 15 output tokens with P0 on all 15 validation requests, then some token
  sequences diverge. This is recorded and may slightly change later activation.
- TTFT p99 is noisy at this low request rate; TPOT provides the stronger H1
  evidence.
- The vLLM collective profiling RPC intermittently hangs. The accepted frozen
  profile is preserved, and no profiling RPC runs inside P1/P2 measurements.

The compact source tables are in `results/processed/`, and the diagnostic plot
is `results/figures/milestone_d_dvfs_sensitivity.pdf`.
