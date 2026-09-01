# Ground-truth Placement × DVFS Motivation Sweep

## Research question

Does exhaustive measured Joint placement plus per-GPU DVFS find a lower-energy SLO-feasible point than the measured Placement→DVFS Sequential Oracle? No performance or service model is fit or used.

## Method and validation

Qwen/Qwen3-30B-A3B, vLLM 0.15.1, TP=1/DP=4/EP=4, four L40S GPUs on Uranus, `allgather_reducescatter`, fixed GSM8K 50-request trace at 0.5 requests/s, and H/M/L = 2520/1635/1005 MHz. P0 is default linear, P1 is the accepted upstream EPLB mapping, and P2-P4 are the preregistered controlled-medium/high/very-high mappings.

The matrix contains five placements × 81 tuples = 405 measured/reused rows. The newly measured placements used one shared seeded tuple permutation (seed `20260831`). Every row completed 50/50 requests with zero failures, used one trace hash, exposed cumulative NVML energy, and passed target-versus-actual clock validation. P1 was reused only after the same trace, mapping, runtime, request, energy, and clock checks. Selection uses measured TTFT p99, TPOT p99, and total energy only; no performance model is used.

## All-high placement selection

| placement | TTFT p99 (ms) | TPOT p99 (ms) | total energy (J) |
|---|---:|---:|---:|
| P1 upstream_eplb | 92.620 | 11.183 | 67412.409 |
| P0 default_linear | 91.924 | 11.460 | 67928.225 |
| P2 controlled_medium | 85.021 | 11.785 | 68321.784 |
| P4 controlled_very_high | 85.516 | 12.259 | 68566.325 |
| P3 controlled_high | 85.462 | 12.300 | 69210.434 |

## Sequential Oracle

At `(H,H,H,H)`, the performance-selected placement was **P1 (upstream_eplb)**, using TPOT p99 as the primary metric and TTFT p99 <= 500 ms as a hard constraint.

## Placement diagnostics

These loads are recorded for interpretation only and are not used to select points.

| placement | aggregate GPU loads | prefill GPU loads | decode GPU loads | aggregate max/mean |
|---|---|---|---|---:|
| P0 default_linear | [5486348, 4975237, 4801756, 5101715] | [4828159, 4385263, 4236514, 4476720] | [658189, 589974, 565242, 624995] | 1.078 |
| P1 upstream_eplb | [5131284, 5031500, 5153294, 5048978] | [4518439, 4423017, 4536607, 4448593] | [612845, 608483, 616687, 600385] | 1.012 |
| P2 controlled_medium | [10375266, 3199230, 3440164, 3350396] | [9140770, 2808216, 3026761, 2950909] | [1234496, 391014, 413403, 399487] | 2.038 |
| P3 controlled_high | [12946751, 2476175, 2464676, 2477454] | [11369837, 2183154, 2180712, 2192953] | [1576914, 293021, 283964, 284501] | 2.543 |
| P4 controlled_very_high | [14002072, 2112491, 2099162, 2151331] | [12276344, 1874548, 1865378, 1910386] | [1725728, 237943, 233784, 240945] | 2.750 |

## Sequential versus Joint Oracle

| SLO | Sequential placement | Sequential DVFS | Sequential energy (J) | Joint placement | Joint DVFS | Joint energy (J) | Saving |
|---|---|---|---:|---|---|---:|---:|
| TPOT <= 12 ms, TTFT <= 500 ms | P1 | [1635, 2520, 2520, 2520] | 63906.976 | P1 | [1635, 2520, 2520, 2520] | 63906.976 | 0.0000% |
| TPOT <= 13 ms, TTFT <= 500 ms | P1 | [1635, 1635, 1635, 1635] | 52474.447 | P0 | [1635, 1635, 1635, 1635] | 52421.711 | 0.1005% |
| TPOT <= 15 ms, TTFT <= 500 ms | P1 | [1005, 1005, 1005, 1635] | 48661.697 | P1 | [1005, 1005, 1005, 1635] | 48661.697 | 0.0000% |

## Assessment

At TPOT <= 13 ms, the measured Joint Oracle changes from P1 to P0 at tuple `[1635, 1635, 1635, 1635]`, reducing energy by 52.736 J (0.1005%).

At 12 ms and 15 ms the two oracles select exactly the same measured point. The only strict gap is the single-run 13 ms result above. Its 0.1005% magnitude is not repeatability-confirmed and is too small to support a meaningful Joint-energy advantage claim from this matrix alone.

## Data and limitations

The complete normalized measured table is `results/processed/milestone_h_ground_truth_matrix.csv`; the oracle table is `results/processed/milestone_h_oracle_summary.csv`. Every row retains actual clocks, trace/config hashes, energy, throughput, failures, and diagnostic placement loads. Raw run directories are not committed.

This is one model, one 50-request GSM8K trace, one L40S node, one run per newly measured point, four active GPUs, and exactly 32 experts per rank. It does not optimize active-GPU count, replication, migration, or communication. The selected 13 ms P0/P1 points and the closest all-high placement competitors require repeats before any paper-level conclusion.
