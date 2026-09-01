# Higher-load Placement × DVFS Repeatability Check

## Method

P1/HHHH calibration measured a conservative capacity lower bound of **22.434 req/s** before client scheduling became invalid. The fixed 65% target selected **16.000 req/s**, the closest client-valid measured rate. The fixed trace contains 300 deterministic GSM8K requests and is identical across all 132 measured runs.

Four preregistered placements (P1-P4), 11 targeted H/M/L tuples, and three independent measurement windows per configuration were measured. Selection uses measured serving behavior only; expert-load diagnostics and performance models are not used.

## HHHH placement selection

| placement | TTFT p99 mean ± std (ms) | TPOT p99 mean ± std (ms) | energy mean ± std (J) |
|---|---:|---:|---:|
| P2 | 226.010 ± 3.156 | 64.481 ± 0.153 | 21368.932 ± 75.932 |
| P1 | 227.160 ± 1.063 | 64.579 ± 0.036 | 21258.167 ± 117.971 |
| P3 | 229.037 ± 3.058 | 64.778 ± 0.109 | 21507.847 ± 110.141 |
| P4 | 225.546 ± 2.542 | 64.790 ± 0.021 | 21386.568 ± 86.118 |

## Measured Pareto frontier

| placement | frequencies (MHz) | TTFT p99 mean (ms) | TPOT p99 mean (ms) | energy mean (J) |
|---|---|---:|---:|---:|
| P2 | [2520, 2520, 2520, 2520] | 226.010 | 64.481 | 21368.932 |
| P1 | [2520, 2520, 2520, 2520] | 227.160 | 64.579 | 21258.167 |
| P1 | [1635, 1635, 1635, 1635] | 256.554 | 70.667 | 16231.013 |
| P2 | [1635, 1635, 1635, 1635] | 254.884 | 70.716 | 16193.924 |
| P3 | [1005, 1635, 1635, 1635] | 276.865 | 77.672 | 15959.230 |
| P1 | [1005, 1635, 1635, 1635] | 277.664 | 77.932 | 15691.369 |
| P1 | [1635, 1005, 1635, 1635] | 281.041 | 78.006 | 15672.346 |
| P2 | [1635, 1635, 1005, 1635] | 278.463 | 78.495 | 15645.235 |
| P1 | [1635, 1635, 1635, 1005] | 288.417 | 79.973 | 15588.443 |
| P1 | [1005, 1005, 1005, 1635] | 294.515 | 82.120 | 15393.173 |
| P1 | [1635, 1005, 1005, 1005] | 291.899 | 82.570 | 15390.019 |
| P1 | [1005, 1635, 1005, 1005] | 291.963 | 82.822 | 15302.021 |
| P1 | [1005, 1005, 1005, 1005] | 295.185 | 83.637 | 15218.445 |

## Preregistered reference SLOs

| TPOT SLO | feasible | Sequential | Joint | saving |
|---:|---|---|---|---:|
| 12 ms | no | - | - | - |
| 13 ms | no | - | - | - |
| 15 ms | no | - | - | - |

## Decision

**B_WEAK_OPPORTUNITY**. The numerical maximum is 24.044% at the single exploratory 70.7 ms threshold. It is a boundary artifact, not a broad saving: the Joint point TPOT is 70.667 ± 0.398 ms while the next Sequential DVFS point is 70.716 ± 0.496 ms. Their mean separation is smaller than repeat variability. At 70.8 ms the saving is 0.000%.

Excluding >=5% one-grid boundary effects, the largest measured saving is 3.221% and also occurs only over a narrow SLO interval. The data therefore support a weak, noise-sensitive interaction, not a strong systems opportunity. Preregistered 12/13/15 ms rows remain visible above and are not replaced by a favorable post-hoc threshold.

## Limitations

The calibration found a 22.434 req/s measured capacity lower bound, not an exact server saturation point: offered rates above 24 req/s became client-scheduling-invalid. One model, one GSM8K trace, one four-L40S node, four fixed placements, 11 targeted tuples, no replication/migration, and a 500 ms TTFT guard. P99 values are summarized across three windows but remain workload- and system-specific.
