# Ground-truth Placement × DVFS Motivation Sweep

## Status

This document preregisters the exhaustive measured comparison. It contains no
results until the five-placement matrix is validated.

## Question and scope

The study directly compares a measured Placement→DVFS Sequential Oracle with a
measured Joint Oracle over the same five static placements and 81 heterogeneous
frequency tuples. It uses no service/performance model and does not generate a
placement after observing results.

The scope is intentionally limited to four active L40S GPUs, exactly 32
logical experts per EP rank per layer, and TP=1/DP=4/EP=4. Active-GPU count,
replication, migration, and online scheduling are outside this experiment.

## Fixed setup

- Model: `Qwen/Qwen3-30B-A3B`, vLLM 0.15.1.
- Node: one exclusive Uranus allocation; four NVIDIA L40S GPUs.
- All-to-all: `allgather_reducescatter`.
- Workload: the accepted 50-request GSM8K trace at 0.5 requests/s, fixed
  ordering/arrival offsets, temperature 0, seed 0, and 128 output-token limit.
- Frequency levels: H/M/L = 2520/1635/1005 MHz.
- Frequency tuples use one shared seeded permutation (seed `20260831`) for all
  newly measured placements, so tuple position is not coupled to elapsed job
  time while the complete 81-point set is unchanged.
- SLOs: TTFT p99 <= 500 ms and TPOT p99 <= 12, 13, or 15 ms.

## Fixed placement set

| Policy | Mapping |
|---|---|
| P0 | vLLM default linear |
| P1 | accepted upstream EPLB mapping |
| P2 | accepted controlled-medium mapping |
| P3 | accepted controlled-high mapping |
| P4 | accepted controlled-very-high mapping |

P2–P4 are diagnostic progressively skewed mappings. Their expert-load
statistics are recorded only for interpretation, never for point selection.

## Matrix and selection

The measured matrix is 5 placements × 81 tuples = 405 rows. The complete P1
81-point Milestone E matrix is reused only after validating exact placement and
trace hashes, model/vLLM/runtime configuration, request completion, cumulative
NVML energy, and actual clocks. P0 and P2–P4 are each run across all 81 tuples
with one server launch per placement and frequency changes only between energy
windows.

At all-high frequency, the Sequential Oracle selects the placement with minimum
TPOT p99 among rows satisfying the TTFT hard limit. That placement is frozen;
for each TPOT SLO its minimum-energy feasible tuple is selected. The Joint Oracle
selects the minimum-energy feasible row across all 405 rows. Both p99 TTFT and
p99 TPOT are reported for every selected point.

## Required evidence

Every row retains request-level outputs, p99 TTFT/TPOT/E2E, energy and
J/request, throughput, failures, target/actual clocks, trace/config/mapping
hashes, and aggregate/prefill/decode expert-load diagnostics. Raw runs remain
ignored by Git; only compact normalized tables, summary JSON, and this report
are committed after validation.
