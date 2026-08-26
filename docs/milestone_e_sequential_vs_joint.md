# Milestone E: Sequential EPLB + DVFS vs Joint Placement + DVFS

## Status

The implementation and Slurm jobs are prepared. The 81-point Sequential B4
sweep has completed successfully on Uranus, and the deterministic Joint plan
has passed its CPU-only sanity check. All eight Joint candidates have now
completed successfully; repeatability and final classification remain pending.

## Research question

Can frequency-aware expert placement beat frozen EPLB plus exhaustive
heterogeneous DVFS under a common absolute SLO?

## Prior evidence

Milestone D supported placement-DVFS coupling on Qwen3-30B-A3B. Downclocking
GPU0 from 2520 to 1635 MHz increased p99 TPOT by 0.942 ms with balanced P1 and
1.719 ms with diagnostic concentrated P2. Milestone E does not rerun P2 or
re-establish activation skew.

## Preregistered method

- Hardware/model: one Uranus node, 4 NVIDIA L40S GPUs,
  Qwen/Qwen3-30B-A3B, vLLM 0.15.1, TP=1/DP=4/EP=4.
- Communication: `allgather_reducescatter`; no replication or migration.
- Immutable workload: the accepted Milestone D 50-request GSM8K trace at
  0.5 requests/s, 128-191 input tokens and exactly 128 output tokens.
- Immutable profile: the accepted Milestone D `p0-frozen-profile` counts. The
  collective profiling RPC is never invoked by Milestone E.
- Clocks: H=2520, M=1635 and L=1005 MHz; memory clocks are untouched.
- Common SLOs: p99 TTFT <= 500 ms and p99 TPOT <= 12, 13, or 15 ms. These are
  pilot thresholds, not production SLOs.
- Sequential B4: all 81 per-GPU tuples with the same frozen P1 upstream EPLB
  placement, in a seeded permutation, using one server launch.
- Joint: 5-8 tuples selected deterministically from the completed Sequential
  frontier. Every Joint measurement is paired against P1 at the identical
  tuple.
- Repeatability: best B4 and best measured Joint at the 13 ms SLO receive three
  new trials. If the Joint tuple differs, P1 at that tuple is also repeated.

## Frequency-aware placement

For each layer, experts are sorted by descending frozen count and assigned by a
capacity-normalized constrained LPT heuristic. Every rank retains exactly 32 of
the 128 logical experts. The primary pre-registered speed proxy is
`s(f) = f / 2520`; it is a placement heuristic, not an assertion that model
throughput scales linearly with clock frequency. Deterministic pairwise swaps
are accepted only when they reduce the maximum per-layer normalized pressure.

The CPU planning stage records a mandatory H/M/M/L sanity comparison between
P1 and the generated mapping before any Joint candidate can be submitted.

## Sequential checkpoint

All 81 tuples completed with zero request failures, one immutable trace hash,
NVML cumulative energy, and maximum p99 send delay of 4.087 ms. The best
Sequential B4 points were 63,906.976 J at `(M,H,H,H)` for the 12 ms SLO,
52,474.447 J at `(M,M,M,M)` for 13 ms, and 48,661.697 J at `(L,L,L,M)` for
15 ms. The corresponding feasible tuple counts were 4, 16, and 77; all passed
the 500 ms TTFT guard.

The generated eight-candidate plan uses the measured frontier. For H/M/M/L,
the frequency-aware placement has normalized-load max/mean 1.015 versus 1.522
for P1 under the registered `s(f)=f/2520` proxy.

## Joint candidate checkpoint

All eight candidates passed live validation with 50/50 requests, zero failures,
one trace hash, cumulative NVML energy, exact target clocks, and cleanup. The
candidate-level result is:

| Candidate | Frequencies (MHz) | Joint raw loads (M tokens) | Norm max/mean | TTFT p99 (ms) | TPOT p99 (ms) | Energy (J) | J/request | SLO 12/13/15 | Sequential TPOT / J |
|---:|---|---|---:|---:|---:|---:|---:|---|---:|
| 0 | M,H,H,H | 3.746,5.774,5.733,5.733 | 1.004 | 95.169 | 12.204 | 64101.444 | 1282.029 | N/Y/Y | 11.951 / 63906.976 |
| 1 | M,H,M,H | 4.181,6.348,4.177,6.279 | 1.011 | 95.131 | 12.292 | 60214.108 | 1204.282 | N/Y/Y | 12.171 / 59982.698 |
| 2 | H,M,M,H | 6.408,4.157,4.141,6.278 | 1.006 | 96.711 | 12.314 | 60447.844 | 1208.957 | N/Y/Y | 12.091 / 60694.421 |
| 3 | M,M,M,M | 5.246,5.246,5.246,5.246 | 1.000 | 96.990 | 12.663 | 52450.538 | 1049.011 | N/Y/Y | 12.471 / 52474.447 |
| 4 | M,M,L,M | 5.825,5.811,3.567,5.782 | 1.003 | 312.394 | 14.366 | 50111.227 | 1002.225 | N/N/Y | 14.134 / 50083.909 |
| 5 | H,H,L,H | 6.252,6.190,2.422,6.120 | 1.015 | 319.827 | 13.562 | 62746.389 | 1254.928 | N/N/Y | 13.728 / 62361.796 |
| 6 | L,L,L,M | 4.622,4.623,4.624,7.116 | 1.014 | 172.531 | 14.765 | 48646.724 | 972.934 | N/N/Y | 14.683 / 48661.697 |
| 7 | L,L,L,L | 5.246,5.246,5.246,5.246 | 1.000 | 300.393 | 15.211 | 47946.757 | 958.935 | N/N/N | 15.248 / 47913.387 |

No Sequential-infeasible tuple became Joint-feasible at any registered SLO.
At TPOT <= 13 ms and TTFT <= 500 ms, candidate 3 is the best measured Joint
configuration: 52,450.538 J versus exhaustive B4's 52,474.447 J, a 0.0456%
reduction. This single-run difference is not a robust result until the planned
three-trial repeatability comparison is complete. Output exact-match fractions
were 0.14-0.28; minimum common prefixes were 4-13 tokens and medians were
51.0-68.5 tokens.

## Staged execution

No script in this milestone calls nested `sbatch`. The campaign has four
explicit gates:

1. Run the exhaustive Sequential job. Failed tuples are retained and block the
   Joint planner until all 81 unique tuples succeed.
2. Run the CPU-only `joint-plan` command against that controller. It records all
   candidate choices before any Joint outcome is observed.
3. Submit the serialized Joint array using the exact array bound reported by
   the plan.
4. Generate and submit the repeatability plan, then run the analyzer.

The exact commands are recorded in `HANDOFF.md`. The Joint GPU array remains
held until the user confirms the Sequential checkpoint and candidate plan.

## Planned outputs

After real data exist, `tools/analyze_milestone_e.py` creates:

- `results/processed/milestone_e_sequential_all.csv`
- `results/processed/milestone_e_joint_candidates.csv`
- `results/processed/milestone_e_policy_summary.csv`
- `results/processed/milestone_e_repeatability.csv`
- `results/processed/milestone_e_summary.json`
- `results/figures/milestone_e_main.pdf`
- `results/figures/milestone_e_pareto.pdf`

The analyzer applies the registered repeatability rule and emits a conservative
classification. An incomplete three-trial comparison or any selected-policy
SLO failure is `INCONCLUSIVE`; the generated evidence must still be reviewed
before updating this document with a scientific conclusion.

## Current limitations

The repeatability result is not available yet. The eventual pilot will still cover one model, one
workload, one low request rate, one L40S node, only three frequency levels, no
replication/migration, a linear speed proxy, and a bounded placement search.
Greedy outputs may diverge after an initially common prefix because custom
placement changes BF16 reduction order. Exact-match fraction plus minimum and
median common-prefix length are retained for paired and repeated Joint runs.
