# Handoff

## Latest status — Ground-truth Placement × DVFS sweep (resource-canceled)

The new exhaustive ground-truth sweep is prepared on branch
`milestone-h-ground-truth-placement-dvfs`, created from the completed
Milestone G commit. The first pending array was canceled before any task
started after correcting its tuple ordering. Replacement array `255352` was
then canceled by Slurm before launch because Uranus resources were unavailable;
the seeded plan below remains valid and no new measurement data was produced.

### Preparation completed

- Added `configs/milestone_h_ground_truth.yaml`,
  `tools/prepare_milestone_h.py`, `tools/analyze_milestone_h.py`,
  `scripts/milestone_h_ground_truth.sbatch`, and CPU-safe oracle tests.
- The fixed matrix is P0 default linear, P1 accepted upstream EPLB, and the
  accepted P2/P3/P4 controlled-medium/high/very-high mappings, each crossed
  with all 81 H/M/L frequency tuples.
- P1 reuse is gated on exact trace/mapping hashes, vLLM/model/runtime settings,
  50/50 completion, zero failures, NVML cumulative energy, and actual clocks.
- Selection uses measured p99 TPOT as the Sequential high-frequency placement
  criterion, a hard p99 TTFT <= 500 ms limit, and measured total energy only.
  No service model, load proxy, or post-hoc placement generation is used.
- Frequency tuples for all newly measured placements use one shared seeded
  permutation (`frequency_order_seed=20260831`). The analyzer accepts only an
  attempt with `result.json` status `complete` and 81 unique completed tuples;
  partial/failed retries remain on disk but are ignored.

### Canceled before launch

- Plan `results/raw/milestone-h-plan-20260831T100215Z-seeded` is generated and
  passed
  inspection: 81 P1 rows are reusable, the matrix is 5 x 81 = 405 rows, and
  the trace hash is `721638a6250b929de05c2c355c8bc9b20258db7b28fedb94acb5572136571cbc`.
- The P0/P1/P2/P3/P4 aggregate max/mean diagnostics are 1.078/1.012/2.038/
  2.543/2.750 respectively. These values are diagnostics only.
- Reliability-fix commit `2d45fdb` is pushed to GitHub. The old pending array
  `255351` was canceled before execution. Replacement array `255352` was
  recorded by Slurm as `CANCELLED` with reason `Resources` at
  `2026-08-31T13:53:14` without a node allocation or array-task records. At
  that time Uranus was occupied by job `255348`; no H sweep task ran and no
  measured artifacts were created. P1's 81 rows remain reusable, so no P1 GPU
  rerun is planned.
- Analyze the validated 405-row matrix and update this handoff with the
  Sequential-vs-Joint Oracle result.

### Exact next command

```bash
cd /data/users/chjing/moe-energy-scheduling
source /data/users/chjing/miniforge3/bin/activate cuda-env
python tools/prepare_milestone_h.py \
  --trace results/raw/milestone-e-sequential-255132-20260826T140049Z/trace.jsonl \
  --phase-counts results/raw/milestone-f-phase-255174-20260826T214727Z/runs/c-math-20260826T214834.669632Z-0248d849/expert_counts.npz \
  --p1-controller results/raw/milestone-e-sequential-255132-20260826T140049Z \
  --p0 results/raw/milestone-d-255131-20260826T104006Z/placements/default_linear.json \
  --p1 results/raw/milestone-d-255131-20260826T104006Z/placements/upstream_default_eplb.json \
  --p2 results/raw/milestone-g-plan-20260829T085033Z-retry/placements/controlled_medium.json \
  --p3 results/raw/milestone-g-plan-20260829T085033Z-retry/placements/controlled_high.json \
  --p4 results/raw/milestone-g-plan-20260829T085033Z-retry/placements/controlled_very_high.json \
  --output-dir results/raw/milestone-h-plan-20260831T100215Z-seeded
```

The generated `matrix-plan.json` and `p1-reuse-summary.json` passed inspection.

Submit the corrected array with:

```bash
 squeue -w uranus
 MILESTONE_H_STUDY=results/raw/milestone-h-plan-20260831T100215Z-seeded \
   sbatch --array=0-3%1 scripts/milestone_h_ground_truth.sbatch
```

## Milestone G (completed)

Milestone G controlled load–frequency characterization is complete on branch
`milestone-g-load-frequency-model`. The 24 accepted measurements (12 initial
matrix points plus 12 preregistered repeats) were collected on one Uranus node
with Qwen/Qwen3-30B-A3B, vLLM 0.15.1, TP=1/DP=4/EP=4, four NVIDIA L40S GPUs,
and `allgather_reducescatter`.

### Completed

- Controlled static placements were generated from the frozen GSM8K expert
  profile. GPU0 decode loads are LOW `612,845`, MEDIUM `1,234,496`, HIGH
  `1,576,914`, and VERY_HIGH `1,725,728` routed tokens; every placement has
  128 experts exactly once and 32 experts per EP rank.
- The 12-point matrix used GPU0 at H/M/L = 2520/1635/1005 MHz with GPUs 1–3
  fixed at 2520 MHz. Repeats covered LOW-H/M, HIGH-H/M, and VERY_HIGH-H/M.
- Job array `255315` completed all four serialized controllers. Every run had
  50/50 completed requests, zero failures, one trace hash, positive NVML
  cumulative energy, exact target/actual average clocks, and clock reset.
- Added opt-in rank-local CUDA-event timing around the FusedMoE dispatch,
  expert compute, and combine region in
  `patches/vllm-0.15.1-moe-timing.patch`. Events are resolved only at profile
  dump; no collective or timing RPC runs in the measured forward.
- vLLM 0.15.1 cleared phase metadata before the wrapper retained it. The
  postprocessor uses an explicit fallback: 1–2 token rows are decode and
  >=129 rows are prefill. This limitation is documented and test-covered.
- Final compact tables are in `results/processed/`, figures in
  `results/figures/`, and the report is
  `docs/milestone_g_load_frequency_model.md`.
- The complete Minerva CPU-safe suite passes: `36 passed, 2 warnings`.

### Findings and decision gate

- TPOT p99 downclock penalties relative to each placement's all-high run were
  not monotonic: Pearson correlation with GPU0 decode load was `-0.178` over
  eight paired transitions. Normalized MoE service penalties were also
  non-monotonic (Pearson `-0.606`).
- Model C is the interpretable max-GPU load/frequency model
  `a + b*max(L*fmax/f) + c*max(L*(fmax/f - 1))`. Its TPOT leave-one-out MAE
  was `17.36 ms`, MAPE `16.21%`, maximum error `43.02 ms`, and rank-order
  accuracy `0.621`. Its MoE per-row proxy leave-one-out MAE was `0.171 ms`,
  maximum error `0.390 ms`, and rank-order accuracy `0.515`.
- On the existing eight Milestone E Joint candidates, offline Model C
  rescoring had MAE `108.18 ms`, maximum error `113.95 ms`, and rank-order
  accuracy `0.036`, versus Milestone F MAE `2.03 ms` and Milestone E MAE
  `4.66 ms`. This is an out-of-domain diagnostic, not a new GPU experiment.
- Classification: **`G1_NOT_SUPPORTED`**. The controlled load axis did not
  show reliable load-dependent DVFS sensitivity beyond run-to-run variation;
  the TPOT model is not accurate enough to justify another placement/DVFS
  search.

### Unresolved issues

- The timing patch's phase fallback is explicit but not a retained vLLM phase
  tag; a future instrumentation pass could expose phase metadata directly.
- No direct all-to-all, synchronization, grouped-GEMM, or wait-time timer is
  available, and custom placement can change BF16 reduction order/output
  identity. No new claim about semantic expert skew is made.
- Results cover one model, one GSM8K trace, one node, three clocks, no
  replication, and no migration. Raw runs remain on Minerva under
  `results/raw/milestone-g-plan-20260829T085033Z-retry` and are ignored by Git.

## Earlier milestones

- Milestone D established the placement perturbation and static mapping path.
- Milestone E measured the exhaustive Sequential B4 sweep and eight Joint
  candidates; its best 13 ms point was `52,450.538 J` versus the Sequential
  B4 optimum `52,474.447 J` (0.0456% lower, not repeatability-confirmed).
- Milestone F diagnosed the frequency-dominant component and added the phase
  profile used as the frozen G input.

## Milestone G exact next command (superseded)

Do not launch another GPU job and do not implement Joint v2. Review the final
negative decision gate and compact evidence:

```bash
cd /data/users/chjing/moe-energy-scheduling
cat docs/milestone_g_load_frequency_model.md
cat results/processed/milestone_g_summary.json
```

The recommended next research decision is to reconsider whether placement
belongs in the DVFS objective. If the thesis is retained, first add direct
communication/synchronization timing or repeat the coupling experiment with a
stronger controlled intervention; do not start Joint v2 from this result.

## Archived milestone-E notes (superseded by the G status above)

The following execution details are retained for provenance only. The G status
and exact next command above are authoritative.

Milestone D remains complete at commit `d233672`. Milestone E experiment code,
CPU planning/analysis tools, and Slurm jobs are prepared on branch
`milestone-e-sequential-vs-joint`. The exhaustive Sequential Stage A job
`255132` was submitted on 2026-08-26 and is running exclusively on Uranus from
code commit `da7c5cd`; it completed successfully. The CPU-only Joint plan was
then generated and passed its H/M/M/L sanity gate. Joint array `255135` then
measured all eight candidates successfully; the later G study superseded the
pending E follow-up for the current research decision.

The active controller is:

```text
results/raw/milestone-e-sequential-255132-20260826T140049Z
```

The accepted Milestone D source on Minerva is:

```text
/data/users/chjing/moe-energy-scheduling/results/raw/milestone-d-255131-20260826T104006Z
```

The pre-submission CPU-safe suite passed on Minerva: `28 passed in 3.09s`.

## Prepared components

- `configs/milestone_e.yaml` preregisters the trace, SLOs, clock levels, search
  budget, and immutable-profile policy.
- `src/placement.py` provides deterministic per-layer frequency-aware LPT
  placement with exact 32-expert/rank cardinality and bounded improving swaps.
- `src/search.py` provides 81-tuple enumeration, absolute-SLO filtering,
  energy selection, Pareto extraction, deterministic Joint candidates, paired
  comparisons, and conservative repeatability summaries.
- `tools/prepare_milestone_e.py` creates the seeded Sequential plan, refuses
  Joint planning without 81 valid unique points, generates 5-8 placements, and
  creates the 13 ms repeat plan.
- `tools/analyze_milestone_e.py` is ready to create the required CSV/JSON tables
  and two PDF figures from real data. It applies a conservative preregistered
  classification rule and never fabricates missing outputs.
- `scripts/milestone_e_sequential.sbatch` runs all 81 P1 tuples using one server
  and records failures instead of replacing them.
- `scripts/milestone_e_joint.sbatch` and `scripts/milestone_e_repeat.sbatch`
  are serialized Slurm-array workers. No script submits another job.
- All jobs retain immutable hashes, actual clocks, request tokens/latencies,
  cumulative NVML energy, monitor samples, output-token IDs, patch state, and
  unique raw run directories. They never call the expert-profile RPC.

## Historical verification completed

- Python source compilation passes with bytecode redirected to `/tmp`.
- `bash -n` passes for all Milestone E shell/Slurm scripts.
- `git diff --check` passes.
- A direct standard-library check confirms 81 unique frequency tuples and
  deterministic eight-candidate selection.
- Sequential job `255132`: `COMPLETED`, exit `0`, elapsed `02:51:15` on Uranus.
- All 81 tuples are unique and complete: 50/50 requests, zero failures, 6,400
  output tokens, one trace hash, NVML cumulative-energy source, and maximum p99
  send delay 4.087 ms. Actual average clocks matched every target exactly.
- Absolute-SLO B4 best points: 12 ms = `(1635,2520,2520,2520)`, 63,906.976 J;
  13 ms = `(1635,1635,1635,1635)`, 52,474.447 J; 15 ms =
  `(1005,1005,1005,1635)`, 48,661.697 J. Every point passed the 500 ms TTFT
  guard; feasible tuple counts were 4, 16, and 77.
- Joint plan `results/raw/milestone-e-joint-plan-255132-20260826T140049Z`
  contains eight deterministic candidates. H/M/M/L normalized pressure is
  1.015 for frequency-aware placement versus 1.522 for P1 under the fixed
  `s(f)=f/2520` proxy, with raw load shifted toward the high-frequency rank.
- Joint array `255135`: all eight tasks completed on Uranus (about 3:42-3:57
  each), with 50/50 requests, zero failures, one trace hash, positive NVML
  energy, exact target-clock readback, and `PASS` live validation. Every task
  reverted the vLLM patch and reset graphics clocks.
- No candidate made a Sequential-infeasible tuple feasible at the 12, 13, or
  15 ms TPOT SLOs. The best 13 ms Joint point is candidate 3, `(M,M,M,M)`,
  at 52,450.538 J and p99 TPOT 12.663 ms versus B4's 52,474.447 J and
  12.471 ms: 23.909 J or 0.0456% lower energy. This is not yet a robust claim;
  the required three-trial repeatability stage has not run.
- Joint output-token exact-match fractions were 0.14-0.28, with minimum common
  prefixes of 4-13 tokens and median prefixes of 51.0-68.5 tokens; this known
  custom-placement limitation is retained in each candidate controller.

The full suite could not run on this Mac because its system Python has neither
NumPy nor pytest; dependencies were deliberately not installed. The established
Minerva environment was used before submission:

```bash
PYTHONPATH=.deps python -m pytest -q
```

Result: `28 passed in 3.09s`.

## Historical staged execution commands

Stage A, exhaustive Sequential B4, was submitted as job `255132` with:

```bash
MILESTONE_D_CONTROLLER=results/raw/milestone-d-255131-20260826T104006Z \
  sbatch scripts/milestone_e_sequential.sbatch
```

After all 81 points succeed, prepare Joint candidates on Minerva (CPU only):

```bash
seq_controller=$(cat results/raw/milestone-e-sequential-latest.txt)
profile_run=$(awk -F '\t' '$1 == "p0-frozen-profile" {print $4; exit}' \
  results/raw/milestone-d-255131-20260826T104006Z/runs.tsv)
joint_study="results/raw/milestone-e-joint-plan-$(date -u +%Y%m%dT%H%M%SZ)"
python -m tools.prepare_milestone_e joint-plan \
  --sequential-controller "${seq_controller}" \
  --expert-counts "${profile_run}/expert_counts.npz" \
  --p1-placement results/raw/milestone-d-255131-20260826T104006Z/placements/upstream_default_eplb.json \
  --output-dir "${joint_study}"
```

Then inspect `sanity-hmml.json` and `joint-plan.json`. Only after that check,
submit the exact reported candidate count as a serialized array:

```bash
candidate_count=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["candidate_count"])' \
  "${joint_study}/joint-plan.json")
MILESTONE_E_JOINT_STUDY="${joint_study}" \
  sbatch --array="0-$((candidate_count - 1))%1" scripts/milestone_e_joint.sbatch
```

Repeat planning and final analysis are intentionally gated on those measured
Joint results. Once every planned Joint candidate has a successful result:

```bash
repeat_study="results/raw/milestone-e-repeat-plan-$(date -u +%Y%m%dT%H%M%SZ)"
python -m tools.prepare_milestone_e repeat-plan \
  --sequential-controller "${seq_controller}" \
  --joint-study "${joint_study}" \
  --output-dir "${repeat_study}"
configuration_count=$(python -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["configurations"]))' \
  "${repeat_study}/repeat-plan.json")
MILESTONE_E_REPEAT_STUDY="${repeat_study}" \
  sbatch --array="0-$((configuration_count - 1))%1" scripts/milestone_e_repeat.sbatch
```

After all repeat configurations complete:

```bash
python -m tools.analyze_milestone_e \
  --milestone-d-controller results/raw/milestone-d-255131-20260826T104006Z \
  --sequential-controller "${seq_controller}" \
  --joint-study "${joint_study}" \
  --repeat-study "${repeat_study}"
```

## Historical Milestone-E unresolved issues

- Three-trial repeatability has not been submitted.
- Joint output divergence remains a known interpretation limitation; the new
  jobs retain exact-match fraction and minimum/median common-prefix lengths.
- The processed Milestone E CSV/JSON tables and figures require the Joint and
  repeatability measurements; no scientific classification is justified yet.

## Archived Milestone-E next command (superseded)

Inspect the generated Joint sanity file without submitting another GPU job:

```bash
cd /data/users/chjing/moe-energy-scheduling
cat results/raw/milestone-e-joint-plan-255132-20260826T140049Z/sanity-hmml.json
```

After review, the next GPU action is the repeat-plan command above; it must not
be submitted until the user confirms.
