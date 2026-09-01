# Handoff

## Latest status — Milestone I formal repeatability matrix prepared

Branch `milestone-i-higher-load-repeatability` was created from completed
Milestone H commit `d7f30c5`. Initial P1/HHHH calibration job `255371`
completed successfully; no P2/P3/P4 higher-load measurement has run yet.

### Prepared and validated

- Added a fixed 300-request GSM8K base trace with seed `20260901` and hash
  `90af9ef2c3bc0c3dd99e4f7cddd4c7dfdabe4223f6278016c98526bbc938bd3d`.
  It contains 300 unique source rows: all 225 available 128–256-token prompts
  plus 75 deterministically selected 120–127-token prompts. Input-token
  mean/p50/p90 is 139.49/135/161.
- The P1/HHHH calibration uses 100 fixed requests at offered rates
  0.5/1.0/1.5/2.0/2.5/3.0/3.5/4.0 req/s. Seed `20260902` fixes their execution
  order independently of results.
- The formal plan generator is fixed to P1–P4, 11 preregistered tuples, three
  repetitions, and recorded placement/run-order seeds. It cannot run until the
  P1-only calibration selects the offered load.
- Added strict run validation, repeat mean/sample-standard-deviation summaries,
  a measured Pareto frontier, a complete 0.1 ms exploratory TPOT-SLO curve,
  and the preregistered A/B/C decision gate. No performance model, scheduler,
  placement heuristic, or expert-load selection is used.
- Python/shell syntax checks and five new pure-function tests pass. `pytest` is
  not installed in the current Minerva environments, so no dependency was
  installed; the tests were imported and executed directly.

### Calibration input

`results/raw/milestone-i-calibration-20260901T082621Z-validated` is the
immutable calibration directory. The older `...082526Z` directory is a
superseded, never-submitted plan retained for provenance.

### Initial calibration result and extension

- Job `255371` completed all 8 rates with 100/100 requests, zero failures,
  positive cumulative NVML energy, target-matching HHHH clocks, final 2520 MHz
  reset, and vLLM patch state `not applied`.
- At 4.0 req/s offered load the measured throughput was 3.675 req/s (91.9%
  efficiency), so the sweep did not bracket saturation under the fixed 90%
  criterion. It is not valid to treat 3.675 req/s as saturation yet.
- Before observing any other placement, an extension at 4.5/5.0/5.5/6.0 req/s
  was fixed with random-order seed `20260905`. Its immutable plan directory is
  `results/raw/milestone-i-calibration-extension-20260901T084416Z-validated`.
- Extension job `255372` completed 4/4 rates with zero failures and safe cleanup.
  Throughput still increased from 4.808 req/s at 5.5 offered to 5.177 req/s at
  6.0 offered, so a stable plateau was not yet established.
- Before any P2/P3/P4 run, the final plateau check was fixed at 7/8/9/10 req/s,
  300 requests per point, order seed `20260906`. Acceptance requires all three
  highest-rate throughputs within 95% of the peak and highest offered rate at
  least 1.2x peak throughput. Immutable plan:
  `results/raw/milestone-i-calibration-plateau-20260901T085110Z-validated`.
- Plateau job `255374` completed safely, but measured throughput still rose from
  6.490 req/s at 7 offered to 8.740 req/s at 10 offered. The fixed plateau rule
  therefore correctly rejected the 10 req/s upper bound.
- A final wider capacity block uses 12/16/20/24 req/s, 300 requests per point,
  and seed `20260907`. It was fixed before other placements and avoids a dense
  additional sweep. Immutable plan:
  `results/raw/milestone-i-calibration-capacity-20260901T085907Z-validated`.
- Capacity job `255375` completed safely, but throughput still rose through
  17.031 req/s at 24 offered. No service model is fit. The final sparse direct
  probe is 32/48/64 req/s, 300 requests each, order seed `20260908`, under:
  `results/raw/milestone-i-calibration-final-plateau-20260901T090539Z-validated`.
- Final probe job `255376` completed safely. Rates >=32 req/s incurred
  0.54–3.69 s client send-delay p99 and are invalid as formal open-loop loads,
  while complete-drain throughput provides a conservative measured capacity
  lower bound of 22.434 req/s. Exact server saturation was not claimed.
- The 65% target is 14.58 req/s. The closest measured client-valid point is
  **16 req/s** (71.3% of the lower bound), with 300/300 completions, TTFT p99
  229.4 ms, and small send delay. This rate is locked before P2/P3/P4 data.
- Formal plan `results/raw/milestone-i-repeatability-20260901T091212Z-r16-validated`
  contains 132 unique runs: P1–P4 x 11 fixed tuples x 3 repetitions. Task block
  order is P3/P1/P2/P4; within-block order is also seeded. Final trace hash is
  `b7cd1171e53fc36bba602ecd800a2a631e3c7505b7eb7d64c3306b0c0be47453`.

### Exact next command

```bash
cd /data/users/chjing/moe-energy-scheduling
MILESTONE_I_STUDY=results/raw/milestone-i-repeatability-20260901T091212Z-r16-validated \
  sbatch --array=0-3%1 scripts/milestone_i_repeat.sbatch
```

## Milestone H — Ground-truth Placement × DVFS sweep complete

The exhaustive measured five-placement by 81-frequency matrix is complete on
branch `milestone-h-ground-truth-placement-dvfs`. The study uses measured
serving results only; no service/performance model or post-hoc placement is
used.

### Execution and validation

- Serialized Slurm array `255356` completed all four new-placement tasks on
  Uranus: P0/P2/P3/P4 each produced 81/81 configurations, 50/50 requests per
  configuration, zero failures, and exit status 0. P1 reuses the accepted
  Milestone E 81-point matrix after strict compatibility checks.
- The final matrix contains 405 unique rows, 81 per placement, one trace hash
  (`721638a6250b929de05c2c355c8bc9b20258db7b28fedb94acb5572136571cbc`),
  cumulative NVML energy, and target-matching actual clocks.
- All controllers reset all four GPUs to 2520 MHz. The final vLLM patch revert
  state is `not applied`.
- P0/P1/P2/P3/P4 aggregate load max/mean is 1.078/1.012/2.038/2.543/2.750.
  These frozen-profile loads are diagnostics only.
- The analyzer's P1 display/runtime-label mismatch was fixed without changing
  the plan, mapping, measurements, or oracle definitions. The complete CPU-safe
  suite passes: `40 passed, 2 warnings`.

### Ground-truth result

- At HHHH, P1 has the lowest measured TPOT p99 (11.183 ms), so the Sequential
  Oracle correctly freezes P1.
- At TPOT <= 12 ms and TTFT <= 500 ms, Sequential and Joint both choose P1 at
  `(1635,2520,2520,2520)`, 63,906.976 J. Saving: 0%.
- At TPOT <= 13 ms, Sequential chooses P1 MMMM at 52,474.447 J; Joint chooses
  P0 MMMM at 52,421.711 J. The strict difference is 52.736 J or 0.1005%.
- At TPOT <= 15 ms, both choose P1 at `(1005,1005,1005,1635)`, 48,661.697 J.
  Saving: 0%.

The measured matrix therefore does **not** establish a meaningful Joint energy
advantage. The only strict gap is 0.1005% at one SLO and is not
repeatability-confirmed; it should be treated as measurement-level evidence,
not a paper claim.

### Artifacts and next step

- Report: `docs/milestone_h_ground_truth.md`
- Full table: `results/processed/milestone_h_ground_truth_matrix.csv`
- Oracle table: `results/processed/milestone_h_oracle_summary.csv`
- Machine-readable summary: `results/processed/milestone_h_summary.json`
- Raw study: `results/raw/milestone-h-plan-20260831T100215Z-seeded`

Do not build a larger Joint optimizer from this result. If the placement-DVFS
hypothesis is retained, first repeat the selected P0/P1 MMMM points and the
closest HHHH placement competitors at least three times.

### Exact next command

```bash
cd /data/users/chjing/moe-energy-scheduling
cat docs/milestone_h_ground_truth.md
cat results/processed/milestone_h_summary.json
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
