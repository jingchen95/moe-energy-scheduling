# Handoff

## Latest status — Milestone F

Milestone F is complete on branch `milestone-f-service-model` through the
offline diagnosis and one accepted no-collective phase-profile run. The
Milestone E measurements were reused; no repeatability sweep, new Joint search,
or online scheduler was started.

### Completed

- Added `src/service_model.py` with Amdahl and small max-GPU service-time fits,
  phase-count validation/aggregation, candidate rescoring, correlations, and
  load-dependent P1/P2 perturbation analysis in `tools/analyze_milestone_f.py`.
- Added the opt-in vLLM 0.15.1 phase instrumentation patch stack. FlashAttention
  query lengths classify one-token rows as decode and each DP rank marks only
  its own contiguous token block; no collective runs in the measured forward.
- Final validation job `255174` completed with 50/50 requests, zero failures,
  positive NVML energy, exact prefill+decode conservation, and successful patch
  and clock cleanup. Profile run:
  `results/raw/milestone-f-phase-255174-20260826T214727Z/runs/c-math-20260826T214834.669632Z-0248d849`.
- Phase totals: 20,365,056 routed assignments; 17,926,656 prefill (88.03%) and
  2,438,400 decode (11.97%). The earlier 255149 profile that used a temporary
  cross-rank count was discarded from analysis.
- Generated `docs/milestone_f_service_model.md` and the three processed tables
  plus `milestone_f_summary.json` under `results/processed/`.

### Diagnosis

- Uniform P1 measurements at 2520/1635/1005 MHz fit
  `T = T_fixed + T_freq * fmax/f` with fixed 8.394 ms, frequency-sensitive
  2.717 ms, sensitive fraction 0.245, and fit MAE 0.074 ms. The old
  `TPOT(high)*fmax/f` proxy has 5.853 ms MAE.
- Decode-only candidate pressure did not provide a clearly better positive
  correlation with measured TPOT (Pearson -0.493, Spearman -0.429; n=8) than
  aggregate pressure (Pearson -0.414, Spearman -0.333). The empirical service
  model reduces aggregate candidate MAE from 4.660 to 2.026 ms, but its
  rank-order accuracy is lower (0.821 vs 0.893), so it does not explain all
  placement residuals.
- Existing D perturbations show higher TPOT penalty for the concentrated high
  load GPU0 (1.719 ms) than the balanced P1 GPU0 (0.942 ms), but the eight-point
  diagnostic is not a causal fit. Classification: `B_FREQUENCY_MODEL_DOMINANT`.

### Validation and unresolved issues

- CPU-safe tests on Minerva remain `32 passed` before the final analyzer-only
  edits; local Mac Python lacks NumPy/pytest. The analyzer and patch files pass
  syntax/diff checks locally and the analyzer ran successfully on Minerva.
- Profile runs `255145`/`255147`/`255148` were failed or superseded debugging
  attempts; only `255174` is accepted for phase analysis. CUDA graphs require
  `--enforce-eager` for this opt-in instrumentation.
- Communication, synchronization, and grouped-GEMM effects are not measured;
  output divergence between custom placements remains a Milestone E limitation.

### Exact next command

Do not launch another GPU job yet. Review the diagnosis with:

```bash
cd /data/users/chjing/moe-energy-scheduling
cat docs/milestone_f_service_model.md
```

If proceeding, the recommended next milestone is a small offline Joint pilot
using decode-only profiles and the empirical `T(L,f)` model before adding any
communication-aware objective. Do not rerun the 81-point sweep or the E
repeatability stage.

## Current working state

Milestone D remains complete at commit `d233672`. Milestone E experiment code,
CPU planning/analysis tools, and Slurm jobs are prepared on branch
`milestone-e-sequential-vs-joint`. The exhaustive Sequential Stage A job
`255132` was submitted on 2026-08-26 and is running exclusively on Uranus from
code commit `da7c5cd`; it completed successfully. The CPU-only Joint plan was
then generated and passed its H/M/M/L sanity gate. Joint array `255135` then
measured all eight candidates successfully; repeatability and final
classification remain pending.

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

## Verification completed

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

## Staged execution commands

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

## Unresolved issues

- Three-trial repeatability has not been submitted.
- Joint output divergence remains a known interpretation limitation; the new
  jobs retain exact-match fraction and minimum/median common-prefix lengths.
- The processed Milestone E CSV/JSON tables and figures require the Joint and
  repeatability measurements; no scientific classification is justified yet.

## Exact next command

Inspect the generated Joint sanity file without submitting another GPU job:

```bash
cd /data/users/chjing/moe-energy-scheduling
cat results/raw/milestone-e-joint-plan-255132-20260826T140049Z/sanity-hmml.json
```

After review, the next GPU action is the repeat-plan command above; it must not
be submitted until the user confirms.
