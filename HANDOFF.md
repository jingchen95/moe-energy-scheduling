# Handoff

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
