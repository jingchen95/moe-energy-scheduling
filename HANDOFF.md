# Handoff

## Current working state

Milestone D is complete. The deterministic serving harness, frozen expert-load
profile, arbitrary static placement, uniform DVFS curves, single-GPU
perturbations, repeatability trials, processed tables, and diagnostic figure are
available. H1 is supported for the one-workload pilot: placement materially
changed per-GPU DVFS sensitivity.

The project is staged on Minerva at:

```text
/data/users/chjing/moe-energy-scheduling
```

The active branch is `milestone-d-placement-dvfs-coupling`. The vLLM wheel is
restored to its unpatched state and GPU graphics-clock locks were reset.

## Completed steps

- Milestone C closeout commit: `ef276ad`.
- Milestone D infrastructure commits: `bdbc10a`, `d5ee90f`, `b8e98f7`,
  `423308e`, `c0c6259`, and `474652a`.
- Live OpenAI streaming validation passed with 15/15 requests, plausible
  TTFT/TPOT, positive four-GPU energy, and about 2 ms p99 send delay.
- Recorded and exercised `patches/vllm-0.15.1-placement-profile.patch`, including
  apply/verify/revert hashes and `--expert-placement-file`.
- Captured one frozen default-placement GSM8K expert profile and generated P0
  linear, P1 upstream `DefaultEplbPolicy`, and P2 diagnostic concentrated maps.
- Used the fixed 50-request GSM8K trace (128-191 input tokens, 128 output tokens,
  0.5 requests/s) for every measured configuration.
- Selected supported graphics clocks 2520, 1635, and 1005 MHz. Actual average
  clocks matched targets in all runs.
- Slurm job `255131` completed the P1/P2 campaign with exit code 0 on an
  exclusive Uranus allocation. It resumes the immutable profile and P0 curve
  from job `255129`; `resumed-from.txt` records this provenance.
- All 29 manifest rows have zero request failures, use cumulative NVML energy,
  and have at most 2.61 ms p99 send delay.
- The complete CPU-safe suite passes: `20 passed in 0.28s` with
  `PYTHONPATH=.deps python -m pytest -q` in `cuda-env`.
- Generated `results/processed/milestone_d_uniform.csv`,
  `results/processed/milestone_d_sensitivity.csv`,
  `results/processed/milestone_d_summary.json`, and
  `results/figures/milestone_d_dvfs_sensitivity.pdf`.

## Central result

P1 predicted rank loads are essentially equal (max/mean 1.00002). P2 assigns
15.39M predicted routed tokens to GPU0 versus 0.47M-3.48M to the other GPUs
(max/mean 2.933).

Downclocking GPU0 from 2520 to 1635 MHz increases p99 TPOT by 0.942 ms under P1
and 1.719 ms under P2, an 82% larger penalty. Under P2, the hot GPU's penalty is
about twice the median cool-GPU penalty. Three-trial means reproduce the GPU0
effect: +0.960 ms for P1 and +1.657 ms for P2. H1 is therefore supported for
this pilot.

The uniform curves do not provide positive H2 evidence: at relative p99 TPOT
budgets of 1.05x, 1.10x, and 1.20x, P0/P1/P2 all select high, high, and medium,
respectively. No Joint optimizer was implemented.

## Slurm jobs and retained failures

- `255099`: partial P0 data; failed on the fourth collective profile dump.
- `255128`: live run succeeded; orchestration failed because the validator
  still required profile output in the streaming-validation phase.
- `255129`: valid frozen profile, placements, P0 curve, and P1 semantic run;
  stopped at the original exact-token semantic gate.
- `255130`: retained failed retry; collective profile dump timed out.
- `255131`: successful resumed P1/P2 campaign, 50:50 elapsed, exit code 0.

No failed or partial run is included in the processed result tables.

## Unresolved issues

- vLLM 0.15.1 collective expert-profile dumping is intermittently deadlocking.
  Reuse the accepted frozen profile until this RPC is debugged separately.
- Custom placement preserves at least an eight-token required prefix and an
  observed minimum 15-token prefix, but later greedy tokens can diverge because
  expert reordering changes BF16 kernel/reduction order. All divergence details
  are retained in the controller validation JSON files.
- TTFT p99 has non-monotonic outliers at 0.5 requests/s. Use TPOT as the primary
  coupling signal and strengthen TTFT repetition in later SLO studies.
- The pilot covers only GSM8K and one intentionally strong P2 intervention.

## Recommended next step

H1 is clearly supported, so proceed to:

> Milestone E - test sequential frozen EPLB plus heterogeneous DVFS versus
> frequency-aware Joint placement under common TTFT/TPOT SLOs.

Do not discard the current negative H2 observation, and retain P1 as the same
frozen EPLB placement for all sequential baselines.

## Exact next command

```bash
git switch -c milestone-e-sequential-vs-joint
```
