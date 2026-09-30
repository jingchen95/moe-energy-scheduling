# Handoff

## Latest status — Milestone M isolated AFD runtime prepared

Branch `milestone-m-afd-fixed-frequency-placement` begins a staged test of
individual expert placement on fixed heterogeneous FFN frequencies after
Attention--FFN disaggregation.

- The protected `cuda-env` baseline is vLLM 0.15.1 and PyTorch 2.9.1+cu128.
  Its interpreter and package roots are under
  `/data/users/chjing/miniforge3/envs/cuda-env`; it was not modified.
- The official `vllm-project/afd-plugin` checkout is isolated under the
  ignored `third_party/afd-plugin` directory and pinned at commit
  `4c2541a53ec8ad507508073eac81ac9feecc6c9a`.
- A project-local `uv 0.12.20` bootstrap created the plugin-local `.venv` with
  vLLM 0.26.0, PyTorch 2.11.0, and plugin version
  `0.26.0rc2.dev61+g4c2541a53`.
- `tools/check_milestone_m_environments.py` asserts both vLLM versions,
  distinct interpreter/package roots, plugin installation, and the pinned
  clean plugin commit.
- Source inspection found generic plugin support for a `2A2F` scenario, while
  the Qwen3 MoE E2E module currently covers only `2A1F`. Qwen3 2A2F remains an
  explicit real-GPU smoke-test prerequisite; no placement or primary job has
  been submitted.
- The repository CPU suite passes `68` tests in the protected runtime. The
  pinned plugin CPU unit suite also exits 0 when GPU/E2E/slow/NPU tests are
  excluded.
- The unmodified-plugin `2A2F` eager smoke launcher is prepared. It uses owned
  Attention/FFN process groups, sends three fixed prompts twice, checks exact
  deterministic outputs, confirms all four GPUs are used, and resets clocks.
  The current project suite passes `72` tests.
- Exclusive Uranus smoke job `269841` was submitted from commit `5c0d612` and
  failed before service readiness. Triton's first CUDA helper JIT could not
  find `/usr/include/python3.12/Python.h` on Uranus. This is an isolated-runtime
  packaging failure, not a 2A2F or research result. Cleanup reset clocks, left
  no GPU compute processes, and preserved a clean plugin checkout and the
  vLLM 0.15.1 protected environment.
- The AFD `.venv` has been rebuilt with fixed uv-managed CPython 3.12.3 and its
  headers under the ignored project-local `.tools/uv-python` tree. The
  environment check now fails if `Python.h` is unavailable. No custom
  placement, profiling job, or full experiment has started before the smoke
  retry passes.
- The corrected unmodified-plugin smoke ran as exclusive Uranus job `269912`
  and failed during Attention initialization. The Python-header fix worked,
  but FlashInfer's sampler JIT required an unavailable `nvcc`; the later FFN
  NCCL receive timeout was secondary. Cleanup reset clocks and left no GPU
  compute processes.
- The launcher now sets vLLM 0.26's documented
  `VLLM_USE_FLASHINFER_SAMPLER=0` process-local switch. This selects the native
  sampler without installing CUDA or editing the pinned runtime. Greedy
  decoding parameters are unchanged. The fix is commit `e298db5`.
- The next unmodified-plugin smoke retry was submitted as exclusive Uranus job
  `269940`. It confirmed that both roles initialize with the native sampler:
  the Attention API returned HTTP 200 and FFN entered its synchronous connector
  loop. The launcher then waited incorrectly for an FFN HTTP API that the
  plugin does not provide, sent no requests, and eventually hit the FFN NCCL
  receive timeout. Cleanup reset clocks and left no GPU compute processes.
- Readiness now matches the pinned plugin's official runner: wait for Attention
  `/v1/models`, require both role processes to remain alive, and record the FFN
  connector-loop startup line. The fix is commit `f5f83f1`.
- Exclusive Uranus job `270001` passed the unmodified Qwen3 2A2F smoke in
  4 minutes 41 seconds. Three fixed prompts completed twice with exact output
  equality; all four GPUs showed role-appropriate memory and SM activity. The
  response hash is
  `c49027c49b6a5a8d785c375530cd4a236504799cd6af31ad0bf4db5ef90bded4`.
- Cleanup reset graphics-clock locks, left no compute processes, preserved a
  clean pinned plugin, and confirmed the isolated/protected runtimes remain
  vLLM 0.26.0 and 0.15.1 respectively. The raw smoke directory is
  `results/raw/milestone-m-afd-smoke-270001-20260929T214119Z`.
- The unmodified 2A2F prerequisite is complete. Static 64-expert/rank FFN
  placement is now implemented using vLLM 0.26's existing live
  `ExpertMapManager`, shared by checkpoint loading and kernel routing. A small
  source-controlled patch only connects the pinned AFD Qwen3 wrapper to the
  project implementation; it applies and reverts cleanly without touching
  site-packages or the protected environment.
- Two profile-free deterministic custom maps preserve 48 layers, 128 experts,
  and exactly 64 experts/rank/layer. Runtime evidence reconstructs ownership
  from each rank's live expert map after checkpoint loading.
- Project CPU tests pass `86` tests; the patched pinned-plugin Qwen3 targeted
  suite passes `35` tests. The planned H/H gate contains only default,
  balanced-round-robin, and deliberately different block mappings, with two
  deterministic smoke passes each. Routing profiling and the 18-run matrix
  remain gated on this job.
- The preparation implementation is committed and pushed as `0bb0172`. The
  immutable H/H correctness study is
  `results/raw/milestone-m-static-hh-20260930T081823Z`; Uranus job `270232`
  completed all three GPU configurations.
- Default, balanced round-robin, and different-block ownership each preserved
  48 layers, 128 experts/layer, and exactly 64 experts/rank/layer. Runtime
  evidence reconstructed three distinct live mappings. Both custom mappings
  matched their requested ownership exactly.
- Every configuration completed the three fixed prompts twice with exact
  deterministic semantic output/token equality to unmodified job `270001`.
  All four GPUs were active and read back at 2520 MHz.
- Slurm reports job `270232` as failed only because the final offline validator
  was launched by file path and could not import `src`. The preserved GPU
  artifacts validate successfully when the same tool is run as a module. The
  launcher and a regression test are corrected; the original exit status is
  retained in the raw study.
- Cleanup reset clocks, left no compute processes, reverted the plugin patch,
  and preserved vLLM 0.26.0 in the isolated environment and vLLM 0.15.1 in
  protected `cuda-env`. The static-placement H/H correctness gate passes.
- No routing profile or 18-run experiment has been started.
- Commit `b7b8c31` adds a selectable one-configuration correctness launcher.
  Required balanced-only retry job `270365` completed `COMPLETED/0` in 2:46;
  live ownership/output validation and all cleanup checks passed.
- Profiling-only code now uses vLLM 0.26's native logical-router capture hook,
  controlled by an explicit measurement-window file. Startup and warm-up are
  excluded; two FFN-rank dumps are retained and summed offline. The pinned
  plugin targeted suite passes 35 tests with the composite patch applied, and
  the patch reverts cleanly.
- The frozen profiling input is the existing 300-request GSM8K r16 schedule
  (source SHA-256 `b7cd1171e53fc36bba602ecd800a2a631e3c7505b7eb7d64c3306b0c0be47453`),
  with request seed normalized to 0. Profiling has not yet been submitted.

### Exact next command

```bash
cd /data/users/chjing/moe-energy-scheduling
study="results/raw/milestone-m-afd-profile-$(date -u +%Y%m%dT%H%M%SZ)"
third_party/afd-plugin/.venv/bin/python -m tools.prepare_milestone_m_afd_profile \
  --output-dir "${study}" \
  --source-trace results/raw/milestone-i-repeatability-20260901T091212Z-r16-validated/gsm8k-300-r16.jsonl \
  --smoke-study results/raw/milestone-m-static-hh-single-20260930T190346Z \
  --profile-patch patches/afd-plugin-0.26.0-static-profile.patch \
  --plugin-root third_party/afd-plugin
```

## Latest status — Milestone L real-GPU validation complete

Slurm array `267080` completed the 21-run original-versus-aligned P1-group
validation on Uranus. All seven tasks exited 0; every run completed 300/300
requests with zero failures, one trace hash, cumulative NVML energy,
target-matching clocks, valid client scheduling, and successful clock/patch
cleanup.

- MHHH alignment versus original changes TTFT p99 by `+3.027%`, TPOT p99 by
  `-0.428%`, total energy by `+0.267%`, and throughput by `+0.278%`.
- MMHH alignment changes TTFT p99 by `+1.141%`, TPOT p99 by `+0.106%`, total
  energy by `-0.003%`, and throughput by `+0.010%`.
- HHHH aligned-placement controls remain approximately neutral; their TPOT
  differences are `+0.511%` and `-0.150%`, within repeat variability.
- The frozen diagnostic confirms slower GPUs receive lighter intact P1 groups,
  but the measured total serving latency/energy effect is negligible.
- Decision: **C — no measured effect**. The Milestone K pressure proxy does not
  identify the serving critical path under the current runtime; do not build a
  broader joint scheduler from this mechanism.
- Primary FusedMoE timing was intentionally not collected because the existing
  path changes the runtime with extra patches and `--enforce-eager`.
- The complete CPU-safe suite passes: `62 passed, 2 warnings`.

Artifacts:

- Report: `docs/milestone_l_real_gpu_alignment.md`
- Raw study: `results/raw/milestone-l-alignment-20260923T141055Z`
- Per-run/aggregate/comparison tables: `results/processed/milestone_l_runs.csv`,
  `results/processed/milestone_l_aggregate.csv`, and
  `results/processed/milestone_l_comparisons.csv`
- Summary: `results/processed/milestone_l_summary.json`
- Figure: `results/figures/milestone_l_alignment_validation.svg`

### Exact next command

```bash
cd /data/users/chjing/moe-energy-scheduling
cat docs/milestone_l_real_gpu_alignment.md
cat results/processed/milestone_l_summary.json
```

## Latest status — Milestone L real-GPU validation submitted

Branch `milestone-l-real-gpu-alignment-validation` prepares the preregistered
21-run real-GPU comparison of original P1 versus intact-group-aligned P1 at
identical HHHH, MHHH, and MMHH frequency tuples.

- Seven configurations and three repetitions each are fixed; L-frequency
  points and any broader sweep are excluded.
- The validated Milestone I 300-request GSM8K trace at 16 req/s is reused.
- Placement construction uses frozen total expert counts and the exact
  Milestone K exhaustive permutation rule. All groups remain intact with 32
  experts per rank per layer.
- Configuration-task and repetition order use recorded seed `20260923`.
- Primary FusedMoE timing is disabled because the existing path requires extra
  patches plus `--enforce-eager`; primary behavior remains Milestone-I-like.
- The job is designed as an exclusive, serialized seven-task Slurm array on
  Uranus. Each task owns and terminates only its server and resets GPU clocks
  on success, failure, or signal.

Preparation commit `c63b9f3` is pushed. Immutable study directory
`results/raw/milestone-l-alignment-20260923T141055Z` records configuration
order `C4,C6,C1,C5,C7,C2,C3`, 21 expected runs, and all frozen hashes. The
exclusive serialized array was submitted as Slurm job `267080_[0-6%1]`.

The measured report, processed CSVs, and decision remain pending until this
array completes. Do not submit another copy while job 267080 is active.

### Exact next command

```bash
cd /data/users/chjing/moe-energy-scheduling
squeue -j 267080
```

## Latest status — Milestone K placement-DVFS alignment complete

Branch `milestone-k-placement-dvfs-alignment` contains an offline-only
exhaustive analysis of permuting the four intact per-layer P1 expert groups
onto heterogeneous physical-GPU frequencies. No GPU job was submitted and no
runtime, vLLM, expert grouping, placement implementation, or DVFS code changed.

- Frozen Milestone J/H profile and P1 hashes were verified. Every P1 group
  retains its original 32 logical experts in every layer.
- All 24 group-to-rank permutations were evaluated for every layer, phase,
  and one of 81 H/M/L frequency tuples using only the preregistered maximum
  normalized-pressure proxy.
- HHHH has exactly zero advantage and preserves original P1.
- Across all physical permutations, MHHH reduces mean maximum pressure by
  14.89--17.93% depending on phase/ordering; MMHH gives 14.50--18.38%; LMHH
  gives 18.85--23.48%.
- For physical MHHH, the slow GPU receives the lightest group in 48/48 layers.
  For MMHH, both slow GPUs receive the two lightest groups in 48/48 layers.
- Decision: **A — strong structural opportunity**. This justifies a small
  real-GPU original-versus-aligned validation, but makes no latency or energy
  claim.
- The complete CPU-safe suite passes: `56 passed, 2 warnings`.

Artifacts:

- Report: `docs/milestone_k_placement_dvfs_alignment.md`
- Analysis: `tools/analyze_rank_permutation_dvfs_alignment.py`
- Detailed/summary CSVs: `results/processed/milestone_k_alignment_layers.csv`
  and `results/processed/milestone_k_alignment_summary.csv`
- Manifest: `results/processed/milestone_k_manifest.json`
- Figures: `results/figures/milestone_k_mmhh_alignment.svg` and
  `results/figures/milestone_k_alignment_summary.svg`

### Exact next command

```bash
cd /data/users/chjing/moe-energy-scheduling
python3 tools/analyze_rank_permutation_dvfs_alignment.py
cat docs/milestone_k_placement_dvfs_alignment.md
```

## Latest status — Milestone J layer-wise load analysis complete

Branch `milestone-j-layerwise-load-analysis` contains an offline-only analysis
of the frozen phase-resolved GSM8K expert profile and the existing P1--P4
placements. No GPU job was submitted and no runtime, vLLM, placement, or DVFS
code was changed.

- The total/prefill/decode logical-expert arrays are each `48 x 128`; phase
  conservation and profile hash were verified.
- Every reused P1--P4 mapping is per layer and preserves exactly 32 experts per
  rank per layer.
- P1 aggregate total load is nearly balanced (`max/mean = 1.012`), but its
  individual-layer mean `max/mean` is `1.207`. The heavy rank changes at 40 of
  47 layer boundaries and no heavy/light identity run exceeds three layers.
- P2--P4 keep GPU0 heaviest for all 48 layers in total, prefill, and decode.
  This is persistent rank-specific imbalance, not changing regional slack.
- Decision: **Case B — little temporal opportunity**. The profile does not
  reveal natural stable multi-layer regions that strongly motivate a
  region-level dynamic DVFS controller.
- The complete CPU-safe suite passes: `51 passed, 2 warnings`.

Artifacts:

- Report: `docs/milestone_j_layerwise_load_analysis.md`
- Analysis: `tools/analyze_layerwise_loads.py`
- Detailed CSV: `results/processed/milestone_j_layer_loads.csv`
- Run/summary CSVs: `results/processed/milestone_j_persistence_runs.csv` and
  `results/processed/milestone_j_summary.csv`
- Manifest: `results/processed/milestone_j_manifest.json`
- Figures: `results/figures/milestone_j_p1_layer_load_heatmaps.svg` and
  `results/figures/milestone_j_placement_comparison.svg`

### Exact next command

```bash
cd /data/users/chjing/moe-energy-scheduling
python3 tools/analyze_layerwise_loads.py
cat docs/milestone_j_layerwise_load_analysis.md
```

## Latest status — Milestone I higher-load repeatability check complete

Branch `milestone-i-higher-load-repeatability` contains the completed final
motivation check. No service model, scheduler, placement heuristic, or
expert-load-based selection was used.

### Calibration and fixed experiment

- Calibration jobs `255371`, `255372`, `255374`, `255375`, and `255376` used
  P1/HHHH only. The measured complete-drain capacity lower bound is 22.434
  req/s; exact server saturation was not claimed because offered rates >=32
  req/s had invalid client send delay.
- The fixed 65% target selected 16 req/s, the nearest client-valid measured
  rate (71.3% of the lower bound). This was locked before P2/P3/P4 data.
- The final trace contains 300 unique GSM8K prompts, input-token mean/p50/p90
  139.49/135/161, and hash
  `b7cd1171e53fc36bba602ecd800a2a631e3c7505b7eb7d64c3306b0c0be47453`.
- Serialized array `255377` completed P3/P1/P2/P4 in seeded block order. All
  132 runs completed 300/300 requests with zero failures, cumulative NVML
  energy, target-matching observed clocks, one trace, and 3 repetitions for
  every one of the 44 placement/tuple configurations.
- Every controller exited 0 and reverted the vLLM patch; final global patch
  state is `not applied`. Clock-reset cleanup ran after every task.
- Five new CPU-safe pure-function tests pass. `pytest` is not installed in the
  existing Minerva environments, so no dependency was installed.

### Result and decision

- At HHHH, Sequential selects P2: TPOT p99 64.481 ± 0.153 ms, TTFT p99
  226.010 ± 3.156 ms, energy 21,368.932 ± 75.932 J.
- The preregistered 12/13/15 ms TPOT SLOs are all infeasible at this offered
  load and remain explicitly reported.
- The exploratory 0.1 ms SLO curve has a numerical 24.044% maximum only at
  70.7 ms. It is a boundary artifact: P1/P2 MMMM TPOT differs by only 0.049 ms
  while their repeat standard deviations are 0.398/0.496 ms. At 70.8 ms the
  saving is 0%.
- Excluding >=5% one-grid boundary effects, the maximum measured saving is
  3.221% and also occurs only over a narrow SLO interval.
- Decision: **B — weak opportunity**. Higher load reveals a real interaction,
  but not a repeatable >=5% benefit over a meaningful SLO interval. This does
  not justify proceeding directly to algorithm/system design.

### Artifacts

- Report: `docs/milestone_i_higher_load.md`
- Per-run table: `results/processed/milestone_i_runs.csv`
- Three-repeat aggregate: `results/processed/milestone_i_aggregate.csv`
- Pareto points: `results/processed/milestone_i_pareto.csv`
- Full exploratory curve: `results/processed/milestone_i_oracle_curve.csv`
- Summary: `results/processed/milestone_i_summary.json`
- Figure: `results/figures/milestone_i_pareto.svg`
- Raw study: `results/raw/milestone-i-repeatability-20260901T091212Z-r16-validated`

### Exact next command

```bash
cd /data/users/chjing/moe-energy-scheduling
cat docs/milestone_i_higher_load.md
cat results/processed/milestone_i_summary.json
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
