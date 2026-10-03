# Milestone N: AFD FFN empirical cost model

## Scope

This milestone characterizes the current Qwen3-30B-A3B AFD 2A2F execution
path. It does not implement placement search, a joint scheduler, or dynamic
DVFS. The output is an empirical rank- and frequency-specific latency model
plus a power model fitted primarily from existing uninstrumented runs.

## Reused measurements

Primary power and window-energy data come from the completed, uninstrumented
Milestone M symmetric-frequency and target-split studies. Their processed-file
hashes and raw-directory identities are frozen in each study's
`reuse-manifest.json`. The colocated vLLM 0.15.1 Milestone G instrumentation is
an implementation reference only and is not model training data.

## Measurement boundaries and routing ground truth

The original preparation at `305dc6f` was invalidated before GPU execution.
It proposed joining one all-HH routing trace to later timing runs by
`(layer, call_sequence)`, and it timed the complete MoE block while calling the
result rank-local `C[g,f](n)`. Frequency and placement can change batching and
generated routes, so the cross-run join was not ground truth. In addition,
`FULL_DECODE_ONLY` bypasses Python layer hooks during graph replay; the first
implementation would not have emitted one fresh record per decode replay.

The corrected instrumentation stores routing and timing from the **same FFN
invocation and same run**. It embeds event records and a logical `topk_ids`
snapshot in each captured FFN CUDA graph, then resolves them at the beginning
of the next worker step. The AFD FFN worker already synchronizes CUDA after
every step, so this adds no per-call synchronization. Logical IDs are copied
after both timed intervals and histogrammed on CPU; no extra `bincount`, CPU
copy, or profiling kernel is inside either interval.

Two boundaries are retained:

- **Local fused-expert boundary:**
  `FusedMoEKernelModularImpl._fused_experts`, including local workspace
  preparation, expert assignment/permutation, and Triton expert kernels. It
  excludes EP all-gather dispatch and reduce-scatter combine. This is the only
  boundary used to fit `C[g,f](n)`.
- **Full per-layer MoE service boundary:** the complete native
  `Qwen3MoeSparseMoeBlock`, including router, EP dispatch, local experts,
  combine, and stream dependencies. It is retained to measure communication /
  synchronization residual and validate the later critical-path composition;
  it is not treated as an independent rank-local function of `n_g`.

Each physical FFN rank records its source-rank logical routes. The two rank
files are combined only within the same run on exact
`(step_sequence, stage, layer)` keys. The immutable placement then maps the
combined logical histogram to the two physical-rank local loads.

## Characterization matrix

One exclusive Uranus allocation performs eleven fresh-server lifecycles:

1. a mandatory first `P_balanced` at 2520/2520 MHz same-run alignment gate;
2. `P_balanced` profiling at symmetric FFN frequencies 1005, 1635, 2070, 2220,
   and 2520 MHz;
3. an independent lifecycle repeat at 1635 and 2070 MHz;
4. timing-only distribution checks for the existing 40/60 rank-0 and rank-1
   placements at 1635 and 2070 MHz.

Every configuration uses a fresh server, one complete conditioning trace, and
one measured trace. After the forced gate, order is randomized with seed
`20261007`. Attention remains at 2520/2520 MHz. Every run produces its own
same-run routing/local-timing/full-timing artifact; no cross-run alignment is
allowed.

## Offline models

The primary latency model is a binned interpolation table
`C[g,f](n)`, kept separate for physical GPUs 2 and 3. Here `n` is the number of
logical expert assignments owned by the physical FFN rank, reconstructed from
that run's routing histogram and immutable placement. Lifecycle repeats and
40/60 placements are held out as complete server lifecycles.

The power model is a physical-rank-specific linear relationship between routed
assignment rate and average power at 1635/2070 MHz. It is validated with
leave-one-configuration-out splits over uninstrumented runs. Other frequencies
remain balanced-load anchors; no unsupported load dependence is invented.
Multiplying predicted power by CUDA-event latency is reported only as a
derived per-call surrogate, never as direct NVML call energy.

## Results

Pending the consolidated GPU job and offline analysis.

## Decision gate

The next milestone may construct an offline `(placement, frequency)` oracle
only if held-out timing and power errors are adequate for correct candidate
ranking. If `n` alone fails and residuals depend on expert distribution, this
milestone stops with that negative result and uses the already captured richer
features for offline analysis rather than submitting another GPU job.
