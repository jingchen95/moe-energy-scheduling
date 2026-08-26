# Milestone F — Decode-Aware Empirical Service Model

## Research question

Why did the load/frequency proxy used in Milestone E fail to predict the measured Joint frontier, and what is the smallest empirical replacement?

## Boundary

This diagnosis does not implement an online scheduler, a Joint optimizer, communication model, or new policy search. It reuses the accepted Milestone D/E measurements.

## Phase accounting

Phase profile available: **True**. Profile SHA-256: `d74af984fb9d9942e0eb15961657e13f0fd0e84c3eef823bdac6f4594533adda`.

The vLLM patch keeps the existing EPLB routed-token counter and adds opt-in prefill/decode windows. It uses local scheduler attention metadata and the existing DP-rank token-block ordering; no collective is executed by the phase mask during a measured forward. The dump rejects any per-expert conservation mismatch.
The accepted profile contains 20,365,056 routed expert assignments: 17,926,656 prefill (88.03%) and 2,438,400 decode (11.97%). Per-layer and per-expert totals are retained in `milestone_f_summary.json`; the CSV contains the GPU-assigned breakdown for P0/P1/P2.

## Placement loads

| placement | basis | GPU0 | GPU1 | GPU2 | GPU3 | max/mean |
|---|---|---:|---:|---:|---:|---:|
| P0_default_linear | aggregate | 5486348 | 4975237 | 4801756 | 5101715 | 1.078 |
| P0_default_linear | prefill | 4828159 | 4385263 | 4236514 | 4476720 | 1.077 |
| P0_default_linear | decode | 658189 | 589974 | 565242 | 624995 | 1.080 |
| P1_upstream_eplb | aggregate | 5131284 | 5031500 | 5153294 | 5048978 | 1.012 |
| P1_upstream_eplb | prefill | 4518439 | 4423017 | 4536607 | 4448593 | 1.012 |
| P1_upstream_eplb | decode | 612845 | 608483 | 616687 | 600385 | 1.012 |
| P2_diagnostic_load_concentrated | aggregate | 9058308 | 7562612 | 3032539 | 711597 | 1.779 |
| P2_diagnostic_load_concentrated | prefill | 8072153 | 6584826 | 2648651 | 621026 | 1.801 |
| P2_diagnostic_load_concentrated | decode | 986155 | 977786 | 383888 | 90571 | 1.618 |

## Frequency scaling

The old proxy assumes `TPOT(f) = TPOT(high) * fmax/f`. The replacement fits `T(f) = a + b*fmax/f` to the three measured uniform P1 points. The fitted fixed component, frequency-sensitive component, residuals, and leave-one-configuration-out errors are recorded in `results/processed/milestone_f_frequency_model.csv` and `milestone_f_summary.json`.

Amdahl fit: fixed = 8.3944 ms, frequency-sensitive at max = 2.7174 ms, sensitive fraction = 0.245, MAE = 0.0745 ms.
The old linear three-point proxy has MAE 5.8532 ms; the Amdahl fit reduces this error while retaining an interpretable fixed component.

## Candidate pressure versus measured TPOT

The original linear pressure is recomputed separately from aggregate, prefill-only, and decode-only loads. These eight-candidate correlations are descriptive only.

| basis | Pearson r | Spearman r | n |
|---|---:|---:|---:|
| aggregate | -0.414 | -0.333 | 8 |
| prefill | -0.403 | -0.333 | 8 |
| decode | -0.493 | -0.429 | 8 |

## Existing P1/P2 downclock diagnostic

The accepted Milestone D single-GPU perturbations provide a small load-dependent check. Deltas are target minus same-placement all-high reference; they are not new measurements.

| placement | GPU | aggregate load | prefill load | decode load | ΔTPOT p99 (ms) | Δtotal energy (J) | ΔGPU energy (J) |
|---|---:|---:|---:|---:|---:|---:|---:|
| P1_upstream_eplb | 0 | 5131284 | 4518439 | 612845 | 0.942 | -3164.3 | -3625.2 |
| P1_upstream_eplb | 1 | 5031500 | 4423017 | 608483 | 0.895 | -2772.3 | -3458.5 |
| P1_upstream_eplb | 2 | 5153294 | 4536607 | 616687 | 0.889 | -2782.1 | -3506.9 |
| P1_upstream_eplb | 3 | 5048978 | 4448593 | 600385 | 0.724 | -3147.2 | -3871.5 |
| P2_diagnostic_load_concentrated | 0 | 9058308 | 8072153 | 986155 | 1.719 | -6415.8 | -5216.8 |
| P2_diagnostic_load_concentrated | 1 | 7562612 | 6584826 | 977786 | 0.802 | -4389.7 | -3631.3 |
| P2_diagnostic_load_concentrated | 2 | 3032539 | 2648651 | 383888 | 0.844 | -4220.1 | -3528.1 |
| P2_diagnostic_load_concentrated | 3 | 711597 | 621026 | 90571 | 0.913 | -4210.5 | -3606.6 |

The corresponding load-versus-ΔTPOT correlations are stored in `milestone_f_summary.json`; with only eight perturbations they are diagnostic, not a fitted causal law.

## Service-model diagnosis

For each candidate, the model uses a max-GPU proxy `a + b*L_g + c*L_g*(fmax/f_g - 1)`. Fits are trained on the accepted 81-row Sequential P1 sweep and evaluated on the eight existing Joint candidates; no Joint measurement is changed or rerun.

Diagnostic classification: **B_FREQUENCY_MODEL_DOMINANT**.
For aggregate candidate rescoring, old-pressure MAE is 4.660 ms and the empirical model MAE is 2.026 ms; rank-order accuracy is 0.821 versus 0.893 for the old score.

The frequency-proportional part of the old score is contradicted by the uniform P1 measurements. Decode-only pressure does not produce a clearly better positive correlation on eight Joint candidates, and the empirical candidate model still has residual error; communication/synchronization and grouped-GEMM effects therefore remain plausible missing terms rather than established causes.

## Limitations

- One model, one L40S node, one GSM8K trace, and no expert replication or migration.
- The service model is deliberately small and should not be interpreted as a production predictor.
- Eight Joint candidates are a diagnostic sample, not a new search or a proof of a global optimum.
- Any phase-profile mismatch or unsupported execution branch is reported rather than silently imputed.

The next milestone remains a research decision based on this diagnosis; this milestone does not begin the Joint scheduler.
