# Milestone G — Controlled Load–Frequency Service Characterization

## Research questions

This milestone measures whether GPU0 DVFS sensitivity grows with controlled decode expert workload and whether a small `T(L,f)` model is accurate enough for a future offline Joint pilot.

## Platform and timing

Qwen/Qwen3-30B-A3B, vLLM 0.15.1, TP=1/DP=4/EP=4, four L40S GPUs on one Uranus node, `allgather_reducescatter`, fixed 50-request GSM8K trace at 0.5 requests/s, and clocks H/M/L = 2520/1635/1005 MHz. The MoE interval is one CUDA-event region around FusedMoE dispatch, expert compute, and combine on each rank. Events are rank-local and resolved only at the explicit profile dump; no collective runs in the measured forward. vLLM 0.15.1 cleared phase kwargs before the timing wrapper could retain them, so unknown records are classified explicitly by the observed non-speculative row boundary: 1–2 rows decode and >=129 rows prefill.

## Controlled load axis

| level | GPU0 decode load | GPU1 | GPU2 | GPU3 | decode max/mean |
|---|---:|---:|---:|---:|---:|
| controlled_low | 612845 | 608483 | 616687 | 600385 | 1.012 |
| controlled_medium | 1234496 | 391014 | 413403 | 399487 | 2.025 |
| controlled_high | 1576914 | 293021 | 283964 | 284501 | 2.587 |
| controlled_very_high | 1725728 | 237943 | 233784 | 240945 | 2.831 |

## DVFS penalties

Deltas are the target frequency minus the corresponding all-high run for the same placement.

| level | GPU0 decode load | transition | TPOT p99 penalty (ms) | MoE decode service penalty (ms / routed row) | energy delta (J) |
|---|---:|---|---:|---:|---:|
| HIGH | 1576914 | 2520->1635 | 6.370 | 0.035 | -1676.8 |
| HIGH | 1576914 | 2520->1005 | -10.456 | -0.094 | -2750.2 |
| LOW | 612845 | 2520->1635 | 4.595 | 0.306 | -2851.5 |
| LOW | 612845 | 2520->1005 | -34.225 | -0.053 | -2928.6 |
| MEDIUM | 1234496 | 2520->1635 | -43.104 | -0.195 | -2087.6 |
| MEDIUM | 1234496 | 2520->1005 | -59.190 | -0.277 | -2184.8 |
| VERY_HIGH | 1725728 | 2520->1635 | -34.462 | -0.199 | -2475.7 |
| VERY_HIGH | 1725728 | 2520->1005 | -53.805 | -0.330 | -2092.6 |

TPOT penalty correlation with GPU0 decode load: Pearson -0.178; MoE decode service penalty correlation: Pearson -0.606. These are four load levels × two transitions and are not a population estimate.

## Model comparison

| model | target | MAE (ms) | MAPE (%) | max abs (ms) | rank accuracy | LOO MAE (ms) | LOO MAPE (%) | LOO max abs (ms) | LOO rank accuracy |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| milestone_e_linear_pressure | tpot_p99_ms | 14.621 | 13.84 | 26.384 | 0.773 | n/a | n/a | n/a | n/a |
| milestone_f_amdahl | tpot_p99_ms | 102.085 | 87.99 | 128.417 | 0.208 | n/a | n/a | n/a | n/a |
| milestone_g_load_frequency | tpot_p99_ms | 13.269 | 12.45 | 29.332 | 0.712 | 17.35879056776514 | 16.206518802384565 | 43.0237329475751 | 0.6212121212121212 |
| milestone_g_load_frequency_moe | moe_decode_ms_per_row | 0.132 | 13.09 | 0.277 | 0.712 | 0.1710174727347197 | 16.843315941450808 | 0.39036674290479034 | 0.5151515151515151 |

Model C is the controlled load × frequency model `T = a + b·max_g(L_g fmax/f_g) + c·max_g(L_g(fmax/f_g − 1))`, with loads expressed in millions of decode routed tokens. It is fitted separately to TPOT and rank-max MoE decode service time per routed row.

## Milestone E offline rescoring

The eight existing Joint candidates were not rerun. Candidate predictions and 12/13/15 ms boundary classifications are in `results/processed/milestone_g_rescore_e.csv`.

Rescore MAE (E/F/G): 4.660 / 2.026 / 108.184 ms. G candidate rank-order accuracy is 0.036.

## Residual and communication diagnostic

No all-to-all or synchronization timer was available without invasive instrumentation; residuals are therefore only a plausibility diagnostic.

## Classification: G1_NOT_SUPPORTED

The controlled experiment did not show a reliable load-dependent DVFS penalty beyond noise; reconsider making placement part of the DVFS objective.

## Limitations

One model, one workload, one node, three clocks, no replication or migration, no forced-token diagnostic, and no direct all-to-all/synchronization timer. Custom placements can change BF16 reduction order and output identities; token identity is not used as a model feature. The model is diagnostic, not an online scheduler.
