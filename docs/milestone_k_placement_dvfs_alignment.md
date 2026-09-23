# Milestone K: P1 group-to-rank alignment with heterogeneous DVFS

## Research question

Can joint placement and heterogeneous DVFS exploit a degree of freedom that
Milestone J did not test: retaining each layer's four upstream-EPLB expert
groups while permuting those intact groups onto physical GPUs with different
frequencies?

This is an offline structural opportunity analysis. It does not measure or
predict latency, execution time, or energy, and it launches no GPU job.

## Frozen inputs and constraint

The analysis uses only the Milestone J/H frozen inputs:

- total, prefill, and decode logical-expert counts, each `48 x 128`;
- profile SHA-256:
  `d74af984fb9d9942e0eb15961657e13f0fd0e84c3eef823bdac6f4594533adda`;
- P1 upstream EPLB SHA-256:
  `0912c7744e87dc35b223ee8d145d6a6ac8bc6f3274a642b5cbb20d4a011454e7`.

P1 assigns all 128 logical experts exactly once in every layer, with 32
experts per original group. The analysis never changes membership within a
group. It enumerates all `4! = 24` assignments of those four intact groups to
physical GPUs. Thus this is a relabeling of existing P1 groups, not a new
expert grouping or placement heuristic.

## Alignment proxy

For each layer, the original P1 group loads are aggregated from logical-expert
counts and normalized to sum to 1. For every tuple in `{H,M,L}^4`, where
`H=2520`, `M=1635`, and `L=1005 MHz`, the baseline is the original P1
group-to-rank identity assignment.

For each candidate permutation, the structural pressure proxy is:

`max_g(load_fraction[layer,g] / (frequency[g] / H))`.

The lowest-pressure permutation is selected independently per layer. Exact
ties use fewer load/frequency rank-order inversions, minimum Hamming distance
from original P1, then lexicographic order. These are deterministic
tie-breakers only. The proxy is a capacity/alignment score, not predicted
runtime.

The exhaustive analysis contains 81 frequency tuples, three phases, 48
layers, and 24 evaluated assignments per layer. Equal-frequency HHHH retains
the identity mapping for all layers and has exactly zero alignment advantage.

## Focused heterogeneous tuples

The following table reports the range across **all physical permutations** of
each frequency multiset. Values are reductions in mean per-layer maximum
pressure relative to original P1 physical assignment.

| Frequency multiset | Physical tuples | Total reduction | Prefill reduction | Decode reduction |
|---|---:|---:|---:|---:|
| MHHH | 4 | 15.26--17.04% | 14.89--16.83% | 16.49--17.93% |
| LHHH | 4 | 19.36--21.27% | 19.08--21.11% | 21.45--23.39% |
| MMHH | 6 | 14.58--16.18% | 14.50--16.06% | 16.76--18.38% |
| LMHH | 12 | 19.11--21.41% | 18.85--21.28% | 21.88--23.48% |
| LLHH | 6 | 14.92--16.51% | 14.87--16.42% | 17.32--18.83% |
| LLLH | 4 | 9.21--10.48% | 9.32--10.68% | 10.27--11.08% |

The benefit is not confined to the most extreme L/H split. Tuples containing
only M and H still reduce the proxy by roughly 15--18% across phases.

## Representative physical assignments

These rows use the exact physical ordering written in the first column, not an
average over its frequency multiset.

| Tuple | Phase | Mean-pressure reduction | Layer P50 | Layer P90 | Changed layers | Distinct assignments | Adjacent changes / 47 |
|---|---|---:|---:|---:|---:|---:|---:|
| MHHH | Total | 16.40% | 13.02% | 35.12% | 38 | 4 | 36 |
| LHHH | Total | 20.93% | 14.93% | 39.97% | 38 | 4 | 36 |
| MMHH | Total | 14.86% | 12.92% | 29.28% | 39 | 6 | 43 |
| LMHH | Total | 20.78% | 14.93% | 39.14% | 47 | 12 | 44 |
| LLHH | Total | 14.94% | 12.92% | 29.28% | 39 | 6 | 43 |
| LLLH | Total | 9.21% | 7.16% | 21.12% | 35 | 4 | 40 |

For MHHH, physical GPU0 is the only slower GPU. Its original mean load share
is `0.252`; after alignment it is `0.199`, and it receives the lightest P1
group in all 48 layers. The three H GPUs receive mean shares
`0.265/0.268/0.268`.

For MMHH, GPUs 0 and 1 are slower. Their aligned mean shares are
`0.222/0.213`, compared with `0.282/0.284` on H GPUs 2 and 3. In every layer,
the two lighter groups are assigned to the two M GPUs. Which M GPU receives
the very lightest group changes by layer (21 versus 27 layers), but the
slow-GPU set remains consistently lighter.

All highlighted aligned tuples have zero load/frequency rank-order inversions
in all 48 layers. Original P1 typically does not because physical heavy/light
identity rotates independently of the fixed frequency tuple.

## Relation to Milestone J

Milestone J found no natural coarse temporal regions: under original P1, the
heavy physical rank changed at 40 of 47 total-profile layer boundaries. This
analysis does not contradict that result. The selected group permutation also
changes frequently (for example, 43 of 47 boundaries for MMHH), because the
identity of P1's light groups rotates.

The new structural observation is different: a static, per-layer placement
file can map each layer's already-existing lighter P1 groups to the same slow
physical GPUs. It does not require changing group membership or migrating
experts while serving. The MMHH heatmap shows the slow columns becoming
consistently lighter even though the selected original group IDs change from
layer to layer.

## Figures

- [Original versus aligned MMHH heatmaps](../results/figures/milestone_k_mmhh_alignment.svg)
- [Focused multiset summary](../results/figures/milestone_k_alignment_summary.svg)

## Decision: A -- strong structural opportunity

Intact per-layer P1 group permutation substantially and consistently improves
load-to-frequency alignment. Moderate M/H tuples improve the proxy by roughly
15--18%, extreme heterogeneous tuples reach roughly 19--23%, and the aligned
assignment achieves correct load/frequency rank order in every layer.

This supports one small real-GPU validation of P1-group rank permutation plus
heterogeneous DVFS. It does **not** establish a latency or energy benefit. A
real validation must preserve the exact groups and compare original versus
aligned mappings at the same frequency tuple and trace.

## Limitations

The analysis excludes frequency-transition overhead, communication topology,
Attention work, FusedMoE execution time, synchronization and waiting, measured
latency, and GPU energy. It also uses one frozen GSM8K routing profile. The
frequent layer-to-layer permutation changes describe a static per-layer model
layout; they are not evidence that runtime expert migration would be cheap.

## Artifacts and reproduction

- `results/processed/milestone_k_alignment_layers.csv`: every layer, phase,
  and frequency tuple, including original group loads and aligned GPU loads.
- `results/processed/milestone_k_alignment_summary.csv`: all 243
  phase/frequency summaries.
- `results/processed/milestone_k_manifest.json`: frozen hashes and exact
  objective/tie-break definitions.

Reproduce without GPUs:

```bash
python3 tools/analyze_rank_permutation_dvfs_alignment.py
```
