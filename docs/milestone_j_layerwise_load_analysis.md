# Milestone J: Layer-wise post-placement expert-token load

## Question and scope

This offline analysis asks whether the existing fixed expert placements expose
stable multi-layer GPU-load regions that could motivate future region-level
DVFS. It launches no server or GPU job, changes no placement, and fits no
execution-time or service model.

The distinction is important:

- **Pre-placement demand** is the measured logical-expert activation vector
  `C[layer, :]`.
- **Post-placement GPU load** is
  `L[layer, gpu] = sum(C[layer, expert])` for experts mapped to that GPU.

Only the second quantity is called GPU load below. It is descriptive routed
expert-token load, not measured execution time.

## Frozen inputs and validation

The analysis reuses the phase-resolved GSM8K profile used by Milestones H/I:

- profile shape: `48 layers x 128 logical experts`;
- arrays: total, prefill, and decode counts;
- profile SHA-256:
  `d74af984fb9d9942e0eb15961657e13f0fd0e84c3eef823bdac6f4594533adda`;
- total counts equal prefill plus decode at every layer/expert.

The fixed mappings are P1 upstream EPLB, P2 controlled-medium, P3
controlled-high, and P4 controlled-very-high. Each is a `48 x 128` per-layer
`logical expert -> EP rank` mapping. Every layer maps all 128 logical experts
exactly once, with exactly 32 experts on each of ranks 0--3. No placement was
generated or changed for this analysis.

## Method

For each placement and phase, the resulting `48 x 4` load matrix is preserved
in the detailed CSV. Each layer is also normalized so its four fractions sum
to 1.0; `[0.25, 0.25, 0.25, 0.25]` is balanced.

Adjacent-layer similarity is histogram intersection:

`similarity(l-1,l) = sum_g min(v[l-1,g], v[l,g]) = 1 - L1(v[l-1],v[l])/2`.

It is 1 for identical normalized load signatures and 0 for disjoint ones.
Heavy/light identity runs are simple consecutive runs of the same argmax or
argmin GPU; ties are recorded explicitly, with the lowest rank used only for
the deterministic identity column. No region partitioning is performed.

## Main diagnostics

Total-profile results:

| Placement | Mean max/mean | P90 max/mean | Maximum | Mean adjacent similarity | Heavy-GPU switches / 47 | Longest heavy run | Longest light run |
|---|---:|---:|---:|---:|---:|---:|---:|
| P1 | 1.207 | 1.323 | 1.492 | 0.897 | 40 | 3 | 3 |
| P2 | 2.038 | 2.285 | 2.576 | 0.929 | 0 | 48 | 4 |
| P3 | 2.543 | 2.841 | 3.007 | 0.946 | 0 | 48 | 4 |
| P4 | 2.750 | 3.036 | 3.201 | 0.947 | 0 | 48 | 4 |

P1 is nearly balanced only after aggregating across all layers: its total
rank loads are `[5,131,284, 5,031,500, 5,153,294, 5,048,978]`, for aggregate
max/mean `1.012`. Individual layers are less balanced, with mean max/mean
`1.207`, but the identity of the heavy GPU changes at 40 of 47 boundaries.
The longest same-heavy and same-light runs are both only three layers. Across
the 48 total-profile layers, GPUs 0/1/2/3 are heaviest in 13/10/12/13 layers;
no rank remains systematically heavy.

The phase split gives the same qualitative result for P1:

| Phase | Mean max/mean | P90 max/mean | Mean adjacent similarity | Heavy switches | Light switches | Longest heavy/light run |
|---|---:|---:|---:|---:|---:|---:|
| Total | 1.207 | 1.323 | 0.897 | 40 | 36 | 3 / 3 |
| Prefill | 1.208 | 1.321 | 0.897 | 39 | 37 | 3 / 3 |
| Decode | 1.235 | 1.356 | 0.884 | 33 | 38 | 3 / 3 |

P2--P4 deliberately create a different pattern. GPU0 is the most-loaded rank
for all 48 layers in total, prefill, and decode. Their adjacent signatures are
similar and their heavy-rank runs span the whole model, while the least-loaded
rank still changes frequently. This is persistent rank-specific imbalance,
not a sequence of naturally distinct multi-layer regions. It is therefore the
kind of load structure already addressable by a run-wide heterogeneous
frequency tuple, whose measured benefit was weak in Milestones H/I.

## Figures

- [P1 total/prefill/decode heatmaps](../results/figures/milestone_j_p1_layer_load_heatmaps.svg)
- [P1--P4 comparison heatmaps](../results/figures/milestone_j_placement_comparison.svg)

All panels share one color scale and normalize each layer independently.

## Decision: Case B -- little temporal opportunity

The frozen profile does show meaningful within-layer expert-token imbalance
under P1, but it does **not** show stable heavy/light rank identities over
multi-layer regions: the heavy rank switches on 70--85% of adjacent-layer
boundaries across total, prefill, and decode, and no identity run exceeds three
layers. P2--P4 have stronger imbalance, but it is the same heavy rank across
all 48 layers rather than a changing regional pattern.

Therefore these measured profiles do not provide a strong motivation for a
region-level dynamic DVFS controller. Very fine-grained layer switching might
theoretically follow P1's rotating load, but this analysis contains no timing,
transition-overhead, or causal energy evidence to support it. Given the weak
static Placement x DVFS gains already measured, dynamic controller work should
not proceed without new evidence that its switching granularity and overhead
are practical.

## Artifacts and reproduction

- `results/processed/milestone_j_layer_loads.csv`: 576 layer rows with four
  absolute loads, four normalized fractions, imbalance, identities, and
  adjacent-layer diagnostics.
- `results/processed/milestone_j_persistence_runs.csv`: all heavy/light
  identity runs.
- `results/processed/milestone_j_summary.csv`: placement/phase summaries.
- `results/processed/milestone_j_manifest.json`: source hashes and definitions.

Reproduce on a CPU-only login node:

```bash
python3 tools/analyze_layerwise_loads.py
```
