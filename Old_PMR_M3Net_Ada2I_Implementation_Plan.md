# Integrating PMR (Prototypical Modal Rebalance) into M3Net + Ada2I

## Implementation Plan

**Scope.** This plan adds the two PMR mechanisms — Prototypical Cross-Entropy (PCE) for accelerating slow-learning modalities and Prototypical Entropy Regularization (PER) for decelerating the dominant one — to the existing M3Net + AFW + AMW codebase described in `M3Net_Ada2I_Integration_Notes.md`. It follows the same engineering conventions already established there (attribute-based side-channels instead of changed return signatures, flag-gated code paths that are byte-identical when off, diagnostics-first validation).

**Status of inputs.** PMR (Fan et al., CVPR 2023) was designed and validated on two-modality, single-instance classification (CREMA-D, AVE, CG-MNIST) with trainable ResNet encoders and SGD. M3Net is a three-modality, graph-batched, conversation-level model with mostly frozen pre-extracted features and Adam. None of PMR's components can be used as-is; each needs a specific adaptation, laid out below.

---

## 0. Why PMR is a natural third piece (and where it differs from AMW)

| | AMW (already built) | PMR (this plan) |
|---|---|---|
| What it measures | Per-modality confidence from **classifier weight-column slicing** on the concatenated `smax_fc` input | Per-modality confidence from **non-parametric prototype classifiers** on per-modality representations |
| Assumption on fusion | Logits must be linearly decomposable by modality (true for M3Net's final concat, but only at the very last layer) | None — only needs a per-modality `[N, D]` tensor and labels |
| Intervention on dominant modality | Multiplicative gradient shrink on encoder params (`1 − tanh(αρ)`) | Entropy regularization on prototype-distance softmax (PER), early epochs only |
| Intervention on weak modality | None (ratio 1.0 → untouched) | **Active** acceleration via PCE loss — a gradient direction that does not pass through the fusion/graph stack |
| Paper's argument | — | Magnitude-only modulation (OGM-GE / AMW-style) is limited because the fused gradient's *direction* drifts away from each modality's own optimal direction (their Fig. 2c) |

The key research motivation for combining them: AMW only brakes the dominant modality; PMR additionally pushes the weak modalities along a fusion-independent direction. In M3Net this matters more than in a concat model, because the hypergraph convolutions mix modalities *before* the classifier — so the "direction interference" PMR describes is structurally stronger here than in the paper's setting.

Established fact to anchor the hypothesis: text dominates MELD and IEMOCAP in essentially every published ERC system (text-only results sit within ~1–3 F1 of full multimodal results; Ada2I's own analysis showed this). Expect `s_l ≫ s_a, s_v` from the first epoch.

---

## 1. Design decisions (resolve before coding)

### 1.1 Where to tap the per-modality representation `z_m`

PMR needs, per modality, one vector per labelled instance. In M3Net, an "instance" is an utterance node, and the model already exposes flat `[N, D]` tensors in dialogue-major order — so unlike AFW, **no segment-mean machinery is needed**; PMR's per-sample formulation maps directly onto M3Net's node list. The question is only *which* `[N, D]` tensor.

| Tap point | Tensor | What trains through PCE | Cross-modal contamination | Notes |
|---|---|---|---|---|
| **A. `pre_graph`** | Output of `linear_m` (+ GRU for text), i.e. the per-modality input to `simple_batch_graphify` | Only `linear_m` / `gru_m` — exactly the params AMW modulates | None — this is the last point where a modality is purely itself | Most faithful to PMR's "independent of other modalities" intent |
| B. `post_graph` | `E_m = V_m ⊕ F_m` from `group_by_modality` (pre-AFW) | Encoders + HyperGCN params | High — hyperedges already mixed modalities | PCE here partly re-introduces the interference PMR was designed to avoid |
| C. `post_afw` | `E_m` after AFW | Encoders + HyperGCN + AFW | High | Useful for measuring whether AFW reduces imbalance |
| D. `classifier` | `last_modality_feat` slices (AMW's tap) | Everything | High | Only useful for cross-checking PMR's ratio against AMW's ratio |

**Decision:** implement the tap as a flag `--pmr_tap {pre_graph,post_graph,post_afw}` with **`pre_graph` as the primary**, `post_graph` as the main ablation. All three share the same code path; only the stored attribute differs.

**Critical caveat for tap A (verify in code first, Phase 0):** per the integration notes, the audio and visual encoders are just `linear_a` / `linear_v` on frozen IS10 / denseface features. PCE at tap A can therefore only reshape a single linear projection for the two weak modalities. PMR assumes a trainable deep encoder per modality. Two mitigations, both flag-gated:

1. `--pmr_av_encoder mlp` — replace `linear_a` / `linear_v` with a 2-layer MLP (hidden = `D_g`, GELU, dropout) when PMR is on, so PCE has non-trivial capacity to act on. Must also be run *without* PMR as a control, since added capacity alone can move results.
2. Use tap B as the alternative where the trainable stack above the tap is deeper.

Both mitigations should be in the ablation matrix (§7). This is the single biggest risk that PMR shows no effect in this architecture, and it should be named explicitly in any writeup.

### 1.2 Generalizing the imbalance ratio from 2 to 3 modalities

PMR's Eq. (8) is pairwise (`ρ = Σp⁰ / Σp¹`). M3Net has `{l, a, v}` plus arbitrary subsets via `--modals`. Define, per modality `m` present in `modals`, the batch-mean prototype confidence on the true label:

```
s_m = (1 / |B|) · Σ_{i∈B} p_m(y_i | z_m,i)            (masked by umask)
```

Then use the same convention AMW already uses (ratio relative to the weakest):

```
ρ_m = s_m / min_l s_l            (≥ 1; the weakest modality has ρ = 1)
```

Coefficients, chosen so that they collapse **exactly** to PMR's Eqs. (11) and (13) when only two modalities are present:

```
PCE weight (accelerate):   β_m = clip(0, s_max / s_m − 1, 1)      → 0 for the dominant modality
PER weight (decelerate):   γ_m = clip(0, s_m / s_min − 1, 1)      → 0 for the weakest modality
```

Sanity check for the 2-modality case: `s_max/s_m − 1 = ρ − 1` for the weak one and `1/ρ − 1` for the strong one — matching the paper. For three modalities, both audio and visual get accelerated (likely both saturating at β = 1 given text dominance), and PER is applied to every modality that outperforms the weakest, proportionally.

Batch-level `s_m` will be noisy because a M3Net batch is a handful of dialogues (IEMOCAP: batch of 16 dialogues ≈ 700 utterances, reasonably stable; MELD batches are smaller in utterances). Add `--pmr_ratio_ema` (default 0.9) to smooth `s_m` across steps before computing β/γ. Log both raw and smoothed.

### 1.3 Distance function and scale

PMR uses plain Euclidean distance in the softmax. With `D_g`-dimensional GRU/linear outputs, raw distances can be large, which saturates the softmax, makes `p_m` near one-hot, and makes `s_m` a noisy 0/1 signal. Provide:

- `--pmr_distance {euclid, sq_euclid, cosine}` (default `euclid` to match the paper; `cosine` = L2-normalize both sides, distance = `1 − cos`)
- `--pmr_temp τ` (default 1.0) — logits = `−d / τ`. Recommended sweep: `τ ∈ {1, √D_g, D_g}` for Euclidean variants.

This is an extension beyond the paper; mark it as such. The paper's own setting (`euclid, τ=1`) must be one of the runs.

### 1.4 Prototype computation schedule

Paper: at the start of each epoch, forward a fixed 10% subset of the training data, compute class centroids per modality, EMA-update with momentum `ε`, detach. Table 5 shows ≥10% is stable, 1% is harmful.

Adaptation for M3Net:

- The training sets are small (IEMOCAP ≈ 5.8k train utterances; MELD ≈ 10k) and features are pre-extracted, so a `no_grad` pass over the **full** training loader in eval mode costs a few seconds. Default `--pmr_proto_subset 1.0` (full set); keep the option to subsample *dialogues* (not utterances, to preserve the graph-batched forward) for fidelity experiments.
- Momentum `--pmr_proto_momentum ε` default 0.9 (paper's `ε`; tune in `{0.5, 0.9, 0.99}`).
- First prototypes: computed at the start of epoch `--pmr_start_epoch` (default 1, i.e. after one vanilla epoch) rather than at random init — the paper does it from epoch 0 but with random-init ResNets; here a one-epoch warm-up gives less degenerate centroids. Also a run with `start_epoch=0` for fidelity.
- Prototypes are **detached** (no gradient into centroids). Stored as registered buffers on a small `PrototypeBank` module so they are checkpointed and resumable.
- **Class imbalance**: MELD is heavily neutral-dominated and IEMOCAP has few `fear`/`disgust`-like classes. Centroids are per-class means so counts don't bias the centroid itself, but minority-class prototypes are estimated from few samples and will be noisier. Log per-class sample counts; the EMA helps. Do not reweight PCE by class frequency in v1 (keep it faithful); note as a possible follow-up.
- Must use the same `umask` as `MaskedNLLLoss` to exclude padded utterances from both centroid computation and PCE.

### 1.5 Loss composition

With everything enabled, the training loss becomes:

```
L_final = L_task(log_prob, label)                      # NLL / MaskedNLL / Focal, per dataset (unchanged)
        + L_afw                                         # existing, if --use_afw
        + α · Σ_m  β_m · L_PCE^m                       # PMR acceleration, if --use_pmr
        − μ · Σ_m  γ_m · H(softmax(−d(z_m, C_m)/τ))    # PMR PER, if --use_pmr and epoch < pmr_reg_epochs
```

where `L_PCE^m` is the masked mean of `−log p_m(y_i | z_m,i)` (Eq. 9) and `H` is the masked mean per-sample entropy of the prototype-distance distribution (Eq. 13). Note the sign: subtracting entropy *maximizes* it, keeping the dominant modality's prototype distribution flat early on.

Defaults from the paper: `α ∈ {1, 2}`, `μ ∈ {1e-3, 1e-2}`. Because M3Net's NLL is on a different scale than ResNet CE on CREMA-D, these need re-tuning; start the sweep at `α ∈ {0.5, 1, 2}`, `μ ∈ {1e-3, 1e-2}`, `pmr_reg_epochs ∈ {3, 5, 10}` (M3Net trains for ~60 epochs on MELD / ~100+ on IEMOCAP, so "first few epochs" should be interpreted relative to that).

Gradient-flow property to preserve and test: at tap A, `L_PCE^m` and the PER term must produce **zero gradient** on every parameter except modality `m`'s encoder (`linear_m`, `gru_m`, and the optional MLP). Nothing in HyperGCN, AFW, or `smax_fc` should receive PCE gradient. This is the property that makes PCE "fusion-independent," and a unit test should assert it.

### 1.6 Interaction with AMW

Three concrete interaction points, each a design option with a flag:

1. **Redundancy between PER and AMW.** Both slow the dominant modality. PER acts through the loss (changes direction and magnitude of the text encoder's gradient); AMW rescales magnitude post-hoc. Running both may over-suppress text. The ablation matrix (§7) isolates this: `+AMW+PCE` (no PER) vs `+PCE+PER` (no AMW) vs all three.
2. **Unified ratio source.** AMW currently computes its ratio from classifier-column slicing. Add `--amw_ratio_source {classifier, prototype}`. With `prototype`, AMW's `ratio[m]` is taken from PMR's `ρ_m` (which is measured at the encoder output, i.e. where AMW actually intervenes). This is a genuinely new variant: AMW's magnitude modulation driven by a fusion-independent imbalance estimate. Log both ratios every step regardless of which drives AMW, so their agreement/disagreement becomes a diagnostic in itself.
3. **Ordering in the training step.** PCE/PER are part of the loss, so they go *before* `backward()`. AMW stays between `backward()` and `optimizer.step()`. Ensure AMW's gradient scaling is applied to the *combined* gradient (task + AFW + PCE + PER) on the dominant encoder. This is intended (paper's Algorithm 1 does the same with its fused loss), but document it.

### 1.7 Optimizer

PMR's analysis assumes SGD; Table 6 shows it still helps with Adam (CREMA-D: 59.8 → 65.3) but with inconsistent gains on AVE. M3Net uses Adam. Keep Adam; do not introduce SGD as a confound. Mention in writeup.

---

## 2. Module design

New file `pmr.py`, mirroring the layout of `afw.py`.

```
PrototypeBank(num_classes, dims: dict[m → D_m], modalities, momentum, distance, temp)
    buffers:   proto[m]  : [C, D_m]         (detached; EMA-updated once per epoch)
               initialized : bool           (False until first compute)
    methods:
        update(new_centroids: dict[m → [C, D_m]])
            first call: proto[m] ← new_centroids[m]
            later:      proto[m] ← ε·proto[m] + (1−ε)·new_centroids[m]
            classes with zero samples in this pass keep the old prototype
        logits(z: dict[m → [N, D_m]]) → dict[m → [N, C]]
            −d(z[m], proto[m]) / τ, with d per --pmr_distance
        unimodal_confidence(z, label, mask) → dict[m → scalar s_m]
            masked batch mean of softmax(logits)[label]

compute_prototypes(model, train_loader, modals, tap, subset_ratio, device)
    model.eval(); torch.no_grad()
    forward each (sub-sampled) batch, read model.last_pmr_repr[m]  (see §3)
    accumulate per-class sums and counts per modality (masked)
    return dict[m → [C, D_m]]; model.train()

pmr_coefficients(s: dict[m → float]) → (beta: dict, gamma: dict, rho: dict)
    implements §1.2 exactly

pce_loss(logits: dict, label, mask) → dict[m → scalar]
per_entropy(logits: dict, mask)     → dict[m → scalar]

pmr_total(bank, z, label, mask, alpha, mu, apply_per: bool, s_ema: dict)
    → (loss_scalar, stats_dict)      stats: s_m, rho_m, beta_m, gamma_m, L_PCE^m, H_m
```

Everything above operates on flat `[N, ·]` tensors; there is no per-dialogue logic, which is why this is simpler than AFW. `dia_len` is only needed for subsampling dialogues in `compute_prototypes`.

---

## 3. Model-side changes (`model.py`, `model_hyper.py`)

Follow the existing attribute side-channel pattern (as with `last_afw_loss`, `last_modality_feat`).

- `Model.__init__` gains `pmr_tap` (string or `None`) and `pmr_av_encoder` (`linear` | `mlp`). When `mlp`, `linear_a` / `linear_v` are replaced by the 2-layer MLP described in §1.1; `get_modality_param_groups` (AMW / diagnostics) must pick up the new parameters automatically — verify it enumerates by name prefix rather than by a hard-coded module list.
- `Model.forward` populates `self.last_pmr_repr : dict[m → [N, D_m]]` at the requested tap:
  - `pre_graph`: the per-modality tensors immediately *before* `simple_batch_graphify` concatenates them, masked/flattened to the same node order used downstream (check that the flattening order matches `group_by_modality`'s dialogue-major order — a shape test in §5 guards this).
  - `post_graph` / `post_afw`: `HyperGCN.forward` already builds `E_dict` before/after AFW; store a reference on `self.last_pmr_repr` there. Not cloned/detached — gradient must flow.
- When `pmr_tap is None`, none of this runs; the forward path stays byte-identical (regression test).
- `HyperGCN` needs no structural change beyond exposing `E_dict` pre- and post-AFW.

---

## 4. Training-loop changes (`train.py`)

Per-epoch:

```
if use_pmr and epoch >= pmr_start_epoch:
    centroids = compute_prototypes(model, train_loader, modals, pmr_tap, pmr_proto_subset)
    bank.update(centroids)
```

Per-step (training only; eval path unchanged except for diagnostics):

```
log_prob = model(...)                                   # populates last_pmr_repr, last_modality_feat, last_afw_loss
loss = criterion(log_prob, label)
loss = loss + afw_loss                                  # existing
if use_pmr and bank.initialized:
    z = model.last_pmr_repr
    pmr_loss, pmr_stats = pmr_total(bank, z, label, umask,
                                    alpha=pmr_alpha, mu=pmr_mu,
                                    apply_per=(epoch < pmr_reg_epochs),
                                    s_ema=s_ema_state)
    loss = loss + pmr_loss
loss.backward()
if use_amw and in AMW window:
    ratio = pmr_stats['rho']  if amw_ratio_source == 'prototype' and bank.initialized
            else get_modality_unimodal_ratio(model, modals, label)
    apply_amw_modulation(model, modals, ratio, amw_alpha)
optimizer.step()
```

`s_ema_state` is a per-modality EMA of `s_m` maintained across steps within the run (§1.2), reset never (it is a running estimate).

Evaluation-time additions (cheap, high value): after each epoch, compute **prototype nearest-centroid accuracy / weighted-F1 per modality on dev and test** using `bank.logits` on `last_pmr_repr`. This is a non-parametric unimodal probe of each branch — the same quantity the paper plots in Fig. 4a/b — and it directly answers "did the audio/visual branches actually get better," which neither the fused F1 nor AMW's ratio can tell you. Compute it also when `use_pmr` is off (prototypes then serve only as a probe, not a loss) so the baseline has the same curve. Flag: `--pmr_probe` (independent of `--use_pmr`).

### 4.1 New CLI surface

```
--use_pmr                 enable PCE (+PER) losses                       (default: off)
--pmr_tap                 pre_graph | post_graph | post_afw              (default: pre_graph)
--pmr_alpha               PCE weight α                                   (default: 1.0)
--pmr_mu                  PER weight μ                                   (default: 1e-2)
--pmr_reg_epochs          PER active for epoch < this                    (default: 5)
--pmr_start_epoch         first epoch prototypes are computed/used       (default: 1)
--pmr_proto_momentum      EMA ε on prototypes                            (default: 0.9)
--pmr_proto_subset        fraction of training dialogues for prototypes  (default: 1.0)
--pmr_distance            euclid | sq_euclid | cosine                    (default: euclid)
--pmr_temp                softmax temperature τ on −d                    (default: 1.0)
--pmr_ratio_ema           EMA on s_m for coefficient computation         (default: 0.9)
--pmr_av_encoder          linear | mlp                                   (default: linear)
--pmr_probe               log prototype-classifier per-modality metrics  (default: off)
--amw_ratio_source        classifier | prototype                         (default: classifier)
```

### 4.2 Diagnostics (extend `write_diagnostics_row`)

Per epoch, per modality: `s_m` (raw and EMA), `ρ_m` (prototype) and AMW's classifier-based ratio side by side, `β_m`, `γ_m`, `L_PCE^m`, `H_m`, prototype-probe dev/test F1, per-class prototype sample counts, and the existing representation/gradient norms. Optional (behind `--pmr_log_angles`): cosine angle between the gradient of `L_PCE^m` and the gradient of `L_task` on modality `m`'s encoder params — a direct replication of PMR's Fig. 2c in this architecture, and the cleanest evidence for/against the "direction interference" hypothesis here. This costs an extra backward per logged step; log every `k` steps, not every step.

---

## 5. Testing and validation (before any GPU run)

All on CPU with the synthetic-data harness already used for the modality-ablation fix.

1. **Regression**: `use_pmr=False, pmr_probe=False, pmr_av_encoder=linear` → outputs bit-identical to the current `+AFW+AMW` code on fixed seeds.
2. **Shapes across modality subsets**: all 7 subsets of `{a, v, l}` at all three tap points; `last_pmr_repr[m]` has `N` rows matching `sum(dia_len)` and the node order used by `group_by_modality` (check by injecting a known per-utterance marker through the pipeline).
3. **2-modality reduction**: with `--modals al`, assert `β, γ` equal PMR's Eqs. (11)/(13) on hand-computed `s` values, including the clip boundaries.
4. **Gradient isolation (tap A)**: with only `pmr_loss` backpropagated, assert `.grad` is `None`/zero on all HyperGCN, AFW, and `smax_fc` params and non-zero on `linear_m` / `gru_m` for each accelerated modality; zero on the dominant modality's encoder when `β_dom = 0` and PER is off.
5. **Prototype bank**: EMA arithmetic; classes absent from a pass keep old prototypes; buffers survive `state_dict` round-trip.
6. **Masking**: padded utterances contribute nothing to centroids, `s_m`, PCE, or PER (compare against an unpadded batch).
7. **AMW with `amw_ratio_source=prototype`**: ratio dict has the same keys/semantics as the classifier-based one (min = 1.0).
8. **Smoke run**: 2 epochs on a tiny real subset on Kaggle; confirm `s_l > s_a, s_v` appears, losses are finite, PER turns off at `pmr_reg_epochs`.

---

## 6. Phased rollout

| Phase | Work | Deliverable |
|---|---|---|
| 0 | Read `model.py` to confirm exactly which params form each modality encoder at tap A (is there a GRU for a/v?), confirm `simple_batch_graphify` node ordering, confirm `get_modality_param_groups` enumerates by name | Short note appended to this doc |
| 1 | `pmr.py`: `PrototypeBank`, `compute_prototypes`, coefficients, losses | Unit tests 3, 5 passing |
| 2 | Model taps (`last_pmr_repr`), optional a/v MLP encoder | Tests 1, 2, 4 passing |
| 3 | Train-loop wiring, CLI, prototype probe, diagnostics | Tests 6, 7, 8 passing |
| 4 | Fidelity run: paper settings (`euclid, τ=1, α=1, μ=1e-2, reg_epochs=5, start_epoch=0, subset=0.1`) on IEMOCAP, 1 seed | First data point |
| 5 | Ablation matrix + sweeps (§7) | Results table, diagnostics CSVs |
| 6 | Analysis: per-modality probe curves, ratio curves, angle plots; 3-seed confirmation of the best configs | Writeup section |

---

## 7. Experimental design

### 7.1 Core ablation (IEMOCAP and MELD, `--modals avl`, 3 seeds each for final table)

| # | AFW | AMW | PCE | PER | Purpose |
|---|---|---|---|---|---|
| 1 | – | – | – | – | M3Net baseline (existing) |
| 2 | ✓ | ✓ | – | – | Current best (existing) |
| 3 | – | – | ✓ | – | PCE alone — is the active acceleration the main effect, as in the paper's Table 2? |
| 4 | – | – | ✓ | ✓ | Full PMR alone |
| 5 | – | ✓ | ✓ | – | AMW brake + PCE push (no PER) — tests PER/AMW redundancy |
| 6 | ✓ | – | ✓ | ✓ | AFW + PMR — PMR as an AMW replacement |
| 7 | ✓ | ✓ | ✓ | – | All but PER |
| 8 | ✓ | ✓ | ✓ | ✓ | Everything |
| 9 | ✓ | ✓(proto ratio) | ✓ | ✓ | Unified ratio source |

Single-seed screening first; promote anything ≥ 0.5 F1 over row 2 on IEMOCAP (noise floor is 0.3–0.4) to 3 seeds. MELD gains will need 3 seeds to be interpretable at all.

### 7.2 Secondary sweeps (on the best row from 7.1, IEMOCAP, 1 seed each)

- Tap point: `pre_graph` vs `post_graph` vs `post_afw`
- `pmr_av_encoder`: `linear` vs `mlp`, **with a no-PMR `mlp` control**
- `α ∈ {0.5, 1, 2}`, `μ ∈ {1e-3, 1e-2}`, `reg_epochs ∈ {3, 5, 10}`
- `distance/temp`: `euclid/1` (paper), `sq_euclid/D_g`, `cosine/0.1`
- `proto_subset ∈ {0.1, 0.5, 1.0}` (replicates paper's Table 5)
- `proto_momentum ∈ {0.5, 0.9, 0.99}`

### 7.3 Modality-subset runs

Rows 1, 2, 4, 8 with `--modals al` and `--modals av` — this is where the 2-modality reduction of §1.2 is exercised, and where PMR's effect should be cleanest (one weak modality, one dominant).

### 7.4 Evidence bar (what counts as a result)

A positive claim requires, in this order of strength:

1. Fused W-F1 gain above the noise floor across 3 seeds.
2. The **prototype-probe F1 of audio and/or visual branches increases** relative to the same tap in row 2 — i.e. the weak modalities actually learned more, not just that the fused head re-weighted. This is the paper's Fig. 4b evidence and is the one that distinguishes "rebalancing" from "regularization."
3. The prototype `ρ_m` curve decreases over training relative to row 2 (paper's Fig. 4c).
4. (Optional, strongest mechanistic evidence) the PCE/task gradient angle on the weak encoders is large and grows in the baseline, supporting the paper's direction-interference claim in this architecture.

If 1 holds but 2 does not, the honest interpretation is that PCE acted as a regularizer on the encoders rather than as modality rebalancing.

---

## 8. Known risks and how the plan addresses them

| Risk | Mitigation |
|---|---|
| Weak-modality encoders are a single linear layer → PCE has nothing to shape | `--pmr_av_encoder mlp` + `post_graph` tap; both in the ablation with controls (§1.1) |
| Softmax over raw Euclidean distances saturates in high `D_g` | `--pmr_distance`, `--pmr_temp` (§1.3); paper setting kept as a run |
| Batch-level `s_m` too noisy with few dialogues per batch | EMA on `s_m` (§1.2); log raw vs. smoothed |
| PER + AMW double-suppress text | Rows 5/7 vs 4/8 isolate it; `pmr_reg_epochs` small by default |
| Minority-class prototypes (MELD) poorly estimated | Full-set prototypes by default, EMA, per-class counts logged; class-aware weighting deferred |
| Added MLP capacity confounds gains | Mandatory no-PMR MLP control |
| Prototype probe conflated with PMR effect | `--pmr_probe` is independent of `--use_pmr`; baseline gets the same curve |
| Adam vs. SGD assumption in the paper | Keep Adam, state it; paper's Table 6 supports applicability |
| Node-order mismatch between tap A and labels | Dedicated shape/order test (§5.2) |

---

## 9. Files touched (summary)

```
pmr.py           NEW  — PrototypeBank, compute_prototypes, pmr_coefficients,
                        pce_loss, per_entropy, pmr_total
model.py              — pmr_tap / pmr_av_encoder params; optional MLP a/v encoders;
                        exposes last_pmr_repr at the chosen tap
model_hyper.py        — HyperGCN exposes E_dict pre- and post-AFW for taps B/C
train.py              — per-epoch prototype refresh; PCE/PER added to loss;
                        amw_ratio_source switch; prototype probe on dev/test;
                        diagnostics columns; new CLI flags
tests/test_pmr.py NEW — §5 test suite
```

---

## 10. References

- Fan, Y., Xu, W., Wang, H., Wang, J., Guo, S. *PMR: Prototypical Modal Rebalance for Multimodal Learning.* CVPR 2023. (PCE / PER; Eqs. 6–13; Algorithm 1; Tables 2, 5, 6.)
- Peng, X., Wei, Y., Deng, A., Wang, D., Hu, D. *Balanced Multimodal Learning via On-the-fly Gradient Modulation.* CVPR 2022. (OGM-GE; the magnitude-modulation family AMW belongs to.)
- Wang, W., Tran, D., Feiszli, M. *What Makes Training Multi-modal Classification Networks Hard?* CVPR 2020. (Gradient blending; modalities overfit at different rates.)
- Snell, J., Swersky, K., Zemel, R. *Prototypical Networks for Few-shot Learning.* NeurIPS 2017. (Non-parametric prototype classifier, squared-Euclidean variant.)
- Chen, F. et al. *Multivariate, Multi-frequency and Multimodal: Rethinking Graph Neural Networks for Emotion Recognition in Conversation.* CVPR 2023. (M3Net.)
- Nguyen, C.V.T. et al. *Ada2I: Enhancing Modality Balance for Multimodal Conversational Emotion Recognition.* ACM MM 2024. (AFW / AMW.)

---

## Appendix A — Phase 0 findings (verified against the code)

Read of `model.py`, `model_hyper.py`, `train.py`, `dataloader.py` before implementation:

1. **Tap-A encoders.** `av_using_lstm` defaults to `False`, so at tap A the audio and visual
   encoders really are a single `nn.Linear` each (`linear_a`: `D_m_a → D_g`, `linear_v`:
   `D_m_v → D_g`); `gru_a` / `gru_v` are never even constructed. Text is `linear_l` + `gru_l`
   (bidirectional, 2 layers) — **except on MELD**, where `Model.forward` skips `linear_l`
   entirely (`if self.dataset=='MELD': pass`) and feeds the normalised RoBERTa average
   straight into `gru_l`. §1.1's caveat is confirmed; `--pmr_av_encoder mlp` is therefore
   implemented and is the recommended non-fidelity variant.

2. **No masking is needed.** `simple_batch_graphify` builds the flat node list as
   `cat([features[:lengths[j], j, :] for j in range(batch)])`, i.e. dialogue-major and
   already unpadded. `train.py` builds labels the identical way
   (`cat([label[j][:lengths[j]] ...])`). So `last_pmr_repr[m]` row `i` corresponds to
   `label[i]` by construction, with no padded rows present. §1.4's `umask` requirement is
   vacuous in this codebase; the PMR code takes an optional mask but `train.py` passes none.
   `group_by_modality` preserves this same order, so taps B/C align too.

3. **`get_modality_param_groups` enumerates by attribute name** (`linear_a`, `gru_a`, …),
   not by a hard-coded module list — so swapping `linear_a` for an `nn.Sequential` MLP under
   the same attribute name is picked up automatically by both AMW and the diagnostics.

4. **Dimensions.** `D_g` = 512 (IEMOCAP) / 1024 (MELD); `graph_h` = 512; `use_residue`
   defaults to `False`, so the tap-B/C dimension is `2 * graph_h = 1024`.
   `n_classes` = 6 (IEMOCAP) / 7 (MELD).

5. **Prototype-bank checkpointing.** `train.py` never saves a checkpoint during training
   (it only *loads* `best_model.pth.tar` under `--testing`), so the bank is kept as a
   standalone module rather than a `Model` submodule — this avoids changing
   `model.state_dict()` keys and breaking existing checkpoints. Prototypes are recomputed
   every epoch, so resumability is not a practical concern. Buffers are still registered.

6. **Pre-existing bug found (unrelated to PMR):** `runner.ipynb` appends `--ablation` when
   `ABLATION=True`, but `train.py` has no such argument — that path would have crashed with
   an argparse error. The notebook now runs the modality ablation as a Python loop over
   `--modals` instead.

---

## Appendix B — Implementation status

Phases 0–3 and the section-5 test suite are complete; phases 4–6 are GPU runs
that remain to be executed.

### Files

```
pmr.py                  NEW  — PrototypeBank, PrototypeAccumulator, compute_prototypes,
                                pmr_coefficients, pce_loss, per_entropy, pmr_total,
                                prototype_predictions, gradient_angle, pairwise_distance
model.py                     — pmr_tap / pmr_av_encoder params, _av_encoder helper,
                                pmr_dims, last_pmr_repr at the pre_graph tap
model_hyper.py               — HyperGCN gains pmr_tap / pmr_dim, exposes E_dict
                                pre- and post-AFW as last_pmr_repr
train.py                     — _forward_graph_batch / make_pmr_batch_forward (factored
                                out of the loop), per-epoch prototype refresh, PCE/PER
                                in the loss, amw_ratio_source switch, prototype probe,
                                extended diagnostics, 16 new CLI flags
tests/test_pmr.py       NEW  — 34 unit checks (section 5, tests 1–7)
tests/test_integration.py NEW— 26 end-to-end checks on synthetic IEMOCAP/MELD pickles
                                (section 5, test 8) across both datasets
tests/_shim.py          NEW  — CPU harness shims (torch_scatter, forced CPU)
tests/stub_pkgs/torch_scatter.py NEW — scatter_add stand-in for the subprocess tests
runner.ipynb                 — configuration cell for AFW/AMW/PMR, ablation runner
                                (single / method / modality), diagnostics reader
```

### Test results

All 34 unit checks and all 26 integration checks pass. Additionally, a
4-epoch synthetic run of the `+AFW+AMW` baseline was compared against `git
HEAD`'s `train.py` / `model.py` / `model_hyper.py`: **every epoch's train and
test loss, accuracy and F1 are bit-identical**, confirming the refactor and the
new flags leave the existing pipeline untouched when PMR is off.

### Deviations from the plan (all deliberate, all verified)

1. **`compute_prototypes` takes a `batch_forward` callback** rather than
   unpacking batches itself, so `pmr.py` carries no M3Net-specific batch
   layout. `train.py` supplies it via `make_pmr_batch_forward`, which routes
   through the same `_forward_graph_batch` the training loop uses — the
   prototype pass therefore cannot drift out of sync with training.

2. **`.clone()` on the tap-A representation.** Found during testing:
   `HyperGCN.forward` adds the speaker embedding to the text tensor *in place*
   (`l += spk_emb_vector`), which silently converted the tap into a
   post-speaker-embedding view and leaked PCE gradient into
   `graph_model.speaker_embeddings`. The clone restores the intended semantics;
   unit test 4 asserts HyperGCN now receives exactly zero PMR gradient.

3. **Coefficient divisors are floored, not epsilon-padded.** `s_max / (s_m + 1e-8)`
   perturbs every coefficient (0.45/0.4 − 1 came out at 0.12499997), which broke
   the exact reduction to Eq. (11). `max(s_m, 1e-12)` is used instead; unit test 3
   checks the reduction to the paper's formulas exactly, including the clips.

4. **No masking in the training path.** See Appendix A.2 — the representations
   are already unpadded and label-aligned. The `mask` argument exists throughout
   `pmr.py` and is unit-tested (test 6), but `train.py` passes `None`.

5. **The prototype bank is not a `Model` submodule**, to avoid changing
   `model.state_dict()` keys. See Appendix A.5.

6. **`--pmr_av_encoder` applies to all base models**, not only GRU, for
   consistency; it is a model-structure flag and is emitted by the notebook
   whether or not PMR's losses are enabled, which is how the mandatory no-PMR
   MLP control (section 8) is expressed.

7. **Prototypes are computed in `eval()` mode while PCE is applied in `train()`
   mode**, so dropout is active for one and not the other on the text encoder
   (and on the MLP a/v encoder). This matches the paper's Algorithm 1, which
   also computes detached prototypes in a separate feed-forward pass.

### Remaining work (phases 4–6)

Phase 4 (paper-fidelity run: `euclid, tau=1, alpha=1, mu=1e-2, reg_epochs=5,
start_epoch=0, proto_subset=0.1`, IEMOCAP, 1 seed), phase 5 (the section-7
ablation matrix and sweeps — the notebook's `RUN_MODE="method"` and
`RUN_MODE="modality"` run these directly), and phase 6 (analysis and 3-seed
confirmation) are GPU work and have not been run.
