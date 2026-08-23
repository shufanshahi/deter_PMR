# Codebase Explanation: M3Net + Ada2I (AFW/AMW) + PMR (PCE/PER)

This repository is a single PyTorch training pipeline for multimodal Emotion
Recognition in Conversation (ERC) on IEMOCAP and MELD. It merges three
published methods into one model:

1. **M3Net** (Chen et al., CVPR 2023) — the backbone architecture. A
   hypergraph neural network that fuses text/audio/visual modalities and
   conversational context, plus a parallel frequency-adaptive graph branch.
2. **Ada2I** (Nguyen et al., ACM MM 2024) — two bolt-on modules, **AFW**
   (Adaptive Feature Weighting) and **AMW** (Adaptive Modality Weighting),
   that rebalance the contribution of each modality at the feature level and
   the gradient level respectively.
3. **PMR** (Fan et al., CVPR 2023) — a third, independent rebalancing
   mechanism, **PCE** (Prototypical Cross-Entropy) and **PER** (Prototypical
   Entropy Regularization), that pushes weak modalities using a
   fusion-independent, prototype-based signal instead of a gradient-magnitude
   rescaling.

All three are additive and individually flag-gated: with every `--use_*`
flag off, the model is exactly M3Net as published. The three source papers
(`M3net.md`, `ada2i.md`, `PMR.md`) and the design document
(`PMR_M3Net_Ada2I_Implementation_Plan.md`) live alongside the code in this
directory and are the primary references for the adaptation decisions
summarized below.

```
M3Net (backbone)                Ada2I (feature/modality balance)   PMR (prototype balance)
─────────────────                ─────────────────────────────      ────────────────────────
modality encoders                AFW: tensor-attention               PCE: accelerate weak
  ↓                                 re-weighting of E_m                modalities via
hypergraph conv (multivariate)     (afw.py, inside HyperGCN)           prototype softmax loss
  +                                                                    (pmr.py)
frequency-adaptive conv            AMW: gradient-magnitude
(multi-frequency)                    modulation of encoder            PER: decelerate the
  ↓                                   params (train.py)                 dominant modality via
concat + classifier                                                    entropy regularization
```

---

## 1. Repository map

| File | Role |
|---|---|
| `train.py` | CLI entry point; argument parsing, data loaders, training/eval loop, AMW application, PMR wiring, diagnostics, checkpointing, multi-seed sweeps. |
| `dataloader.py` | `IEMOCAPDataset` / `MELDDataset`: unpickle pre-extracted features, collate into padded per-dialogue batches. |
| `model.py` | `Model`: per-modality input encoders (Linear/GRU/LSTM/Transformer), RoBERTa-layer normalization/averaging, dispatch into the graph backbone, final classifier. Also legacy loss classes and older (non-hypergraph) fusion code. |
| `model_hyper.py` | `HyperGCN`: the M3Net backbone — hypergraph construction, multivariate propagation (`HypergraphConv`), multi-frequency propagation (`highConv`), AFW hook, PMR tap points. |
| `HypergraphConv.py` | M3Net's hypergraph convolution layer with edge-dependent node weights (paper §3.2.2, Eq. 5). |
| `high_fre_conv.py` | M3Net's FAGCN-style self-gated frequency filter (paper §3.3, Eqs. 9–11). |
| `afw.py` | Ada2I-inspired **Adaptive Feature Weighting**, reimplemented for M3Net's flat graph-batched tensors. |
| `pmr.py` | PMR's **PrototypeBank**, PCE loss, PER entropy term, and the N-modality generalization of PMR's imbalance ratio/coefficients. |
| `model_GCN.py` | Legacy/alternative graph modules (`GCNLayer1`, `GCN_2Layers`, `GCNII`, `GCNII_lyc`, `TextCNN`) from an earlier (`graph_type != 'hyper'`) code path. Only `TextCNN` is reachable from the default configuration (behind `use_bert_seq`, which defaults to `False`), so this file is effectively dead code in the `hyper` pipeline documented below. |
| `deterministic_scatter.py` | `index_add_`-based replacement for PyG/`torch_scatter`'s CUDA-nondeterministic `scatter_add`, used by both graph convolutions. |
| `runner.ipynb` | Kaggle-oriented notebook: environment setup, dataset staging, a `build_command()` helper that assembles `train.py` CLI invocations, and a 9-row AFW/AMW/PCE/PER ablation matrix (§7 of the implementation plan) plus a modality-subset sweep. |
| `README.md` | Original M3Net usage instructions (requirements, example commands). |
| `ada2i.md`, `M3net.md`, `PMR.md` | The three source papers, extracted to Markdown. |
| `PMR_M3Net_Ada2I_Implementation_Plan.md` | Design document for the PMR integration: tap-point choices, the 2→N modality generalization, deviations from the paper, test plan, and an implementation-status appendix. |

The standard training entry point (per `README.md`) is:

```
python -u train.py --base-model 'GRU' --dropout 0.5 --lr 0.0001 --batch-size 16 \
  --graph_type='hyper' --epochs=80 --graph_construct='direct' --multi_modal \
  --mm_fusion_mthd='concat_DHT' --modals='avl' --Dataset='IEMOCAP' --norm BN \
  --num_L=3 --num_K=4
```

`--graph_type='hyper'` and `--mm_fusion_mthd='concat_DHT'` select the code
paths this document describes (`HyperGCN` + the two-branch V/F concat
classifier). Other values of `graph_type` (`GCN3`, `DeepGCN`, `relation`,
`None`) reach the legacy code in `model.py`/`model_GCN.py` and are not used
by any documented training command.

---

## 2. Data pipeline (`dataloader.py`)

Both datasets store one entry per **dialogue** (a variable-length sequence of
utterances), not per utterance. `__getitem__` returns, per dialogue:

- `roberta1..roberta4`: four RoBERTa-Large layer activations per utterance
  (`[seq_len, 1024]` each) — M3Net's paper extracts the final four layers of
  a fine-tuned RoBERTa-Large and averages them (see `M3net.md` §4.2); here
  the four are kept separate and averaged/normalized inside `Model.forward`
  instead (see §3.1 below).
- `videoVisual`: pre-extracted visual features (DenseNet features on MELD,
  3D-CNN on IEMOCAP per the M3Net paper; `denseface`, dim 342, per
  `train.py`'s `feat2dim`).
- `videoAudio`: pre-extracted acoustic features (openSMILE-derived IS10, dim
  1582 on IEMOCAP; a 300-dim MELD-specific audio feature).
- speaker one-hot (`qmask`), an all-ones utterance mask (`umask`, present for
  API compatibility — see the note on masking in §6), integer emotion labels,
  and the dialogue id.

`collate_fn` uses `pandas` + `torch.nn.utils.rnn.pad_sequence` to stack
dialogues of different lengths into `[max_seq_len, batch, dim]` tensors
(features) and `[batch, max_seq_len]` (mask/labels), i.e. **time-major**,
padded batches — the same convention DialogueRNN-family ERC code uses.

---

## 3. Forward pass walkthrough

### 3.1 `Model.forward` — modality encoders (`model.py:759-938`)

1. **RoBERTa layer fusion.** The four RoBERTa layers `r1..r4` are each
   normalized (`--norm` selects `LN`, `BN`, `LN2`, or none — `LN2` is a
   parameter-free per-utterance LayerNorm over `(seq_len, feature_dim)`; `BN`
   is the default used by the training commands above) and then averaged:
   `U = (r1+r2+r3+r4)/4`. This is the code's version of M3Net paper's "average
   the final four layers" step, but with a learnable/parameter-free
   normalization inserted first — a detail not in the original paper.
2. **Per-modality encoders**, selected by `--base-model` (`GRU` is what the
   README commands use):
   - Audio/visual: `linear_a` / `linear_v`, each a single `nn.Linear`
     (`D_m_a → D_g` / `D_m_v → D_g`) by default. If `--av_using_lstm` a GRU is
     stacked on top; the README commands leave it off, so **audio and visual
     are, by default, a single linear projection of frozen features** — this
     is smaller capacity than M3Net's own audio/visual encoder description
     and is exactly the caveat flagged in the implementation plan (§1.1,
     Appendix A.1) when PMR taps this point.
   - Text: `linear_l` (skipped entirely on MELD — `Model.forward` special-cases
     `if self.dataset == 'MELD': pass`, feeding the normalized RoBERTa average
     straight into the GRU) followed by a bidirectional 2-layer `gru_l`.
   - `self.pmr_av_encoder='mlp'` swaps `linear_a`/`linear_v` for a 2-layer
     `nn.Sequential(Linear, GELU, Dropout, Linear)` under the *same attribute
     name* (`_av_encoder`, `model.py:728-745`), so downstream code that
     enumerates parameters by attribute name (AMW's `get_modality_param_groups`
     in `train.py`) picks up the new parameters automatically without any
     other change. This is a PMR-only option (see §5).
3. **`simple_batch_graphify`** (`model.py:458-468`) strips padding: for each
   modality it concatenates `features[:lengths[j], j, :]` over the batch, in
   dialogue order, producing a flat `[N, D_g]` tensor where
   `N = sum(dia_len)` and each row is one utterance of one modality, in
   **dialogue-major** order (all utterances of dialogue 0, then dialogue 1,
   …). This flattening — and its exact row order — is the contract every
   downstream module (`HyperGCN`, AFW, PMR, AMW) relies on; `train.py` builds
   the label vector the same way (`_forward_graph_batch`, `train.py:347`), so
   representations and labels stay row-aligned with no mask needed (see §6).
4. **PMR `pre_graph` tap** (`model.py:875-885`): if `pmr_tap == 'pre_graph'`,
   `self.last_pmr_repr` is set to a `.clone()` of each modality's flat
   `[N, D_g]` tensor here — the last point at which a modality's
   representation is purely a function of its own encoder. The `.clone()` is
   load-bearing, not defensive: `HyperGCN.forward` later does `l +=
   spk_emb_vector` **in place**, which would otherwise retroactively turn this
   tap into a post-speaker-embedding view and leak PCE gradient into the
   speaker-embedding table (documented as a bug found during testing in the
   implementation plan, Appendix B, deviation 2).
5. Dispatch into `self.graph_model` (`HyperGCN`, when `graph_type='hyper'`),
   dropout + ReLU, then `smax_fc` → `log_softmax` → `log_prob`.
6. **AMW tap** (`model.py:925-935`): after the classifier's input activation
   is computed, `Model.forward` slices it back into contiguous per-modality
   chunks (`self.last_modality_feat[m]`) using `HyperGCN.last_modality_dims`.
   Because dropout/ReLU are elementwise, slicing the *activated* tensor is
   equivalent to slicing `smax_fc`'s matching input columns — which is what
   `get_modality_unimodal_ratio` in `train.py` does with those slices and
   `smax_fc.weight`/`bias` (§5).

### 3.2 `HyperGCN.forward` — the M3Net backbone (`model_hyper.py:176-246`)

This is the direct implementation of M3Net paper §3.2–3.3 (Fig. 2).

1. **Speaker embedding.** `spk_idx = argmax(qmask)`; `spk_emb_vector =
   speaker_embeddings(spk_idx)` is added to the text node features only
   (`l += spk_emb_vector`), matching the paper's "speaker- and context-aware
   unimodal representation" (Eq. 3), except here it's applied to only the
   `l` modality rather than all three as in the paper — a deviation from
   the paper's Eq. 3, which the audio/visual branches skip. (`use_position`
   and `use_modal` are separate optional embeddings, both off by default.)
2. **`create_hyper_index`** (`model_hyper.py:250-322`) builds the hypergraph
   `H = (V_H, E_H, ω, γ)` of paper §3.2.1:
   - **Nodes**: one per (utterance, present modality) — `|V_H| = 3N` when all
     three modalities are present (`N` = utterances in the batch).
   - **Contextual hyperedges**: for each modality `m`, one hyperedge connecting
     all utterances of that modality within a dialogue (`hyperedge_type1 = 1`).
   - **Multimodal hyperedges**: for each utterance, one hyperedge connecting
     its nodes across all present modalities (`hyperedge_type1 = 0`).
   - `hyperedge_attr1` / `hyperedge_attr2` are the two learnable hyperedge
     *type* embeddings (the paper's edge weight `ω(e)`, simplified to one
     value per edge **type** rather than one per edge); `hyperedge_weight`
     and `EW_weight` (both randomly initialized learnable vectors, sliced
     to the batch's edge/incidence count) supply the paper's edge weight and
     edge-dependent node weight `γ_e(v)` respectively — matching the paper's
     explicit design choice to use randomly initialized rather than
     similarity-derived weights (`M3net.md` §3.2.1, "Weights").
3. **Multivariate propagation** (paper §3.2.2, Eq. 5): `x1 = fc1(features)`
   initializes node features; `num_L` (`--num_L`) stacked `HypergraphConv`
   layers propagate over the hypergraph incidence structure using
   `hyperedge_weight` and `EW_weight`. `HypergraphConv.forward`
   (`HypergraphConv.py:94-160`) implements the paper's two-stage
   node→hyperedge→node propagation exactly:
   ```
   D = 1 / Σ_e hyperedge_weight[e]         over edges incident to each node
   B = 1 / Σ_v EW_weight[v]                over nodes incident to each edge
   out = propagate(source_to_target, x, norm=B)   # node -> hyperedge
   out = propagate(target_to_source, out, norm=D) # hyperedge -> node
   ```
   which is `V^(l+1) = σ(D⁻¹ H Wₑ B⁻¹ Ĥᵀ V^(l))` (paper Eq. 5). The
   `aggregate` step is overridden to use `deterministic_scatter_add` instead
   of PyG's default scatter (see §7). After `num_L` layers, `out` is the
   paper's multivariate representation `V`.
4. **Multi-frequency propagation** (paper §3.3, Eqs. 7–11): `create_gnn_index`
   (`model_hyper.py:337-372`) builds the **pairwise** undirected graph `G`
   with the same connectivity pattern (same-modality-same-dialogue edges +
   same-utterance-cross-modality edges), but as ordinary pairwise edges
   rather than hyperedges. `num_K` (`--num_K`) stacked `highConv` layers
   (`high_fre_conv.py`) implement the FAGCN-style self-gated frequency filter:
   ```python
   alpha_g = tanh(gate(cat([x_i, x_j])))     # paper's r_l - r_h (Eq. 11)
   message = norm * x_j * alpha_g            # signed, so this can subtract
   ```
   accumulated with a residual (`gnn_out = gnn_out + conv_k(gnn_out, ...)`,
   matching Eq. 10's `f^(k+1) = f^(k) + Σ...`). Because `alpha_g ∈ (-1, 1)`
   is a *learned, signed* per-edge gate rather than a fixed low-pass kernel,
   this lets the network attenuate rather than only average neighbor
   features — the paper's mechanism for injecting high-frequency
   (discrepancy) information that a plain hypergraph conv (a low-pass filter)
   would erase. This differs from `M3net.md` §3.3.4's stated distinction from
   FAGCN in one respect: the paper says M3Net updates from the *previous
   layer's* output (not the first layer's input, as FAGCN does) — which this
   `gnn_out = gnn_out + conv(gnn_out)` recursion matches.
5. **Fusion of the two branches**: `out2 = cat([out, gnn_out], dim=1)`, then
   (if `use_residue`, default `False` in `train.py`'s argparse but enabled by
   the constant `use_residue=self.use_residue` passed through — check the
   run's `--use_residue` flag) concatenated again with the raw input
   `features`. This is the paper's Eq. 13 (`e_i = v_t ⊕ f_t ⊕ v_a ⊕ f_a ⊕
   v_v ⊕ f_v`), except the paper concatenates per-modality; here `out`/`gnn_out`
   already interleave all modalities' nodes and are only split back apart by
   modality afterward (`group_by_modality`, step 6).
6. **`group_by_modality`** (`model_hyper.py:324-334`) splits the flat
   `[num_modality*N, D]` tensor back into `E_dict = {m: [N, D]}` per present
   modality, using the same dialogue-major ordering `create_hyper_index`
   produced. This dict is:
   - PMR's `post_graph` tap, if requested (`self.last_pmr_repr = E_dict`,
     before AFW).
   - Passed through `self.afw` (Ada2I's AFW, §4) if `--use_afw`, producing a
     rebalanced `E_dict` and `self.last_afw_loss`.
   - PMR's `post_afw` tap, if requested (after AFW).
   - Finally concatenated across present modalities
     (`out1 = cat([E_dict[m] for m in present], dim=-1)`) and returned to
     `Model.forward` for the classifier — this concatenated tensor is also
     what `Model.forward` slices per-modality for AMW (step 6 in §3.1).

### 3.3 Classifier

`Model.forward` (hyper branch): `emotions_feat = graph_model(...)` →
`dropout` → `ReLU` → `smax_fc` (a single `nn.Linear` over the concatenation of
all present modalities' `E_m`) → `log_softmax`. Because `smax_fc` operates on
a concatenation, its weight matrix decomposes cleanly into per-modality
column blocks — the property AMW's classifier-based ratio (§5) depends on,
and the same "logits must be linearly decomposable by modality" caveat the
PMR paper names as a limitation of concatenation/summation fusion methods
(`PMR.md` §4.1) that motivated PMR's prototype-based alternative.

---

## 4. Ada2I integration: AFW (`afw.py`)

### What Ada2I's AFW does (paper)

Ada2I's Adaptive Feature Weighting (`ada2i.md` §3.3) computes tensor-ring
query/key representations per modality, contracts them into an attention
coefficient matrix, pools it into a feature-aware attention matrix, and
combines it with the original feature via a residual gate `β`. A companion
**Attention Mapping Network** (§3.5, Eq. 12) is trained with an L1 loss to
keep the attention weights consistent with the original unimodal features —
this is `L_feature` in Ada2I's joint loss (Eq. 14).

### Why it needed reimplementation, not reuse

Ada2I assumes padded `[batch, seq_len, dim]` tensors and pools each
modality's attention summary with `.mean(dim=1)` over the sequence axis.
M3Net's `HyperGCN` produces a flat, graph-batched `[N, dim]` node list with no
batch/seq axes to average over (`afw.py:1-14` docstring). `afw.py`
reimplements the same tensor-contraction idea but replaces the sequence-mean
with a **per-dialogue segment-mean**, using `dia_len` boundaries:

```python
def segment_mean_broadcast(x, dia_len):
    # replaces each node's value with its dialogue's mean, broadcast back
    # to every node in that dialogue — the flat-node equivalent of Ada2I's
    # "pool over this conversation's utterances"
```

### Module structure (`afw.py`)

- **`M3NetTensorAttention`**: for each modality `m`, builds two rank-`r`
  outer-product tensors `G_q = trans_q1(x_m) ⊗ trans_q2(x_m)` and
  `G_k = trans_k1(x_m) ⊗ trans_k2(x_m)` (Ada2I's tensor-ring-style query/key,
  simplified from a full tensor-ring decomposition to a single Khatri-Rao-like
  outer product), multiplies them elementwise (`G_qk = G_q * G_k`), then
  contracts `G_qk[m]` against every *other* present modality's segment-mean
  `M_bcast[other]` via `einsum('nkl,nlo->nko', ...)` — this is the
  cross-modal, inter-modal-interaction step Ada2I's AFW is named for. The
  contracted tensor is projected back to `dim_m` (`lin_att`) and combined with
  the original feature: `aware[m] = attn * x[m] + β * x[m]` — Ada2I's Eq. 7
  balancing parameter `β` (`--afw_beta`, default 0.7).
- **`M3NetAttentionMapping`**: reimplements Ada2I's Attention Mapping Network
  (Eq. 11) as a per-modality linear map `fc[m]` from the *raw* input `x[m]` to
  the *attended* output `aware[m]`, trained with an L1 loss
  (`M3NetAttentionMapping.get_loss`) — this is `L_feature` (Ada2I Eq. 12).
- **`M3NetAFWLayer`** wraps attention + dropout + the mapping-network loss for
  one layer; **`M3NetAFW`** stacks `afw_nlayers` (default 2) of these,
  summing their L1 losses into a single `total_loss` returned alongside the
  rebalanced `E_dict`.

`HyperGCN.__init__` builds `self.afw = M3NetAFW({m: e_dim for m in present},
afw_rank, present, afw_beta, afw_nlayers, afw_dropout)` when `--use_afw` is
set, where `e_dim` is the dimension of `E_m` (`n_dim + 2*nhidden` if
`use_residue` else `2*nhidden`). `HyperGCN.forward` calls it on `E_dict`
right after `group_by_modality` and stores the returned loss as
`self.last_afw_loss`, which `train.py` adds directly into the training loss
(`train.py:463-465`).

---

## 5. Ada2I integration: AMW (`train.py`, no separate module)

### What Ada2I's AMW does (paper)

Ada2I's Adaptive Modality Weighting normalizes modality-wise features
(L2-normalization inspired by NormFace/ring-loss, `ada2i.md` §3.4) and
supervises training with a **discrepancy ratio** `ρ_m` (a generalization of
Peng et al.'s OGM-GE two-modality ratio to three modalities), used to
modulate each modality's encoder gradient by `1 − tanh(α·ρ_m)` when `ρ_m > 1`
and `1` otherwise (Ada2I Eq. in §3.5, echoing OGM-GE's coefficient family).

### How this codebase implements it

AMW here is **not** a `nn.Module` — it is two plain functions in `train.py`
that operate directly on `.grad` tensors between `loss.backward()` and
`optimizer.step()`:

- **`get_modality_unimodal_ratio`** (`train.py:201-224`): exploits the fact
  that `smax_fc` is a linear layer over the concatenation of modality
  representations, so its weight matrix decomposes into per-modality column
  blocks `w_m`. For each present modality, it computes a "unimodal" logit
  `logit_m = feat_m @ w_m.T + bias/len(present)` from `model.last_modality_feat`
  (the AMW tap from §3.1 step 6), softmaxes it, and sums the true-label
  probability over the batch as that modality's `score_m`. The ratio is
  `ratio[m] = score_m / min_l score_l` — the same "ratio to the weakest
  modality" convention Ada2I uses, generalized to however many modalities are
  present (`--modals`).
- **`apply_amw_modulation`** (`train.py:227-257`): for each modality's encoder
  parameter group (from `get_modality_param_groups`, which enumerates
  `linear_a`/`gru_a`, `linear_v`/`gru_v`, `linear_l`/`gru_l` **by attribute
  name**, so it transparently picks up the PMR `mlp` audio/visual encoder
  variant too), multiplies `.grad` in place by:
  ```
  coeff = 1 - tanh(alpha * ratio[m])     if ratio[m] > 1   (Ada2I / OGM-GE-style brake)
  coeff = boost                          otherwise          (weakest modality)
  boost = 1 + mu * tanh(alpha * max(r_max - 1, 0))   if mu > 0 else 1.0
  ```
  With `--amw_mu 0` (the paper-faithful default) this is exactly Ada2I's
  rule: dominant modalities are throttled, the weakest is left untouched.
  `--amw_mu > 0` is **an addition beyond the paper**: it also actively boosts
  the weakest modality's gradient, scaled by how far the most-dominant
  modality has pulled ahead (`r_max - 1`), bounded so the boost saturates at
  `1 + mu` and vanishes to `1.0` when modalities are balanced. This is called
  out in the code comment (`train.py:246-248`) as *not* OGM-GE's "GE"
  (Gaussian-noise) step — M3Net has no 4-D conv kernels for that noise
  injection to apply to; it is a different, M3Net-specific boost mechanism
  built to occupy a similar role.
- **`--amw_ratio_source {classifier, prototype}`**: with `prototype`, the
  ratio driving `apply_amw_modulation` is swapped for PMR's own `ρ_m`
  (`pmr_stats['rho']`, computed at the encoder-output tap rather than the
  classifier) — see `train.py:515-516`. This is a genuinely new variant
  proposed in the implementation plan (§1.6.2): AMW's magnitude modulation
  driven by a fusion-independent imbalance estimate instead of a
  classifier-column one. Both ratios are always logged (`diag['amw_ratio']`
  and `diag['pmr_rho']`) regardless of which one drives the modulation, so
  their agreement is itself a diagnostic.
- **`--amw_start_epoch` / `--amw_end_epoch`** gate the epoch range over which
  modulation is applied (`train.py:517`), letting AMW be confined to (for
  instance) only the early epochs.

---

## 6. PMR integration (`pmr.py`)

### What PMR does (paper)

PMR (`PMR.md`) observes that magnitude-only gradient modulation (OGM-GE,
and by extension AMW) is limited because the *direction* of the fused
gradient — not just its magnitude — is disturbed by the dominant modality
(paper Fig. 2c). It proposes two mechanisms that act on per-modality
representations directly, bypassing the fusion stack:

- **PCE (Prototypical Cross-Entropy, Eq. 9)**: per-class **prototypes**
  (centroids of each modality's own representation space, Eq. 6) define a
  non-parametric classifier via a softmax over negative distances to
  prototypes (Eq. 7); PCE is the cross-entropy of that classifier. Because it
  only touches modality `m`'s own encoder, its gradient direction cannot be
  disturbed by other modalities — this is the paper's central claim.
- **PER (Prototypical Entropy Regularization, Eq. 13)**: during the first few
  epochs, the entropy of the dominant modality's prototype-softmax
  distribution is *maximized* (subtracted from the loss) to delay its
  premature convergence, reducing the suppression it exerts on weaker
  modalities.
- Both are weighted by coefficients `β` (accelerate) / `γ` (decelerate)
  derived from an imbalance ratio `ρ_t` (Eq. 8), computed and clipped per
  Eq. 11.

### Adaptations required for M3Net (`pmr.py` docstring, `pmr.py:1-34`)

1. **2 → N modalities.** PMR's Eq. 11 is written as a single pairwise ratio
   between exactly two modalities. `pmr_coefficients` (`pmr.py:204-234`)
   generalizes it:
   ```python
   rho_m   = s_m / min_l(s_l)                       # >= 1, weakest = 1.0
   beta_m  = clip(s_max/s_m - 1, 0, 1)               # PCE weight (accelerate)
   gamma_m = clip(s_m/s_min - 1, 0, 1)               # PER weight (decelerate)
   ```
   The docstring proves this reduces *exactly* to the paper's Eq. 11
   (including clip bounds) in the two-modality case — this reduction is
   covered by a dedicated unit test per the implementation plan (§5, test 3).
   A numerical subtlety fixed during implementation: dividing by
   `s_m + epsilon` (an additive epsilon) perturbs every coefficient away from
   the exact paper values; the code instead floors the divisor
   (`max(s_m, 1e-12)`) so the two-modality case reduces exactly (implementation
   plan, Appendix B, deviation 3).
2. **No padding/masking machinery needed.** PMR's formulation is
   per-instance. Because `simple_batch_graphify` already produces flat,
   unpadded, dialogue-major node lists (§3.1 step 3) that align row-for-row
   with the flattened label vector `train.py` builds, PMR's per-sample losses
   map directly onto M3Net's tensors with **no segment-mean pooling** of the
   kind `afw.py` needed (`pmr.py:27-33`; confirmed in the implementation
   plan's Appendix A.2). A `mask` argument exists throughout `pmr.py` for
   API generality but `train.py` always passes `None`.
3. **Batch-level noise.** PMR's original batches (CREMA-D/AVE/CG-MNIST,
   batch size 64 images) are far larger and more i.i.d. than an M3Net batch
   (a handful of whole dialogues — IEMOCAP batch size 16 ≈ hundreds of
   utterances, but drawn from only 16 independent conversations). `pmr_total`
   (`pmr.py:259-311`) smooths the raw per-batch confidence `s_m` with an EMA
   (`--pmr_ratio_ema`, default 0.9) before computing `β`/`γ`, and reports both
   raw and smoothed values in the diagnostics.

### Module structure (`pmr.py`)

- **`pairwise_distance`**: Euclidean (paper default), squared-Euclidean, or
  cosine distance between representations and prototypes — `--pmr_distance`
  and `--pmr_temp` extend beyond the paper's fixed Euclidean/`τ=1` setting,
  explicitly flagged as such (`--pmr_distance` help text; implementation
  plan §1.3) because raw high-dimensional Euclidean distances can saturate
  the softmax.
- **`PrototypeBank`** (`pmr.py:68-133`): holds one `[num_classes, D_m]`
  buffer per modality (registered buffers, not parameters — detached, as the
  paper requires: "Calculate the detached prototypes"). `update()` implements
  the EMA of Eq. 12, with the fix that classes absent from the current pass
  keep their old prototype rather than being zeroed. `logits()` returns
  `-d(z, proto)/τ` per modality (gradient flows through `z` only).
  `confidence()` computes the batch-mean true-label softmax probability —
  PMR's `s_m` from Eq. 8's numerator/denominator, generalized to per-modality
  scalars rather than a pairwise ratio directly.
- **`PrototypeAccumulator`** / **`compute_prototypes`** (`pmr.py:136-199`):
  a streaming per-class sum/count accumulator and a `no_grad`, eval-mode pass
  over the training loader (or a dialogue-level subsample,
  `--pmr_proto_subset`) that produces per-class centroids — the paper's "feed
  forward the subset, compute centroids" step (Algorithm 1). It takes a
  `batch_forward` callback rather than unpacking M3Net batches itself
  (`train.py`'s `make_pmr_batch_forward`, `train.py:351-357`, wraps the exact
  same `_forward_graph_batch` the training loop uses), so the prototype pass
  is structurally guaranteed to match the training forward pass and cannot
  drift out of sync with it (implementation plan, Appendix B, deviation 1).
- **`pce_loss`** / **`per_entropy`** (`pmr.py:239-256`): per-modality
  cross-entropy against the prototype logits (Eq. 9), and per-modality Shannon
  entropy of the prototype softmax (the `H` in Eq. 13).
- **`pmr_total`** (`pmr.py:259-311`): assembles
  `L_pmr = α · Σ_m β_m · L_PCE^m − μ · Σ_m γ_m · H_m` (the `apply_per`
  argument gates the entropy term, letting `train.py` turn PER off after
  `--pmr_reg_epochs`), and returns a `stats` dict logged verbatim into the
  diagnostics CSV.
- **`prototype_predictions`** (`pmr.py:316-323`): nearest-centroid
  predictions per modality — a non-parametric unimodal "probe" independent of
  any trained classifier, used to measure whether a weak modality's *own*
  representation actually improved (the paper's Fig. 4a/b evidence), as
  opposed to the fused head simply re-weighting what it already had.
  Controlled by `--pmr_probe`, independent of `--use_pmr`, so the baseline
  gets the same curve for comparison.
- **`gradient_angle`** (`pmr.py:326-351`): computes the angle between the
  task loss's gradient and the PCE loss's gradient on one modality's encoder
  parameters — a direct replication of the paper's Fig. 2c "direction
  interference" measurement inside this architecture. Gated behind
  `--pmr_log_angles` (costs an extra pair of backward passes per logged
  step, hence `--pmr_angle_every`).

### Where PMR taps into the model

Three tap points, selected by `--pmr_tap`:

| Tap | Where | What trains through PCE | Fidelity to the paper's "independent of other modalities" claim |
|---|---|---|---|
| `pre_graph` (default) | `Model.forward`, right after `simple_batch_graphify`, before `HyperGCN` | Only `linear_m` (+`gru_l` for text) | Highest — last point a modality is purely itself |
| `post_graph` | `HyperGCN.forward`, `E_dict` before AFW | Encoders + hypergraph/frequency conv stack | Lower — hyperedges have already mixed modalities |
| `post_afw` | `HyperGCN.forward`, `E_dict` after AFW | Everything up to AFW | Lowest of the three; useful for measuring whether AFW itself changes the imbalance PMR sees |

The known risk this table encodes (implementation plan §1.1, §8): at
`pre_graph`, the audio/visual "encoder" PCE can reshape is a single
`nn.Linear` on frozen features by default — very little capacity. The
`--pmr_av_encoder mlp` flag (§3.1) exists specifically to give PCE something
non-trivial to act on, and the plan mandates it always be run alongside a
**no-PMR MLP control**, since added capacity alone can move results
independent of PMR.

### Training-loop wiring (`train.py`)

Per epoch (`train.py:930-936`), if `--use_pmr` or `--pmr_probe` and
`epoch >= --pmr_start_epoch`: prototypes are recomputed from a fresh pass
over the training loader and blended into the bank via EMA. `--pmr_start_epoch
1` (default) means training runs one vanilla epoch before prototypes exist
(the paper computes them from epoch 0 against random-init ResNets; a warm
start avoids equally degenerate prototypes here).

Per training step (`train_or_eval_graph_model`, `train.py:419-561`):

```python
log_prob, label, lengths, aux = _forward_graph_batch(...)     # populates last_pmr_repr / last_modality_feat / last_afw_loss
loss = loss_function(log_prob, label)
if afw_loss is not None: loss = loss + afw_loss
if use_pmr and pmr_bank.initialized:
    pmr_loss, pmr_stats = pmr_total(pmr_bank, last_pmr_repr, label, alpha, mu,
                                     apply_per=(epoch < pmr_reg_epochs), s_ema=...)
    loss = loss + pmr_loss
loss.backward()
if use_amw:
    ratio = pmr_stats['rho'] if amw_ratio_source == 'prototype' else get_modality_unimodal_ratio(...)
    apply_amw_modulation(model, modals, ratio, amw_alpha, amw_mu)   # only within [amw_start_epoch, amw_end_epoch]
optimizer.step()
```

This ordering — PCE/PER folded into the scalar loss *before* `backward()`,
AMW rescaling the resulting `.grad` tensors *after* — means AMW's brake is
applied to the **combined** task+AFW+PCE+PER gradient on the dominant
encoder, matching the paper's own Algorithm 1 (which modulates a similarly
combined loss). This is a deliberate design choice documented in the
implementation plan (§1.6.3), not an accident of ordering.

### Diagnostics

`DIAGNOSTIC_COLUMNS` (`train.py:260-266`) and `write_diagnostics_row`
(`train.py:269-294`) append one CSV row per modality per epoch (behind
`--log_diagnostics`), including: encoder gradient norm, per-modality V/F
representation norms (from `HyperGCN._compute_modality_norms`,
`model_hyper.py:149-174`), AMW's classifier-based ratio, and — whenever PMR
is active — raw/smoothed `s_m`, `ρ_m`, `β_m`, `γ_m`, the PCE loss, the PER
entropy, the prototype-probe F1/accuracy, and per-class prototype sample
counts. AMW's ratio and PMR's `ρ_m` are always logged side by side
regardless of which one drives the modulation (`--amw_ratio_source`), so
their agreement or disagreement is itself diagnostic evidence.

---

## 7. Determinism (`deterministic_scatter.py`, `train.py:seed_everything`)

Both graph convolutions (`HypergraphConv.aggregate`, `high_fre_conv.py`'s
`highConv.aggregate`) override PyG's default `MessagePassing` aggregation,
which on CUDA dispatches to an atomic-add scatter whose result depends on
GPU thread-completion order — not fixed by any seed or by
`cudnn.deterministic`. `deterministic_scatter_add` (`deterministic_scatter.py`)
instead uses `Tensor.index_add_`, which has a deterministic CUDA kernel when
`torch.use_deterministic_algorithms(True)` is set. `train.py`'s
`seed_everything()` (`train.py:30-45`) sets all the standard seeds plus
`torch.use_deterministic_algorithms(True, warn_only=True)` — `warn_only` so
any remaining non-deterministic op degrades to a warning rather than a hard
crash, since not every op has a deterministic kernel. `CUBLAS_WORKSPACE_CONFIG`
is set at the top of `train.py` (`train.py:1-4`) because it must be set
before any CUDA context exists for deterministic CUBLAS matmuls to be
available at all.

---

## 8. CLI surface (`train.py` argparse)

### Backbone / M3Net

| Flag | Meaning |
|---|---|
| `--base-model` | `GRU` (used by the README commands) / `LSTM` / `Transformer` / `None` — per-modality sequence encoder. |
| `--graph_type` | `hyper` selects the `HyperGCN` backbone this document describes; other values reach legacy code. |
| `--mm_fusion_mthd` | `concat_DHT` selects the two-branch (hypergraph + frequency) concatenation classifier. |
| `--modals` | Subset of `{a, v, l}` to use, e.g. `avl`, `al`, `av`. |
| `--num_L` / `--num_K` | Number of stacked `HypergraphConv` / `highConv` layers (paper's `L` and `K`, swept in Fig. 3 of `M3net.md`). |
| `--norm` | RoBERTa-layer normalization strategy (`LN`/`BN`/`LN2`/other). |
| `--use_residue` | Whether `E_m` concatenates the raw input features alongside the V/F branches. |
| `--use_speaker` / `--use_modal` | Speaker / modality-type embeddings inside `HyperGCN`. |
| `--Dataset` | `IEMOCAP` or `MELD`. |

### Ada2I: AFW

`--use_afw`, `--afw_rank`, `--afw_beta`, `--afw_nlayers`, `--afw_dropout` —
see §4.

### Ada2I: AMW

`--use_amw`, `--amw_alpha`, `--amw_mu`, `--amw_start_epoch`,
`--amw_end_epoch`, `--amw_ratio_source {classifier,prototype}` — see §5.

### PMR

`--use_pmr`, `--pmr_tap {pre_graph,post_graph,post_afw}`, `--pmr_alpha`,
`--pmr_mu`, `--pmr_reg_epochs`, `--pmr_start_epoch`, `--pmr_proto_momentum`,
`--pmr_proto_subset`, `--pmr_distance {euclid,sq_euclid,cosine}`,
`--pmr_temp`, `--pmr_ratio_ema`, `--pmr_av_encoder {linear,mlp}`,
`--pmr_probe`, `--pmr_log_angles`, `--pmr_angle_every` — see §6.

### Diagnostics / checkpointing / experiment management

`--log_diagnostics`, `--diagnostics_path`, `--save_model`, `--save_path`,
`--save_optimizer`, `--multi_seed` (sweeps the `SEEDS` list at the top of
`train.py`), `--seed_sweep_path`, `--testing` (load `--save_path` and
evaluate once, `n_epochs=0`).

---

## 9. Loss composition, end to end

With every optional module enabled, the scalar minimized each step is:

```
L = L_task(log_prob, label)                                     # NLLLoss (IEMOCAP, class-weighted) or FocalLoss (MELD)
  + L_afw                                                        # Σ_layers L1(fc(x_m), aware_m), if --use_afw
  + α · Σ_m β_m · L_PCE^m                                       # if --use_pmr, β from prototype confidences
  − μ · Σ_m γ_m · H(softmax(−d(z_m, C_m)/τ))                    # if --use_pmr and epoch < --pmr_reg_epochs
```

`backward()` is called once on this sum. AMW then rescales the resulting
`.grad` on each modality's encoder parameters (§5) before `optimizer.step()`
(Adam, `--lr` / `--l2`). With every `--use_*` flag off, `L` reduces to plain
`L_task` and the model trains as unmodified M3Net — the implementation plan
records a byte-identical-output regression test verifying exactly this
(Appendix B, "Test results").

---

## 10. Cross-reference: paper equations → code

| Paper | Equation / concept | Code |
|---|---|---|
| M3Net §3.1 | Speaker embedding `S_i`, per-modality encoders (Eqs. 1–3) | `model.py` `linear_a/v/l`, `gru_l`; `model_hyper.py:180-182` speaker add |
| M3Net §3.2.1 | Hypergraph construction, weights `ω(e)`, `γ_e(v)` | `model_hyper.py:create_hyper_index` |
| M3Net §3.2.2 Eq. 5 | Hypergraph convolution | `HypergraphConv.py:forward`; called `num_L` times in `model_hyper.py:209-210` |
| M3Net §3.3.2–3.3.3 Eqs. 7–11 | Low/high-pass filters, self-gated coefficient `r_l−r_h` | `high_fre_conv.py:highConv`; called `num_K` times in `model_hyper.py:218-219` |
| M3Net Eq. 13 | Concatenated emotion representation `e_i` | `model_hyper.py:224-244` (`out2`, `E_dict`, final `out1` concat) |
| M3Net Eq. 14–15 | Classifier, CE + L2 loss | `model.py:920` (`smax_fc`+`log_softmax`); L2 via `--l2` in the Adam optimizer |
| Ada2I Eqs. 3–7 §3.3 | Tensor-ring Q/K, feature-aware attention, β-gated residual | `afw.py:M3NetTensorAttention` |
| Ada2I Eq. 11–12 §3.5 | Attention Mapping Network, `L_feature` | `afw.py:M3NetAttentionMapping` |
| Ada2I §3.4, discrepancy ratio | Modality-wise ratio, gradient coefficient `1−tanh(αρ)` | `train.py:get_modality_unimodal_ratio`, `apply_amw_modulation` |
| PMR Eq. 6 | Prototypes (class centroids) | `pmr.py:PrototypeAccumulator.centroids`, `compute_prototypes` |
| PMR Eq. 7 | Prototype-distance softmax classifier | `pmr.py:PrototypeBank.logits` |
| PMR Eq. 8 | Imbalance ratio `ρ_t` | `pmr.py:pmr_coefficients` (`rho`), generalized to N modalities |
| PMR Eq. 9–10 | PCE loss, acceleration loss | `pmr.py:pce_loss`, folded into `pmr_total` |
| PMR Eq. 11 | β/γ coefficient clipping rule | `pmr.py:pmr_coefficients` |
| PMR Eq. 12 | EMA prototype update | `pmr.py:PrototypeBank.update` |
| PMR Eq. 13 | PER entropy regularization | `pmr.py:per_entropy`, subtracted in `pmr_total` |
| PMR Algorithm 1 | Full per-epoch/per-step procedure | `train.py:924-961` (prototype refresh) + `train_or_eval_graph_model` (per-step loss/backward) |
| PMR Fig. 2c | Gradient-direction interference | `pmr.py:gradient_angle`, logged via `--pmr_log_angles` |
| PMR Fig. 4a/b | Per-modality unimodal probe | `pmr.py:prototype_predictions`, `--pmr_probe` |

---

## 11. Known deviations from the source papers (deliberate, documented in the implementation plan)

- **AFW** pools per-dialogue (flat graph batches) instead of per-sequence
  (padded batches) — a structural adaptation, not a simplification of the
  mechanism itself (§4).
- **AMW**'s `--amw_mu` weak-modality boost is an addition beyond Ada2I's
  published rule (§5); `amw_mu=0` recovers the paper-faithful behavior.
- **AMW**'s `--amw_ratio_source prototype` option (driving AMW from PMR's
  ratio) is a new variant not in either source paper, proposed in the
  implementation plan as a way to unify the two ratio estimates (§5, §1.6.2
  of the plan).
- **PMR**'s coefficient rule is generalized from 2 to N modalities (§6),
  exactly reducing to the paper's Eq. 11 in the 2-modality case.
- **PMR**'s `--pmr_distance`/`--pmr_temp` options extend beyond the paper's
  fixed Euclidean/`τ=1` setting (§6), to counteract softmax saturation in
  M3Net's higher-dimensional representations; the paper's own setting remains
  available and is the recommended fidelity run.
- **PMR** defaults to computing prototypes over the **full** training set
  each epoch (`--pmr_proto_subset 1.0`) rather than the paper's 10% subset,
  because IEMOCAP/MELD are small enough that a full `no_grad` pass costs
  seconds, not the paper's stated cost concern — subsampling is still
  available and exercises the paper's own ablation (`PMR.md` Table 5).
- **PMR**'s `--pmr_av_encoder mlp` (§3.1, §6) is a structural addition to
  give PCE non-trivial capacity on the audio/visual branches, required
  because those encoders are a single `nn.Linear` on frozen features by
  default — unlike the paper's trainable ResNet-18 encoders. It must be run
  with a no-PMR control per the implementation plan, since added capacity
  alone can move results.
- The optimizer is kept as **Adam** throughout (matching M3Net's and Ada2I's
  own choice) even though PMR's analysis assumes SGD; the PMR paper's own
  Table 6 reports PMR still helps under Adam, so this is treated as an
  acceptable mismatch rather than a required change (§1.7 of the
  implementation plan).

---

## 12. Legacy / dead code paths

- **`model_GCN.py`**: `GCNLayer1`, `GCN_2Layers`, `GCNII`, `GCNII_lyc` support
  `graph_type` values (`GCN3`, `DeepGCN`, `relation`) that none of the
  documented training commands select; `GCNII_lyc` is imported into
  `model_hyper.py` but the line that would instantiate it
  (`self.graph_net = GCNII_lyc(...)`) is commented out. `TextCNN` is reachable
  only via `--use_bert_seq` inside `Model`, which has no CLI flag and is
  hardcoded `False`.
- **`model.py`**'s `att_type == 'gated'` / `'concat_subsequently'` branches
  and `MMGatedAttention` are alternative fusion heads to `concat_DHT`; not
  used by the README's or `runner.ipynb`'s commands, which all pass
  `--mm_fusion_mthd concat_DHT`.
- **`graph_type in {'GCN3', 'DeepGCN'}`** branch of `Model.forward`
  (`model.py:887-915`) is the pre-hypergraph fusion path; reachable only if
  `--graph_type` is set away from its documented value.
