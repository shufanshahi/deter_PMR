"""
M3Net-adapted Prototypical Modal Rebalance (PMR), after Fan et al., CVPR 2023
("PMR: Prototypical Modal Rebalance for Multimodal Learning").

PMR has two mechanisms:

  * PCE -- Prototypical Cross-Entropy (paper Eq. 9/10). A per-modality
    classification loss computed against non-parametric class prototypes
    (centroids of that modality's representations). Because it never passes
    through the fusion stack, its gradient direction is not disturbed by the
    dominant modality -- which is the paper's central argument (their Fig. 2c).
    It is weighted per modality so that only the *slow-learning* modalities are
    accelerated.

  * PER -- Prototypical Entropy Regularization (paper Eq. 13). The entropy of
    the prototype-distance distribution is *maximized* (subtracted from the
    loss) for the *dominant* modalities during the first few epochs only, to
    delay their premature convergence and reduce the suppression they exert.

Two adaptations were needed for M3Net:

  1. The paper is two-modality; its coefficient rules (Eq. 11) are written as a
     single pairwise ratio. `pmr_coefficients` below generalizes them to an
     arbitrary present-modality subset while reducing *exactly* to Eqs. (11)
     and (13) when only two modalities are present (see the docstring there).

  2. The paper operates on padded per-instance batches. M3Net's representations
     are flat, graph-batched, dialogue-major node lists `[N, D]` that are
     already aligned one-to-one with the flattened label vector and contain no
     padded rows -- so PMR's per-sample formulation maps onto them directly,
     with no segment/pooling machinery of the kind `afw.py` needed. A `mask`
     argument is still accepted throughout for safety, but the M3Net training
     loop passes `None`.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------- distances

def pairwise_distance(z, proto, distance='euclid'):
    """z: [N, D], proto: [C, D] -> [N, C] distances (larger = less similar)."""
    if distance == 'euclid':
        return torch.cdist(z, proto, p=2)
    if distance == 'sq_euclid':
        return torch.cdist(z, proto, p=2).pow(2)
    if distance == 'cosine':
        zn = F.normalize(z, dim=-1)
        pn = F.normalize(proto, dim=-1)
        return 1.0 - zn @ pn.t()
    raise ValueError('unknown pmr distance: {}'.format(distance))


def _masked_mean(x, mask):
    """x: [N] (or [N, ...] reduced already). mask: None or bool/float [N]."""
    if mask is None:
        return x.mean()
    mask = mask.to(x.dtype)
    denom = mask.sum().clamp(min=1.0)
    return (x * mask).sum() / denom


# ---------------------------------------------------------------- prototypes

class PrototypeBank(nn.Module):
    """Per-modality class centroids, held as registered buffers and refreshed
    once per epoch with an EMA (paper Eq. 12)."""

    def __init__(self, num_classes, dims, modalities, momentum=0.9,
                 distance='euclid', temp=1.0):
        super(PrototypeBank, self).__init__()
        self.num_classes = num_classes
        self.modalities = list(modalities)
        self.momentum = momentum
        self.distance = distance
        self.temp = temp
        for m in self.modalities:
            self.register_buffer('proto_{}'.format(m), torch.zeros(num_classes, dims[m]))
        self.register_buffer('_initialized', torch.zeros(1))
        # per-class sample counts from the most recent refresh (diagnostics)
        self.last_counts = {m: None for m in self.modalities}

    @property
    def initialized(self):
        return bool(self._initialized.item())

    def proto(self, m):
        return getattr(self, 'proto_{}'.format(m))

    @torch.no_grad()
    def update(self, centroids, counts=None):
        """centroids: {m: [C, D_m]}. Classes with zero samples in this pass keep
        their previous prototype."""
        first = not self.initialized
        for m in self.modalities:
            new = centroids[m].detach().to(self.proto(m).dtype).to(self.proto(m).device)
            if first:
                self.proto(m).copy_(new)
            else:
                seen = torch.ones(self.num_classes, 1, device=new.device)
                if counts is not None:
                    seen = (counts[m] > 0).float().unsqueeze(1).to(new.device)
                blended = self.momentum * self.proto(m) + (1.0 - self.momentum) * new
                self.proto(m).copy_(seen * blended + (1.0 - seen) * self.proto(m))
        if counts is not None:
            self.last_counts = {m: counts[m].detach().cpu() for m in self.modalities}
        self._initialized.fill_(1.0)

    def logits(self, z):
        """z: {m: [N, D_m]} -> {m: [N, C]} of -d(z, proto)/temp.

        Gradient flows through `z` only; prototypes are buffers (detached), as
        in the paper ("Calculate the detached prototypes").
        """
        out = {}
        for m in z:
            if m not in self.modalities:
                continue
            d = pairwise_distance(z[m], self.proto(m).detach(), self.distance)
            out[m] = -d / self.temp
        return out

    def confidence(self, logits, label, mask=None):
        """s_m: batch-mean softmax probability assigned to the true class."""
        s = {}
        for m, lg in logits.items():
            p = F.softmax(lg, dim=1)
            p_true = p.gather(1, label.view(-1, 1)).squeeze(1)
            s[m] = _masked_mean(p_true, mask)
        return s


class PrototypeAccumulator:
    """Streaming per-class sums/counts of representations, per modality."""

    def __init__(self, num_classes, dims, modalities, device):
        self.num_classes = num_classes
        self.modalities = list(modalities)
        self.sums = {m: torch.zeros(num_classes, dims[m], device=device) for m in self.modalities}
        self.counts = {m: torch.zeros(num_classes, device=device) for m in self.modalities}

    def add(self, z, label, mask=None):
        label = label.view(-1)
        if mask is not None:
            keep = mask.view(-1).bool()
            label = label[keep]
        ones = torch.ones(label.size(0), device=self.counts[self.modalities[0]].device)
        for m in self.modalities:
            zm = z[m].detach().float()
            if mask is not None:
                zm = zm[mask.view(-1).bool()]
            self.sums[m].index_add_(0, label, zm)
            self.counts[m].index_add_(0, label, ones)

    def centroids(self):
        cents = {}
        for m in self.modalities:
            denom = self.counts[m].clamp(min=1.0).unsqueeze(1)
            cents[m] = self.sums[m] / denom
        return cents, self.counts


@torch.no_grad()
def compute_prototypes(model, loader, batch_forward, num_classes, dims, modalities,
                       device, subset_ratio=1.0, seed=0):
    """One `no_grad` eval-mode pass over (a dialogue-level subsample of) the
    training loader, accumulating per-class centroids of `model.last_pmr_repr`.

    `batch_forward(data) -> (repr_dict, label)` is supplied by the caller so
    this file stays free of M3Net's batch-unpacking specifics.

    Subsampling is done over *batches* (each batch is a set of whole dialogues),
    not utterances, because the forward pass is graph-batched per dialogue.
    """
    was_training = model.training
    model.eval()
    acc = PrototypeAccumulator(num_classes, dims, modalities, device)

    n_batches = len(loader)
    keep = None
    if subset_ratio < 1.0:
        k = max(1, int(math.ceil(subset_ratio * n_batches)))
        g = torch.Generator().manual_seed(seed)
        keep = set(torch.randperm(n_batches, generator=g)[:k].tolist())

    for i, data in enumerate(loader):
        if keep is not None and i not in keep:
            continue
        z, label = batch_forward(data)
        if z is None:
            continue
        acc.add(z, label)

    if was_training:
        model.train()
    return acc.centroids()


# ------------------------------------------------------------- coefficients

def pmr_coefficients(s):
    """Generalize PMR's Eq. (11) coefficients from 2 to any number of modalities.

    Given per-modality true-class confidences `s = {m: float}`:

        rho_m   = s_m / min_l s_l            (>= 1; the weakest modality has 1.0)
        beta_m  = clip(s_max / s_m - 1, 0, 1)   PCE weight -- accelerate the weak
        gamma_m = clip(s_m / s_min - 1, 0, 1)   PER weight -- decelerate the strong

    Two-modality reduction (weak = 0, dominant = 1, paper's rho_t = s_0/s_1 < 1):
        beta_0  = s_1/s_0 - 1 = 1/rho_t - 1  -> paper's beta   (accelerates modality 0)
        gamma_0 = 0                          -> paper's PER coefficient on H^0
        beta_1  = 0                          -> paper's gamma == 0
        gamma_1 = s_1/s_0 - 1 = 1/rho_t - 1  -> paper's PER coefficient on H^1
    i.e. exactly Eqs. (11) and (13), including the clip bounds.
    """
    vals = {m: float(v) for m, v in s.items()}
    s_max = max(vals.values())
    s_min = min(vals.values())
    # Guard against a degenerate zero confidence by flooring the divisor rather
    # than adding an epsilon to it -- adding one perturbs every coefficient
    # (e.g. 0.45/0.4 - 1 would come out at 0.12499997 instead of exactly 0.125),
    # which would break the exact reduction to the paper's Eq. (11).
    tiny = 1e-12
    den_min = max(s_min, tiny)
    beta, gamma, rho = {}, {}, {}
    for m, v in vals.items():
        rho[m] = v / den_min
        beta[m] = min(max(s_max / max(v, tiny) - 1.0, 0.0), 1.0)
        gamma[m] = min(max(v / den_min - 1.0, 0.0), 1.0)
    return beta, gamma, rho


# ------------------------------------------------------------------- losses

def pce_loss(logits, label, mask=None):
    """Paper Eq. (9): -log softmax(-d(z, c))[y], per modality."""
    out = {}
    for m, lg in logits.items():
        ce = F.cross_entropy(lg, label.view(-1), reduction='none')
        out[m] = _masked_mean(ce, mask)
    return out


def per_entropy(logits, mask=None):
    """Entropy H of the prototype-distance softmax, per modality (paper Eq. 13)."""
    out = {}
    for m, lg in logits.items():
        logp = F.log_softmax(lg, dim=1)
        p = logp.exp()
        h = -(p * logp).sum(dim=1)
        out[m] = _masked_mean(h, mask)
    return out


def pmr_total(bank, z, label, alpha=1.0, mu=1e-2, apply_per=True, mask=None,
              s_ema=None, ratio_ema=0.9):
    """Assemble PMR's contribution to the training loss.

        L_pmr = alpha * sum_m beta_m * L_PCE^m  -  mu * sum_m gamma_m * H_m

    The minus sign is the paper's: subtracting entropy *maximizes* it, keeping
    the dominant modality's prototype distribution flat early in training.

    `s_ema` is an optional mutable dict carrying an EMA of the per-modality
    confidences across steps (M3Net batches are only a handful of dialogues, so
    the raw per-batch `s_m` is noisy). Coefficients are computed from the
    smoothed values; both are reported in the stats.

    Returns (loss_tensor, stats_dict).
    """
    logits = bank.logits(z)
    s_raw = bank.confidence(logits, label, mask)
    s_raw_f = {m: float(v.item()) for m, v in s_raw.items()}

    if s_ema is None:
        s_used = s_raw_f
    else:
        for m, v in s_raw_f.items():
            s_ema[m] = v if m not in s_ema else ratio_ema * s_ema[m] + (1.0 - ratio_ema) * v
        s_used = {m: s_ema[m] for m in s_raw_f}

    beta, gamma, rho = pmr_coefficients(s_used)

    pce = pce_loss(logits, label, mask)
    ent = per_entropy(logits, mask) if apply_per else None

    device = next(iter(logits.values())).device
    loss = torch.zeros((), device=device)
    for m in pce:
        if beta[m] > 0.0:
            loss = loss + alpha * beta[m] * pce[m]
    if apply_per and mu > 0.0:
        for m in ent:
            if gamma[m] > 0.0:
                loss = loss - mu * gamma[m] * ent[m]

    stats = {
        's_raw': s_raw_f,
        's': dict(s_used),
        'rho': rho,
        'beta': beta,
        'gamma': gamma,
        'pce': {m: float(v.item()) for m, v in pce.items()},
        'entropy': {m: float(v.item()) for m, v in ent.items()} if ent is not None else {},
        'pce_terms': pce,          # live tensors, for the optional angle diagnostic
    }
    return loss, stats


# --------------------------------------------------------------- probe utils

def prototype_predictions(bank, z):
    """Nearest-centroid predictions per modality -- the non-parametric unimodal
    probe used for the paper's Fig. 4a/b evidence."""
    preds = {}
    with torch.no_grad():
        for m, lg in bank.logits(z).items():
            preds[m] = lg.argmax(dim=1)
    return preds


def gradient_angle(task_loss, pce_term, params):
    """Angle (degrees) between the task loss's gradient and the PCE loss's
    gradient on one modality's encoder parameters -- a direct replication of
    PMR's Fig. 2c "direction interference" measurement in this architecture.

    Must be called *before* `loss.backward()`; uses `retain_graph=True` so the
    subsequent backward still works.
    """
    params = [p for p in params if p.requires_grad]
    if not params:
        return float('nan')
    g_task = torch.autograd.grad(task_loss, params, retain_graph=True, allow_unused=True)
    g_pce = torch.autograd.grad(pce_term, params, retain_graph=True, allow_unused=True)

    def flat(gs):
        parts = [g.reshape(-1) for g in gs if g is not None]
        return torch.cat(parts) if parts else None

    a, b = flat(g_task), flat(g_pce)
    if a is None or b is None:
        return float('nan')
    denom = (a.norm() * b.norm()).item()
    if denom < 1e-12:
        return float('nan')
    cos = float((a @ b).item()) / denom
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))
