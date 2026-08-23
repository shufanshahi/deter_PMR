"""
M3Net-adapted Adaptive Feature Weighting (AFW), inspired by Ada2I
(ACM-MM_ADA2I/model/AFW.py).

Ada2I's AFW assumes padded [batch, seq_len, dim] tensors and pools each
modality's tensor-attention summary with a plain `.mean(dim=1)` over the
sequence axis. M3Net has no such axis: HyperGCN produces a flat, graph-batched
list of per-utterance nodes (shape [N, dim], N = sum(dia_len)), with no
batch/length dimensions to mean over. This module reimplements the same
tensor-contraction attention idea, but replaces the sequence-mean with a
per-dialogue segment-mean (using dia_len boundaries) that is broadcast back
to each utterance in that dialogue -- the flat-node equivalent of "pool over
this conversation's utterances".
"""
import torch
import torch.nn as nn


def segment_mean_broadcast(x, dia_len):
    """x: [N, ...]. Replaces each node's value with the mean over its own
    dialogue (as delimited by dia_len), broadcast back to every node in
    that dialogue. Same shape in/out."""
    out = torch.empty_like(x)
    start = 0
    for ln in dia_len:
        seg = x[start:start + ln]
        seg_mean = seg.mean(dim=0, keepdim=True)
        out[start:start + ln] = seg_mean.expand_as(seg)
        start += ln
    return out


class M3NetTensorAttention(nn.Module):
    """Per-modality Khatri-Rao tensor attention (Ada2I-style), contracted
    against the other modalities' per-dialogue summaries."""

    def __init__(self, dims, rank, modalities, beta):
        super().__init__()
        self.modalities = modalities
        self.rank = rank
        self.beta = beta
        self.trans_q1 = nn.ModuleDict({m: nn.Linear(dims[m], rank) for m in modalities})
        self.trans_q2 = nn.ModuleDict({m: nn.Linear(dims[m], rank) for m in modalities})
        self.trans_k1 = nn.ModuleDict({m: nn.Linear(dims[m], rank) for m in modalities})
        self.trans_k2 = nn.ModuleDict({m: nn.Linear(dims[m], rank) for m in modalities})
        self.lin_att = nn.ModuleDict({m: nn.Linear(rank * rank, dims[m]) for m in modalities})

    def forward(self, x, dia_len):
        """x: dict {modal: [N, dim_m]}. Returns dict {modal: [N, dim_m]}."""
        G_qk = {}
        for m in self.modalities:
            G_q = self.trans_q1[m](x[m]).unsqueeze(-1) * self.trans_q2[m](x[m]).unsqueeze(-2)
            G_k = self.trans_k1[m](x[m]).unsqueeze(-1) * self.trans_k2[m](x[m]).unsqueeze(-2)
            G_qk[m] = G_q * G_k  # [N, rank, rank]

        M_bcast = {m: segment_mean_broadcast(G_qk[m], dia_len) for m in self.modalities}

        aware = {}
        for m in self.modalities:
            a = G_qk[m]
            for other in self.modalities:
                if other == m:
                    continue
                a = torch.einsum('nkl,nlo->nko', a, M_bcast[other])
            n = a.size(0)
            a = a.reshape(n, self.rank * self.rank)
            a = self.lin_att[m](a)
            aware[m] = a * x[m] + self.beta * x[m]
        return aware


class M3NetAttentionMapping(nn.Module):
    """Auxiliary consistency regularizer: a linear map from the raw modality
    input should be able to reconstruct the AFW output. Mirrors Ada2I's
    AttentionMapping L1 loss."""

    def __init__(self, dims, modalities):
        super().__init__()
        self.modalities = modalities
        self.fc = nn.ModuleDict({m: nn.Linear(dims[m], dims[m]) for m in modalities})
        self.loss = nn.L1Loss()

    def get_loss(self, aware, x):
        pred = torch.cat([self.fc[m](x[m]) for m in self.modalities], dim=1)
        target = torch.cat([aware[m] for m in self.modalities], dim=1)
        return self.loss(pred, target)


class M3NetAFWLayer(nn.Module):
    def __init__(self, dims, rank, modalities, beta, droprate=0.1):
        super().__init__()
        self.modalities = modalities
        self.attention = M3NetTensorAttention(dims, rank, modalities, beta)
        self.dropout = nn.Dropout(p=droprate)
        self.wmn = M3NetAttentionMapping(dims, modalities)

    def forward(self, x, dia_len):
        aware = self.attention(x, dia_len)
        aware = {m: self.dropout(aware[m]) for m in self.modalities}
        loss = self.wmn.get_loss(aware, x)
        return aware, loss


class M3NetAFW(nn.Module):
    """Stack of M3NetAFWLayer, dim-preserving so layers can be chained.
    Adapted from Ada2I's AFW (ACM-MM_ADA2I/model/AFW.py)."""

    def __init__(self, dims, rank, modalities, beta, nlayers, droprate=0.1):
        super().__init__()
        self.modalities = modalities
        self.layers = nn.ModuleList([
            M3NetAFWLayer(dims, rank, modalities, beta, droprate)
            for _ in range(nlayers)
        ])

    def forward(self, x, dia_len):
        out = x
        total_loss = None
        for layer in self.layers:
            out, loss = layer(out, dia_len)
            total_loss = loss if total_loss is None else total_loss + loss
        return out, total_loss
