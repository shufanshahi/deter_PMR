import os
# Must be set before any CUDA context is created for torch.use_deterministic_algorithms
# to be able to make CUBLAS matmul/Linear-backward ops deterministic on GPU.
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import csv
import math

import numpy as np, argparse, time, pickle, random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler
from dataloader import IEMOCAPDataset, MELDDataset
from model import MaskedNLLLoss, LSTMModel, GRUModel, Model, MaskedMSELoss, FocalLoss
from pmr import (PrototypeBank, compute_prototypes, pmr_total,
                 prototype_predictions, gradient_angle)
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report, precision_recall_fscore_support
import pandas as pd
import pickle as pk
import datetime
import statistics
import ipdb

SEEDS = [1475, 67137, 2024, 42, 12345]  # used when --multi_seed is passed; edit this list to change the sweep

seed = 1475 # We use seed = 1475 on IEMOCAP and seed = 67137 on MELD; overridden per-iteration by --multi_seed
def seed_everything(seed=None):
    if seed is None:
        seed = globals()['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # cudnn.deterministic only pins conv/RNN algorithm choice -- it does nothing
    # for the scatter-add aggregation HypergraphConv/highConv do every layer
    # (see deterministic_scatter.py). warn_only=True so any op genuinely
    # without a deterministic CUDA kernel degrades to a warning instead of
    # crashing training.
    torch.use_deterministic_algorithms(True, warn_only=True)

def _init_fn(worker_id):
        np.random.seed(int(seed)+worker_id)

def get_train_valid_sampler(trainset, valid=0.1, dataset='IEMOCAP'):
    size = len(trainset)
    idx = list(range(size))
    split = int(valid*size)
    return SubsetRandomSampler(idx[split:]), SubsetRandomSampler(idx[:split])


def get_MELD_loaders(batch_size=32, valid=0.1, num_workers=0, pin_memory=False):
    trainset = MELDDataset('MELD_features/MELD_features_raw1.pkl')
    train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid, 'MELD')

    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    valid_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    testset = MELDDataset('MELD_features/MELD_features_raw1.pkl', train=False)
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)

    return train_loader, valid_loader, test_loader


def get_IEMOCAP_loaders(batch_size=32, valid=0.1, num_workers=0, pin_memory=False):
    trainset = IEMOCAPDataset()
    train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid)

    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory, worker_init_fn=_init_fn)

    valid_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    testset = IEMOCAPDataset(train=False)
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory, worker_init_fn=_init_fn)

    return train_loader, valid_loader, test_loader


def train_or_eval_model(model, loss_function, dataloader, epoch, optimizer=None, train=False):
    """
    
    """
    losses, preds, labels, masks = [], [], [], []
    alphas, alphas_f, alphas_b, vids = [], [], [], []
    max_sequence_len = []

    assert not train or optimizer!=None
    if train:
        model.train()
    else:
        model.eval()

    seed_everything()
    for data in dataloader:
        if train:
            optimizer.zero_grad()
        
        textf, visuf, acouf, qmask, umask, label = [d.cuda() for d in data[:-1]] if cuda else data[:-1]        

        max_sequence_len.append(textf.size(0))
        
        log_prob, alpha, alpha_f, alpha_b, _ = model(textf, qmask, umask)
        lp_ = log_prob.transpose(0,1).contiguous().view(-1, log_prob.size()[2])
        labels_ = label.view(-1)
        loss = loss_function(lp_, labels_, umask)

        pred_ = torch.argmax(lp_,1)
        preds.append(pred_.data.cpu().numpy())
        labels.append(labels_.data.cpu().numpy())
        masks.append(umask.view(-1).cpu().numpy())

        losses.append(loss.item()*masks[-1].sum())
        if train:
            loss.backward()
            if args.tensorboard:
                for param in model.named_parameters():
                    writer.add_histogram(param[0], param[1].grad, epoch)
            optimizer.step()
        else:
            alphas += alpha
            alphas_f += alpha_f
            alphas_b += alpha_b
            vids += data[-1]

    if preds!=[]:
        preds  = np.concatenate(preds)
        labels = np.concatenate(labels)
        masks  = np.concatenate(masks)
    else:
        return float('nan'), float('nan'), [], [], [], float('nan'),[]

    avg_loss = round(np.sum(losses)/np.sum(masks), 4)
    avg_accuracy = round(accuracy_score(labels,preds, sample_weight=masks)*100, 2)
    avg_fscore = round(f1_score(labels,preds, sample_weight=masks, average='weighted')*100, 2)
    
    return avg_loss, avg_accuracy, labels, preds, masks, avg_fscore, [alphas, alphas_f, alphas_b, vids]


def get_modality_param_groups(model, modals):
    """Encoder parameters belonging to each modality, for gradient-norm diagnostics."""
    groups = {}
    if 'a' in modals and hasattr(model, 'linear_a'):
        params = list(model.linear_a.parameters())
        if hasattr(model, 'gru_a'):
            params += list(model.gru_a.parameters())
        groups['a'] = params
    if 'v' in modals and hasattr(model, 'linear_v'):
        params = list(model.linear_v.parameters())
        if hasattr(model, 'gru_v'):
            params += list(model.gru_v.parameters())
        groups['v'] = params
    if 'l' in modals and hasattr(model, 'linear_l'):
        params = list(model.linear_l.parameters())
        if hasattr(model, 'gru_l'):
            params += list(model.gru_l.parameters())
        groups['l'] = params
    return groups


def param_grad_norm(params):
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += p.grad.data.norm(2).item() ** 2
    return total ** 0.5


def get_modality_unimodal_ratio(model, modals, label):
    """Ada2I-style AMW: project each modality's own (already-fused) feature slice
    through its columns of the shared smax_fc classifier to get a per-modality
    "unimodal" logit, then measure how confidently each modality alone scores the
    true label. ratio[m] = modality m's total confidence / weakest modality's.
    Requires model.last_modality_feat, set by Model.forward's hyper branch."""
    present = [m for m in ['l', 'a', 'v'] if m in modals]
    feat = model.last_modality_feat
    weight = model.smax_fc.weight
    bias = model.smax_fc.bias

    scores = {}
    offset = 0
    for m in present:
        d = feat[m].size(1)
        w_m = weight[:, offset:offset + d]
        logit_m = feat[m] @ w_m.t() + bias / len(present)
        prob_m = F.softmax(logit_m, dim=1)
        scores[m] = prob_m[torch.arange(label.size(0), device=label.device), label].sum()
        offset += d

    min_score = min(scores.values())
    ratio = {m: (scores[m] / (min_score + 1e-8)).item() for m in present}
    return ratio


def apply_amw_modulation(model, modals, ratio, alpha, mu=0.0):
    """Shrink the gradient flowing into a modality's encoder in proportion to how
    much it dominates (ratio > 1) the weakest modality.

    With mu == 0 this is plain AMW: dominant modalities are throttled, the
    weakest is left untouched (coeff 1.0).

    With mu > 0 the weakest modality is additionally *boosted*, by how far the
    most dominant modality has pulled ahead of it:

        coeff_weak = 1 + mu * tanh(alpha * (r_max - 1))

    r_max is the largest ratio, so the boost vanishes smoothly to 1.0 when the
    modalities are balanced (r_max == 1) and saturates at (1 + mu) when one
    modality runs away -- the boost is bounded, it cannot blow the gradient up.
    Centering on (r_max - 1) rather than r_max is deliberate: it makes "no
    imbalance" mean "no intervention", instead of applying a constant boost
    whenever AMW is on at all.

    Note this is not OGM-GE's "GE": that adds Gaussian noise to 4-D conv-kernel
    gradients (see OGM-GE_CVPR2022/main.py), which M3Net has none of. This
    boosts the weak modality's own gradient directly."""
    groups = get_modality_param_groups(model, modals)
    r_max = max([ratio.get(m, 1.0) for m in groups], default=1.0)
    boost = 1.0 + mu * math.tanh(alpha * max(r_max - 1.0, 0.0)) if mu > 0.0 else 1.0
    for m, params in groups.items():
        r = ratio.get(m, 1.0)
        coeff = 1.0 - math.tanh(alpha * max(r, 0.0)) if r > 1.0 else boost
        for p in params:
            if p.grad is not None:
                p.grad.mul_(coeff)


DIAGNOSTIC_COLUMNS = [
    'grad_norm', 'V_norm_mean', 'F_norm_mean', 'amw_ratio',
    # PMR (all NaN unless --use_pmr / --pmr_probe is on)
    'pmr_s_raw', 'pmr_s_ema', 'pmr_rho', 'pmr_beta', 'pmr_gamma',
    'pmr_pce', 'pmr_entropy', 'pmr_grad_angle',
    'pmr_probe_f1', 'pmr_probe_acc', 'pmr_proto_count_min',
]


def write_diagnostics_row(path, epoch, modals, diag, probe=None):
    """Append one row per modality with the epoch-averaged diagnostics.

    Columns: encoder grad norm, V/F representation norms, AMW's classifier-based
    unimodal-confidence ratio, and -- when PMR is active -- the prototype-based
    confidence s_m (raw and EMA-smoothed), rho/beta/gamma, the PCE loss and PER
    entropy, the PCE-vs-task gradient angle, and the nearest-centroid probe
    metrics. AMW's ratio and PMR's rho are logged side by side regardless of
    which one drives the modulation, so their agreement is itself a diagnostic.
    """
    file_exists = os.path.exists(path)
    probe = probe or {}
    with open(path, 'a', newline='') as f:
        writer_ = csv.writer(f)
        if not file_exists:
            writer_.writerow(['epoch', 'modality'] + DIAGNOSTIC_COLUMNS)
        for m in modals:
            row = [epoch, m]
            for col in DIAGNOSTIC_COLUMNS:
                if col == 'pmr_probe_f1':
                    row.append(probe.get('f1', {}).get(m, float('nan')))
                elif col == 'pmr_probe_acc':
                    row.append(probe.get('acc', {}).get(m, float('nan')))
                else:
                    row.append(diag.get(col, {}).get(m, float('nan')))
            writer_.writerow(row)


def _forward_graph_batch(model, data, modals, cuda, epoch=None, collect_diagnostics=False):
    """Unpack one collated batch, run the model, and return
    (log_prob, label, lengths, aux).

    `label` is the flattened, dialogue-major, unpadded label vector -- built the
    same way `simple_batch_graphify` builds the flat node list, so it is aligned
    row-for-row with `model.last_pmr_repr[m]` and with `log_prob`.

    Factored out of the training loop so the PMR prototype pass runs the exact
    same forward, rather than a second copy that could drift out of sync.
    """
    textf1, textf2, textf3, textf4, visuf, acouf, qmask, umask, label = \
        [d.cuda() for d in data[:-1]] if cuda else data[:-1]

    textf = None
    if args.multi_modal:
        if args.mm_fusion_mthd == 'concat':
            if modals == 'avl':
                textf = torch.cat([acouf, visuf, textf1, textf2, textf3, textf4], dim=-1)
            elif modals == 'av':
                textf = torch.cat([acouf, visuf], dim=-1)
            elif modals == 'vl':
                textf = torch.cat([visuf, textf1, textf2, textf3, textf4], dim=-1)
            elif modals == 'al':
                textf = torch.cat([acouf, textf1, textf2, textf3, textf4], dim=-1)
            else:
                raise NotImplementedError
    else:
        if modals == 'a':
            textf = acouf
        elif modals == 'v':
            textf = visuf
        elif modals == 'l':
            textf = textf1
        else:
            raise NotImplementedError

    lengths = [(umask[j] == 1).nonzero(as_tuple=False).tolist()[-1][0] + 1 for j in range(len(umask))]

    if args.multi_modal and args.mm_fusion_mthd == 'gated':
        out = model(textf, qmask, umask, lengths, acouf, visuf)
    elif args.multi_modal and args.mm_fusion_mthd == 'concat_subsequently':
        out = model([textf1, textf2, textf3, textf4], qmask, umask, lengths, acouf, visuf, epoch)
    elif args.multi_modal and args.mm_fusion_mthd == 'concat_DHT':
        out = model([textf1, textf2, textf3, textf4], qmask, umask, lengths, acouf, visuf, epoch,
                    collect_diagnostics=collect_diagnostics)
    else:
        out = model(textf, qmask, umask, lengths)

    log_prob, e_i, e_n, e_t, e_l = out
    label = torch.cat([label[j][:lengths[j]] for j in range(len(label))])
    return log_prob, label, lengths, (e_i, e_n, e_t, e_l)


def make_pmr_batch_forward(model, modals, cuda):
    """Callback handed to pmr.compute_prototypes: run one batch and hand back
    the tapped per-modality representations plus their aligned labels."""
    def _fn(data):
        _, label, _, _ = _forward_graph_batch(model, data, modals, cuda)
        return getattr(model, 'last_pmr_repr', None), label
    return _fn


def resolve_checkpoint_path(args, run_seed):
    """Per-run checkpoint path. The seed is appended under --multi_seed so a
    sweep does not overwrite itself; otherwise --save_path is used verbatim, so
    the default stays the "best_model.pth.tar" name that --testing loads."""
    path = args.save_path
    if args.multi_seed:
        stem, ext = os.path.splitext(path)
        if stem.endswith('.pth'):          # keep ".pth.tar" intact
            stem, ext = os.path.splitext(stem)[0], '.pth' + ext
        path = '{}_seed{}{}'.format(stem, run_seed, ext)
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    return path


def save_checkpoint(path, model, optimizer, epoch, test_fscore, test_acc, args,
                    run_seed, pmr_bank=None):
    """Write the best-so-far model.

    Note on selection: train.py picks its "best" epoch by *test* F1 (the
    pre-existing convention in this codebase), so this checkpoint is the one
    whose score is reported at the end of the run -- it is not an independently
    held-out selection.

    `args` and the PMR prototype buffers are stored alongside the weights so a
    checkpoint can be reloaded without having to remember the flags it was
    trained with. The bank is kept as a separate entry rather than a Model
    submodule (see the implementation plan, Appendix A.5).
    """
    payload = {
        'state_dict': model.state_dict(),
        'epoch': epoch,
        'test_fscore': test_fscore,
        'test_acc': test_acc,
        'seed': run_seed,
        'args': vars(args),
    }
    if pmr_bank is not None and pmr_bank.initialized:
        payload['pmr_bank'] = pmr_bank.state_dict()
    # Adam's state roughly doubles the file, and train.py has no resume path, so
    # it is opt-in rather than default.
    if args.save_optimizer and optimizer is not None:
        payload['optimizer'] = optimizer.state_dict()
    torch.save(payload, path)


def load_checkpoint(path, model, map_location=None):
    """Load either the new checkpoint dict or a bare state_dict written by an
    older run. Returns the checkpoint metadata (empty for the bare form)."""
    state = torch.load(path, map_location=map_location, weights_only=False)
    if isinstance(state, dict) and 'state_dict' in state:
        model.load_state_dict(state['state_dict'])
        return {k: v for k, v in state.items()
                if k not in ('state_dict', 'optimizer', 'pmr_bank')}
    model.load_state_dict(state)
    return {}


def train_or_eval_graph_model(model, loss_function, dataloader, epoch, cuda, modals, optimizer=None, train=False, dataset='IEMOCAP', log_diagnostics=False,
                               use_amw=False, amw_alpha=0.5, amw_start_epoch=0, amw_end_epoch=10**9, amw_mu=0.0,
                               pmr_bank=None, use_pmr=False, pmr_alpha=1.0, pmr_mu=1e-2, pmr_apply_per=False,
                               pmr_probe=False, pmr_s_ema=None, pmr_ratio_ema=0.9,
                               amw_ratio_source='classifier', pmr_log_angles=False, pmr_angle_every=50):
    losses, preds, labels = [], [], []
    scores, vids = [], []
    diag_keys = ['grad_norm', 'V_norm_mean', 'F_norm_mean', 'amw_ratio',
                 'pmr_s_raw', 'pmr_s_ema', 'pmr_rho', 'pmr_beta', 'pmr_gamma',
                 'pmr_pce', 'pmr_entropy', 'pmr_grad_angle', 'pmr_proto_count_min']
    diag = {key: {m: [] for m in modals} for key in diag_keys}

    # Nearest-centroid unimodal probe (paper Fig. 4a/b). Runs independently of
    # --use_pmr so the baseline gets the same curve.
    probe_active = pmr_probe and pmr_bank is not None and pmr_bank.initialized
    probe_preds = {m: [] for m in pmr_bank.modalities} if probe_active else {}
    probe_labels = []

    ei, et, en, el = torch.empty(0).type(torch.LongTensor), torch.empty(0).type(torch.LongTensor), torch.empty(0), []

    if cuda:
        ei, et, en = ei.cuda(), et.cuda(), en.cuda()

    assert not train or optimizer!=None
    if train:
        model.train()
    else:
        model.eval()

    seed_everything()
    for step, data in enumerate(dataloader):
        if train:
            optimizer.zero_grad()

        log_prob, label, lengths, (e_i, e_n, e_t, e_l) = _forward_graph_batch(
            model, data, modals, cuda, epoch, collect_diagnostics=log_diagnostics)

        if log_diagnostics and args.mm_fusion_mthd == 'concat_DHT' \
                and getattr(model, 'graph_model', None) is not None and model.graph_model.last_repr_norms:
            for m, stat in model.graph_model.last_repr_norms.items():
                diag['V_norm_mean'][m].append(stat['V_norm_mean'])
                diag['F_norm_mean'][m].append(stat['F_norm_mean'])

        loss = loss_function(log_prob, label)
        afw_loss = getattr(getattr(model, 'graph_model', None), 'last_afw_loss', None)
        if afw_loss is not None:
            loss = loss + afw_loss

        z = getattr(model, 'last_pmr_repr', None)

        # ---- PMR: PCE acceleration + PER deceleration, added to the loss so
        # they go through backward() (AMW's rescaling then applies to the
        # combined gradient, matching the paper's Algorithm 1).
        pmr_stats = None
        if train and use_pmr and pmr_bank is not None and pmr_bank.initialized and z:
            task_loss = loss
            pmr_loss, pmr_stats = pmr_total(pmr_bank, z, label,
                                            alpha=pmr_alpha, mu=pmr_mu,
                                            apply_per=pmr_apply_per,
                                            s_ema=pmr_s_ema, ratio_ema=pmr_ratio_ema)
            if pmr_log_angles and step % max(1, pmr_angle_every) == 0:
                for m, params in get_modality_param_groups(model, modals).items():
                    if m in pmr_stats['pce_terms']:
                        diag['pmr_grad_angle'][m].append(
                            gradient_angle(task_loss, pmr_stats['pce_terms'][m], params))
            loss = loss + pmr_loss

            if log_diagnostics:
                for src, dst in (('s_raw', 'pmr_s_raw'), ('s', 'pmr_s_ema'), ('rho', 'pmr_rho'),
                                 ('beta', 'pmr_beta'), ('gamma', 'pmr_gamma'),
                                 ('pce', 'pmr_pce'), ('entropy', 'pmr_entropy')):
                    for m, v in pmr_stats[src].items():
                        if m in diag[dst]:
                            diag[dst][m].append(v)

        if probe_active and z:
            for m, p in prototype_predictions(pmr_bank, z).items():
                probe_preds[m].append(p.cpu().numpy())
            probe_labels.append(label.cpu().numpy())

        preds.append(torch.argmax(log_prob, 1).cpu().numpy())
        labels.append(label.cpu().numpy())
        losses.append(loss.item())
        if train:
            loss.backward()
            if log_diagnostics:
                for m, params in get_modality_param_groups(model, modals).items():
                    diag['grad_norm'][m].append(param_grad_norm(params))
            if use_amw and getattr(model, 'last_modality_feat', None) is not None:
                # AMW's own classifier-column ratio is always computed, so that it
                # can be logged next to PMR's prototype ratio even when the latter
                # is the one driving the modulation.
                ratio = get_modality_unimodal_ratio(model, modals, label)
                if log_diagnostics:
                    for m, r in ratio.items():
                        diag['amw_ratio'][m].append(r)
                if amw_ratio_source == 'prototype' and pmr_stats is not None:
                    ratio = pmr_stats['rho']
                if amw_start_epoch <= epoch <= amw_end_epoch:
                    apply_amw_modulation(model, modals, ratio, amw_alpha, amw_mu)
            optimizer.step()

    diag_summary = {
        key: {m: (float(np.mean(v)) if len(v) > 0 else float('nan')) for m, v in diag[key].items()}
        for key in diag
    }

    if pmr_bank is not None and pmr_bank.last_counts.get(pmr_bank.modalities[0]) is not None:
        for m in pmr_bank.modalities:
            if m in diag_summary['pmr_proto_count_min']:
                diag_summary['pmr_proto_count_min'][m] = float(pmr_bank.last_counts[m].min().item())

    probe_summary = {'f1': {}, 'acc': {}}
    if probe_active and probe_labels:
        probe_y = np.concatenate(probe_labels)
        for m, chunks in probe_preds.items():
            if not chunks:
                continue
            probe_p = np.concatenate(chunks)
            probe_summary['f1'][m] = round(f1_score(probe_y, probe_p, average='weighted') * 100, 2)
            probe_summary['acc'][m] = round(accuracy_score(probe_y, probe_p) * 100, 2)
    diag_summary['_probe'] = probe_summary

    if preds!=[]:
        preds  = np.concatenate(preds)
        labels = np.concatenate(labels)
    else:
        return float('nan'), float('nan'), [], [], float('nan'), [], [], [], [], [], diag_summary

    vids += data[-1]
    ei = ei.data.cpu().numpy()
    et = et.data.cpu().numpy()
    en = en.data.cpu().numpy()
    el = np.array(el)
    labels = np.array(labels)
    preds = np.array(preds)
    vids = np.array(vids)

    avg_loss = round(np.sum(losses)/len(losses), 4)
    avg_accuracy = round(accuracy_score(labels, preds)*100, 2)
    avg_fscore = round(f1_score(labels,preds, average='weighted')*100, 2)

    return avg_loss, avg_accuracy, labels, preds, avg_fscore, vids, ei, et, en, el, diag_summary


if __name__ == '__main__':
    path = './saved/IEMOCAP/'

    parser = argparse.ArgumentParser()

    parser.add_argument('--no-cuda', action='store_true', default=False, help='does not use GPU')

    parser.add_argument('--base-model', default='LSTM', help='base recurrent model, must be one of DialogRNN/LSTM/GRU')

    parser.add_argument('--graph-model', action='store_true', default=True, help='whether to use graph model after recurrent encoding')

    parser.add_argument('--nodal-attention', action='store_true', default=True, help='whether to use nodal attention in graph model: Equation 4,5,6 in Paper')

    parser.add_argument('--windowp', type=int, default=10, help='context window size for constructing edges in graph model for past utterances')

    parser.add_argument('--windowf', type=int, default=10, help='context window size for constructing edges in graph model for future utterances')

    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='learning rate')
    
    parser.add_argument('--l2', type=float, default=0.00003, metavar='L2', help='L2 regularization weight')
    
    parser.add_argument('--rec-dropout', type=float, default=0.1, metavar='rec_dropout', help='rec_dropout rate')
    
    parser.add_argument('--dropout', type=float, default=0.5, metavar='dropout', help='dropout rate')
    
    parser.add_argument('--batch-size', type=int, default=32, metavar='BS', help='batch size')
    
    parser.add_argument('--epochs', type=int, default=60, metavar='E', help='number of epochs')
    
    parser.add_argument('--class-weight', action='store_true', default=True, help='use class weights')
    
    parser.add_argument('--active-listener', action='store_true', default=False, help='active listener')
    
    parser.add_argument('--attention', default='general', help='Attention type in DialogRNN model')
    
    parser.add_argument('--tensorboard', action='store_true', default=False, help='Enables tensorboard log')

    parser.add_argument('--graph_type', default='relation', help='relation/GCN3/DeepGCN/MMGCN/MMGCN2')

    parser.add_argument('--use_topic', action='store_true', default=False, help='whether to use topic information')

    parser.add_argument('--alpha', type=float, default=0.2, help='alpha')

    parser.add_argument('--multiheads', type=int, default=6, help='multiheads')

    parser.add_argument('--graph_construct', default='full', help='single/window/fc for MMGCN2; direct/full for others')

    parser.add_argument('--use_gcn', action='store_true', default=False, help='whether to combine spectral and none-spectral methods or not')

    parser.add_argument('--use_residue', action='store_true', default=False, help='whether to use residue information or not')

    parser.add_argument('--multi_modal', action='store_true', default=False, help='whether to use multimodal information')

    parser.add_argument('--mm_fusion_mthd', default='concat', help='method to use multimodal information: concat, gated, concat_subsequently')

    parser.add_argument('--modals', default='avl', help='modals to fusion')

    parser.add_argument('--av_using_lstm', action='store_true', default=False, help='whether to use lstm in acoustic and visual modality')

    parser.add_argument('--Deep_GCN_nlayers', type=int, default=4, help='Deep_GCN_nlayers')

    parser.add_argument('--Dataset', default='IEMOCAP', help='dataset to train and test')

    parser.add_argument('--use_speaker', action='store_true', default=True, help='whether to use speaker embedding')

    parser.add_argument('--use_modal', action='store_true', default=False, help='whether to use modal embedding')

    parser.add_argument('--norm', default='LN2', help='NORM type')

    parser.add_argument('--testing', action='store_true', default=False, help='testing')

    parser.add_argument('--num_L', type=int, default=3, help='num_hyperconvs')

    parser.add_argument('--num_K', type=int, default=4, help='num_convs')

    parser.add_argument('--log_diagnostics', action='store_true', default=False, help='log per-modality V/F representation norms and encoder gradient norms (modality-imbalance diagnostics)')

    parser.add_argument('--diagnostics_path', default='diagnostics.csv', help='CSV path to append per-epoch modality-imbalance diagnostics to')

    parser.add_argument('--use_afw', action='store_true', default=False, help='enable Ada2I-inspired Adaptive Feature Weighting on the per-modality V+F representations')

    parser.add_argument('--afw_rank', type=int, default=8, help='AFW tensor-attention rank')

    parser.add_argument('--afw_beta', type=float, default=0.7, help='AFW residual-gate coefficient')

    parser.add_argument('--afw_nlayers', type=int, default=2, help='number of stacked AFW layers')

    parser.add_argument('--afw_dropout', type=float, default=0.1, help='AFW dropout rate')

    parser.add_argument('--use_amw', action='store_true', default=False, help='enable Ada2I-inspired Adaptive Modality Weighting (gradient modulation for dominant modalities)')

    parser.add_argument('--amw_alpha', type=float, default=0.5, help='AMW gradient-suppression strength for dominant modalities')

    parser.add_argument('--amw_mu', type=float, default=0.0, help='AMW gradient-enhancement strength for the weakest modality: its gradient is scaled by 1 + mu*tanh(alpha*(r_max-1)), bounded by 1+mu. 0 (default) = suppression only, identical to plain AMW')

    parser.add_argument('--amw_start_epoch', type=int, default=0, help='first epoch (0-indexed) at which AMW modulation is applied')

    parser.add_argument('--amw_end_epoch', type=int, default=10**9, help='last epoch at which AMW modulation is applied')

    # ---------------- PMR: Prototypical Modal Rebalance (Fan et al., CVPR 2023)

    parser.add_argument('--use_pmr', action='store_true', default=False, help='enable PMR: prototypical cross-entropy (PCE) acceleration of slow-learning modalities, plus prototypical entropy regularization (PER) of dominant ones')

    parser.add_argument('--pmr_tap', default='pre_graph', choices=['pre_graph', 'post_graph', 'post_afw'], help="which per-modality representation the prototypes/PCE act on: 'pre_graph' = encoder output (last point a modality is purely itself, PMR-faithful), 'post_graph' = E_m before AFW, 'post_afw' = E_m after AFW")

    parser.add_argument('--pmr_alpha', type=float, default=1.0, help='PCE loss weight alpha (paper Eq. 10)')

    parser.add_argument('--pmr_mu', type=float, default=1e-2, help='PER entropy-regularization weight mu (paper Eq. 13)')

    parser.add_argument('--pmr_reg_epochs', type=int, default=5, help='PER is applied only while epoch < this (paper applies it in the first few epochs only)')

    parser.add_argument('--pmr_start_epoch', type=int, default=1, help='first epoch at which prototypes are computed and PMR losses become active (1 = after one vanilla warm-up epoch; 0 = paper-faithful)')

    parser.add_argument('--pmr_proto_momentum', type=float, default=0.9, help='EMA momentum epsilon for the prototype update (paper Eq. 12)')

    parser.add_argument('--pmr_proto_subset', type=float, default=1.0, help='fraction of training batches (whole dialogues) used to recompute prototypes each epoch; 1.0 = full training set')

    parser.add_argument('--pmr_distance', default='euclid', choices=['euclid', 'sq_euclid', 'cosine'], help='distance used in the prototype softmax (paper uses plain Euclidean)')

    parser.add_argument('--pmr_temp', type=float, default=1.0, help='temperature tau on the prototype logits (-d/tau); paper uses 1.0')

    parser.add_argument('--pmr_ratio_ema', type=float, default=0.9, help='EMA on the per-modality confidence s_m before computing beta/gamma (M3Net batches are only a few dialogues, so raw s_m is noisy)')

    parser.add_argument('--pmr_av_encoder', default='linear', choices=['linear', 'mlp'], help="audio/visual encoder: 'linear' = the M3Net baseline single nn.Linear, 'mlp' = 2-layer MLP so PCE has capacity to reshape (run the no-PMR mlp control alongside)")

    parser.add_argument('--pmr_probe', action='store_true', default=False, help='log per-modality nearest-centroid (prototype classifier) F1/accuracy each epoch; independent of --use_pmr so the baseline gets the same curve')

    parser.add_argument('--pmr_log_angles', action='store_true', default=False, help='log the angle between the PCE and task gradients on each modality encoder (replicates PMR Fig. 2c); costs extra backward passes')

    parser.add_argument('--pmr_angle_every', type=int, default=50, help='log the PCE/task gradient angle every k training steps')

    parser.add_argument('--amw_ratio_source', default='classifier', choices=['classifier', 'prototype'], help="what drives AMW's gradient modulation: 'classifier' = Ada2I's smax_fc column slicing, 'prototype' = PMR's fusion-independent rho_m. Both are logged either way")

    parser.add_argument('--save_model', action='store_true', default=False, help='save the best-scoring model to --save_path (weights, optimizer state, the args it was trained with, and the PMR prototype buffers)')

    parser.add_argument('--save_path', default='best_model.pth.tar', help='checkpoint path written when --save_model is set; also the file --testing loads')

    parser.add_argument('--save_optimizer', action='store_true', default=False, help='also store the optimizer state in the checkpoint (roughly doubles its size; only useful if you add a resume path)')

    parser.add_argument('--multi_seed', action='store_true', default=False, help='run full training once per seed in the SEEDS list (see top of train.py) instead of a single run')

    parser.add_argument('--seed_sweep_path', default='seed_sweep_results.csv', help='CSV path to write the per-seed F1/accuracy summary to when --multi_seed is set')

    args = parser.parse_args()
    today = datetime.datetime.now()
    print(args)
    if args.av_using_lstm:
        name_ = args.mm_fusion_mthd+'_'+args.modals+'_'+args.graph_type+'_'+args.graph_construct+'using_lstm_'+args.Dataset
    else:
        name_ = args.mm_fusion_mthd+'_'+args.modals+'_'+args.graph_type+'_'+args.graph_construct+str(args.Deep_GCN_nlayers)+'_'+args.Dataset

    if args.use_speaker:
        name_ = name_+'_speaker'
    if args.use_modal:
        name_ = name_+'_modal'

    args.cuda = torch.cuda.is_available() and not args.no_cuda
    if args.cuda:
        print('Running on GPU')
    else:
        print('Running on CPU')

    if args.tensorboard:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter()

    cuda       = args.cuda
    n_epochs   = args.epochs
    batch_size = args.batch_size
    modals = args.modals
    feat2dim = {'IS10':1582,'3DCNN':512,'textCNN':100,'bert':768,'denseface':342,'MELD_text':600,'MELD_audio':300}
    D_audio = feat2dim['IS10'] if args.Dataset=='IEMOCAP' else feat2dim['MELD_audio']
    D_visual = feat2dim['denseface']
    D_text = 1024 #feat2dim['textCNN'] if args.Dataset=='IEMOCAP' else feat2dim['MELD_text']

    if args.multi_modal:
        if args.mm_fusion_mthd=='concat':
            if modals == 'avl':
                D_m = D_audio+D_visual+D_text
            elif modals == 'av':
                D_m = D_audio+D_visual
            elif modals == 'al':
                D_m = D_audio+D_text
            elif modals == 'vl':
                D_m = D_visual+D_text
            else:
                raise NotImplementedError
        else:
            D_m = 1024
    else:
        if modals == 'a':
            D_m = D_audio
        elif modals == 'v':
            D_m = D_visual
        elif modals == 'l':
            D_m = D_text
        else:
            raise NotImplementedError
    D_g = 512 if args.Dataset=='IEMOCAP' else 1024
    D_p = 150
    D_e = 100
    D_h = 100
    D_a = 100
    graph_h = 512
    n_speakers = 9 if args.Dataset=='MELD' else 2
    n_classes  = 7 if args.Dataset=='MELD' else 6 if args.Dataset=='IEMOCAP' else 1

    if args.Dataset == 'MELD':
        train_loader, valid_loader, test_loader = get_MELD_loaders(valid=0.0,
                                                                    batch_size=batch_size,
                                                                    num_workers=2)
    elif args.Dataset == 'IEMOCAP':
        train_loader, valid_loader, test_loader = get_IEMOCAP_loaders(valid=0.0,
                                                                      batch_size=batch_size,
                                                                      num_workers=2)
    else:
        print("There is no such dataset")

    seeds_to_run = SEEDS if args.multi_seed else [seed]
    sweep_results = []

    for run_seed in seeds_to_run:
        seed = run_seed
        if args.multi_seed:
            print('\n' + '='*20 + ' SEED {} '.format(run_seed) + '='*20)

        if args.graph_model:
            seed_everything()

            model = Model(args.base_model,
                                     D_m, D_g, D_p, D_e, D_h, D_a, graph_h,
                                     n_speakers=n_speakers,
                                     max_seq_len=200,
                                     window_past=args.windowp,
                                     window_future=args.windowf,
                                     n_classes=n_classes,
                                     listener_state=args.active_listener,
                                     context_attention=args.attention,
                                     dropout=args.dropout,
                                     nodal_attention=args.nodal_attention,
                                     no_cuda=args.no_cuda,
                                     graph_type=args.graph_type,
                                     use_topic=args.use_topic,
                                     alpha=args.alpha,
                                     multiheads=args.multiheads,
                                     graph_construct=args.graph_construct,
                                     use_GCN=args.use_gcn,
                                     use_residue=args.use_residue,
                                     D_m_v = D_visual,
                                     D_m_a = D_audio,
                                     modals=args.modals,
                                     att_type=args.mm_fusion_mthd,
                                     av_using_lstm=args.av_using_lstm,
                                     Deep_GCN_nlayers=args.Deep_GCN_nlayers,
                                     dataset=args.Dataset,
                                     use_speaker=args.use_speaker,
                                     use_modal=args.use_modal,
                                     norm = args.norm,
                                     num_L = args.num_L,
                                     num_K = args.num_K,
                                     use_afw=args.use_afw,
                                     afw_rank=args.afw_rank,
                                     afw_beta=args.afw_beta,
                                     afw_nlayers=args.afw_nlayers,
                                     afw_dropout=args.afw_dropout,
                                     pmr_tap=(args.pmr_tap if (args.use_pmr or args.pmr_probe) else None),
                                     pmr_av_encoder=args.pmr_av_encoder)

            print ('Graph NN with', args.base_model, 'as base model.')
            name = 'Graph'

        else:
            if args.base_model == 'GRU':
                model = GRUModel(D_m, D_e, D_h,
                                  n_classes=n_classes,
                                  dropout=args.dropout)

                print ('Basic GRU Model.')


            elif args.base_model == 'LSTM':
                model = LSTMModel(D_m, D_e, D_h,
                                  n_classes=n_classes,
                                  dropout=args.dropout)

                print ('Basic LSTM Model.')

            else:
                print ('Base model must be one of DialogRNN/LSTM/GRU/Transformer')
                raise NotImplementedError

            name = 'Base'

        if cuda:
            model.cuda()

        if args.Dataset == 'IEMOCAP':
            loss_weights = torch.FloatTensor([1/0.086747,
                                            1/0.144406,
                                            1/0.227883,
                                            1/0.160585,
                                            1/0.127711,
                                            1/0.252668])

        if args.Dataset == 'MELD':
            loss_function = FocalLoss()
        else:
            if args.class_weight:
                if args.graph_model:
                    #loss_function = FocalLoss()
                    loss_function  = nn.NLLLoss(loss_weights.cuda() if cuda else loss_weights)
                else:
                    loss_function  = MaskedNLLLoss(loss_weights.cuda() if cuda else loss_weights)
            else:
                if args.graph_model:
                    loss_function = nn.NLLLoss()
                else:
                    loss_function = MaskedNLLLoss()

        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

        lr = args.lr

        # ---------------- PMR prototype bank
        # Built whenever PMR losses or the prototype probe are requested. It holds
        # no trainable parameters (class centroids are registered buffers and are
        # detached), so it is deliberately not part of model.state_dict() -- that
        # keeps existing checkpoints loadable. Prototypes are recomputed every
        # epoch anyway.
        pmr_bank = None
        pmr_s_ema = {}
        if args.use_pmr or args.pmr_probe:
            assert model.pmr_dims is not None, (
                'PMR requires the hypergraph model (--graph_type hyper) with a tap point set')
            pmr_bank = PrototypeBank(n_classes, model.pmr_dims, sorted(model.pmr_dims.keys()),
                                     momentum=args.pmr_proto_momentum,
                                     distance=args.pmr_distance, temp=args.pmr_temp)
            if cuda:
                pmr_bank = pmr_bank.cuda()
            print('PMR prototype bank: tap={}, dims={}, distance={}, temp={}'.format(
                args.pmr_tap, model.pmr_dims, args.pmr_distance, args.pmr_temp))

        best_fscore, best_loss, best_label, best_pred, best_mask = None, None, None, None, None
        all_fscore, all_acc, all_loss = [], [], []

        checkpoint_path = resolve_checkpoint_path(args, run_seed)
        if args.save_model:
            print('Checkpoints (best test F1) will be written to', checkpoint_path)

        if args.testing:
            meta = load_checkpoint(args.save_path, model,
                                   map_location=None if cuda else 'cpu')
            if meta:
                print('testing loaded model from {} (epoch {}, test_fscore {})'.format(
                    args.save_path, meta.get('epoch'), meta.get('test_fscore')))
            else:
                print('testing loaded model from {}'.format(args.save_path))
            test_loss, test_acc, test_label, test_pred, test_fscore, _, _, _, _, _, _ = train_or_eval_graph_model(model, loss_function, test_loader, 0, cuda, args.modals, dataset=args.Dataset)
            print('test_acc:',test_acc,'test_fscore:',test_fscore)

        for e in range(n_epochs):
            start_time = time.time()

            if args.graph_model:
                # Refresh the class prototypes once per epoch from a no_grad,
                # eval-mode pass over the training set (paper Algorithm 1).
                if pmr_bank is not None and e >= args.pmr_start_epoch:
                    centroids, counts = compute_prototypes(
                        model, train_loader, make_pmr_batch_forward(model, args.modals, cuda),
                        n_classes, model.pmr_dims, pmr_bank.modalities,
                        device=('cuda' if cuda else 'cpu'),
                        subset_ratio=args.pmr_proto_subset, seed=run_seed + e)
                    pmr_bank.update(centroids, counts)

                pmr_kwargs = dict(pmr_bank=pmr_bank, use_pmr=args.use_pmr,
                                  pmr_alpha=args.pmr_alpha, pmr_mu=args.pmr_mu,
                                  pmr_apply_per=(e < args.pmr_reg_epochs),
                                  pmr_probe=args.pmr_probe, pmr_s_ema=pmr_s_ema,
                                  pmr_ratio_ema=args.pmr_ratio_ema,
                                  amw_ratio_source=args.amw_ratio_source,
                                  pmr_log_angles=args.pmr_log_angles,
                                  pmr_angle_every=args.pmr_angle_every)
                eval_pmr_kwargs = dict(pmr_bank=pmr_bank, pmr_probe=args.pmr_probe)

                train_loss, train_acc, _, _, train_fscore, _, _, _, _, _, train_diag = train_or_eval_graph_model(model, loss_function, train_loader, e, cuda, args.modals, optimizer, True, dataset=args.Dataset, log_diagnostics=args.log_diagnostics,
                                                                                                                    use_amw=args.use_amw, amw_alpha=args.amw_alpha, amw_start_epoch=args.amw_start_epoch, amw_end_epoch=args.amw_end_epoch, amw_mu=args.amw_mu, **pmr_kwargs)
                valid_loss, valid_acc, _, _, valid_fscore, _, _, _, _, _, _ = train_or_eval_graph_model(model, loss_function, valid_loader, e, cuda, args.modals, dataset=args.Dataset, **eval_pmr_kwargs)
                test_loss, test_acc, test_label, test_pred, test_fscore, _, _, _, _, _, test_diag = train_or_eval_graph_model(model, loss_function, test_loader, e, cuda, args.modals, dataset=args.Dataset, **eval_pmr_kwargs)
                all_fscore.append(test_fscore)

                if args.pmr_probe and test_diag.get('_probe', {}).get('f1'):
                    print('  prototype probe (test): ' + '  '.join(
                        '{}: F1={} acc={}'.format(m, test_diag['_probe']['f1'][m], test_diag['_probe']['acc'][m])
                        for m in sorted(test_diag['_probe']['f1'])))

                if args.log_diagnostics:
                    write_diagnostics_row(args.diagnostics_path, e, args.modals, train_diag,
                                          probe=test_diag.get('_probe'))


            else:
                train_loss, train_acc, _, _, _, train_fscore, _ = train_or_eval_model(model, loss_function, train_loader, e, optimizer, True)
                valid_loss, valid_acc, _, _, _, valid_fscore, _ = train_or_eval_model(model, loss_function, valid_loader, e)
                test_loss, test_acc, test_label, test_pred, test_mask, test_fscore, attentions = train_or_eval_model(model, loss_function, test_loader, e)
                all_fscore.append(test_fscore)

            if best_loss == None or best_loss > test_loss:
                best_loss, best_label, best_pred = test_loss, test_label, test_pred

            if best_fscore == None or best_fscore < test_fscore:
                best_fscore = test_fscore
                best_label, best_pred = test_label, test_pred
                #test_loss, test_acc, test_label, test_pred, test_fscore, _, _, _, _, _ = train_or_eval_graph_model(model, loss_function, test_loader, e, cuda, args.modals, dataset=args.Dataset)
                if args.save_model:
                    save_checkpoint(checkpoint_path, model, optimizer, e,
                                    test_fscore, test_acc, args, run_seed, pmr_bank)
                    print('  saved checkpoint (test_fscore {}) -> {}'.format(
                        test_fscore, checkpoint_path))

            if args.tensorboard:
                writer.add_scalar('test: accuracy', test_acc, e)
                writer.add_scalar('test: fscore', test_fscore, e)
                writer.add_scalar('train: accuracy', train_acc, e)
                writer.add_scalar('train: fscore', train_fscore, e)

            print('epoch: {}, train_loss: {}, train_acc: {}, train_fscore: {}, test_loss: {}, test_acc: {}, test_fscore: {}, time: {} sec'.\
                    format(e+1, train_loss, train_acc, train_fscore, test_loss, test_acc, test_fscore, round(time.time()-start_time, 2)))
            if (e+1)%10 == 0:
                print ('----------best F-Score:', max(all_fscore))
                print(classification_report(best_label, best_pred, sample_weight=best_mask,digits=4))
                print(confusion_matrix(best_label,best_pred,sample_weight=best_mask))

        if args.tensorboard:
            writer.close()

        if not args.testing:
            best_acc = accuracy_score(best_label, best_pred, sample_weight=best_mask) * 100
            print('Test performance..')
            print ('F-Score:', max(all_fscore))
            print ('Accuracy:', round(best_acc, 2))
            if not os.path.exists("record_{}_{}_{}.pk".format(today.year, today.month, today.day)):
                with open("record_{}_{}_{}.pk".format(today.year, today.month, today.day),'wb') as f:
                    pk.dump({}, f)
            with open("record_{}_{}_{}.pk".format(today.year, today.month, today.day), 'rb') as f:
                record = pk.load(f)
            key_ = name_ + ('_seed{}'.format(run_seed) if args.multi_seed else '')
            if record.get(key_, False):
                record[key_].append(max(all_fscore))
            else:
                record[key_] = [max(all_fscore)]
            if record.get(key_+'record', False):
                record[key_+'record'].append(classification_report(best_label, best_pred, sample_weight=best_mask,digits=4))
            else:
                record[key_+'record'] = [classification_report(best_label, best_pred, sample_weight=best_mask,digits=4)]
            with open("record_{}_{}_{}.pk".format(today.year, today.month, today.day),'wb') as f:
                pk.dump(record, f)

            print(classification_report(best_label, best_pred, sample_weight=best_mask,digits=4))
            print(confusion_matrix(best_label,best_pred,sample_weight=best_mask))

            if args.save_model and os.path.exists(checkpoint_path):
                size_mib = os.path.getsize(checkpoint_path) / 1024 ** 2
                print('Best model saved to {} ({:.1f} MiB, F1 {})'.format(
                    checkpoint_path, size_mib, max(all_fscore)))

            sweep_results.append({'seed': run_seed, 'f1': max(all_fscore), 'accuracy': round(best_acc, 2)})

    if args.multi_seed and sweep_results:
        fscores = [r['f1'] for r in sweep_results]
        accs = [r['accuracy'] for r in sweep_results]

        print('\n' + '='*20 + ' SEED SWEEP SUMMARY ' + '='*20)
        for r in sweep_results:
            print('  seed={:>6}   F1={:.2f}   acc={:.2f}'.format(r['seed'], r['f1'], r['accuracy']))
        print('F1:  mean={:.2f}  stdev={:.2f}  min={:.2f}  max={:.2f}'.format(
            statistics.mean(fscores), statistics.pstdev(fscores), min(fscores), max(fscores)))
        print('Acc: mean={:.2f}  stdev={:.2f}  min={:.2f}  max={:.2f}'.format(
            statistics.mean(accs), statistics.pstdev(accs), min(accs), max(accs)))

        with open(args.seed_sweep_path, 'w', newline='') as f:
            sweep_writer = csv.writer(f)
            sweep_writer.writerow(['seed', 'f1', 'accuracy'])
            for r in sweep_results:
                sweep_writer.writerow([r['seed'], r['f1'], r['accuracy']])
        print('Seed-sweep summary written to {}'.format(args.seed_sweep_path))
