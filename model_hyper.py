import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.nn import RGCNConv, GraphConv, FAConv
from torch.nn import Parameter
import numpy as np, itertools, random, copy, math
import math
import scipy.sparse as sp
from model_GCN import GCNII_lyc
import ipdb
from HypergraphConv import HypergraphConv
from torch_geometric.nn import GCNConv
from itertools import permutations
from torch_geometric.nn.pool.select.topk import topk
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp, global_add_pool as gsp
from torch_geometric.nn.inits import glorot
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import add_self_loops
from high_fre_conv import highConv
from afw import M3NetAFW
class STEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        return F.hardtanh(grad_output)

class GraphConvolution(nn.Module):

    def __init__(self, in_features, out_features, residual=False, variant=False):
        super(GraphConvolution, self).__init__() 
        self.variant = variant
        if self.variant:
            self.in_features = 2*in_features 
        else:
            self.in_features = in_features

        self.out_features = out_features
        self.residual = residual
        self.weight = Parameter(torch.FloatTensor(self.in_features,self.out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, adj , h0 , lamda, alpha, l):
        theta = math.log(lamda/l+1)
        hi = torch.spmm(adj, input)
        if self.variant:
            support = torch.cat([hi,h0],1)
            r = (1-alpha)*hi+alpha*h0
        else:
            support = (1-alpha)*hi+alpha*h0
            r = support
        output = theta*torch.mm(support, self.weight)+(1-theta)*r
        if self.residual:
            output = output+input
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    def forward(self, x, dia_len):
        """
        x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        tmpx = torch.zeros(0).cuda()
        tmp = 0
        for i in dia_len:
            a = x[tmp:tmp+i].unsqueeze(1)
            a = a + self.pe[:a.size(0)]
            tmpx = torch.cat([tmpx,a], dim=0)
            tmp = tmp+i
        #x = x + self.pe[:x.size(0)]
        tmpx = tmpx.squeeze(1)
        return self.dropout(tmpx)

class HyperGCN(nn.Module):
    def __init__(self, a_dim, v_dim, l_dim, n_dim, nlayers, nhidden, nclass, dropout, lamda, alpha, variant, return_feature, use_residue,
                new_graph='full',n_speakers=2, modals=['a','v','l'], use_speaker=True, use_modal=False, num_L=3, num_K=4,
                use_afw=False, afw_rank=8, afw_beta=0.7, afw_nlayers=2, afw_dropout=0.1,
                pmr_tap=None):
        super(HyperGCN, self).__init__()
        self.return_feature = return_feature  #True
        self.use_residue = use_residue
        self.new_graph = new_graph

        #self.graph_net = GCNII_lyc(nfeat=n_dim, nlayers=nlayers, nhidden=nhidden, nclass=nclass,
        #                       dropout=dropout, lamda=lamda, alpha=alpha, variant=variant,
        #                       return_feature=return_feature, use_residue=use_residue)
        self.act_fn = nn.ReLU()
        self.dropout = dropout
        self.alpha = alpha
        self.lamda = lamda
        self.modals = modals
        self.modal_embeddings = nn.Embedding(3, n_dim)
        self.speaker_embeddings = nn.Embedding(n_speakers, n_dim)
        self.use_speaker = use_speaker
        self.use_modal = use_modal
        self.use_position = False
        #------------------------------------    
        self.fc1 = nn.Linear(n_dim, nhidden)      
        #self.fc2 = nn.Linear(n_dim, nhidden)     
        self.num_L =  num_L
        self.num_K =  num_K
        for ll in range(num_L):
            setattr(self,'hyperconv%d' %(ll+1), HypergraphConv(nhidden, nhidden))
        self.act_fn = nn.ReLU()
        self.hyperedge_weight = nn.Parameter(torch.ones(1000))
        self.EW_weight = nn.Parameter(torch.ones(5200))
        self.hyperedge_attr1 = nn.Parameter(torch.rand(nhidden))
        self.hyperedge_attr2 = nn.Parameter(torch.rand(nhidden))
        #nn.init.xavier_uniform_(self.hyperedge_attr1)
        for kk in range(num_K):
            setattr(self,'conv%d' %(kk+1), highConv(nhidden, nhidden))
        #self.conv = highConv(nhidden, nhidden)

        present = [m for m in ['l', 'a', 'v'] if m in modals]
        e_dim = (n_dim + 2 * nhidden) if use_residue else (2 * nhidden)
        self.use_afw = use_afw
        if use_afw:
            self.afw = M3NetAFW({m: e_dim for m in present}, afw_rank, present, afw_beta, afw_nlayers, afw_dropout)
        else:
            self.afw = None

        # PMR: which per-modality [N, D] tensor to expose for prototype/PCE use.
        # 'post_graph' = E_m before AFW, 'post_afw' = E_m after AFW. 'pre_graph'
        # is tapped in Model.forward instead (it is upstream of this module).
        self.pmr_tap = pmr_tap
        self.pmr_dim = e_dim

        self.last_repr_norms = None
        self.last_afw_loss = None
        self.last_modality_dims = None
        self.last_pmr_repr = None

    def _compute_modality_norms(self, out, gnn_out, dia_len):
        """Per-modality L2-norm stats of the V (hypergraph) and F (high-freq) node
        representations, split the same way group_by_modality splits them."""
        present = [m for m in ['l', 'a', 'v'] if m in self.modals]
        num_modality = len(present)
        groups_v = {m: [] for m in present}
        groups_f = {m: [] for m in present}
        o, g = out, gnn_out
        for i in dia_len:
            for idx, m in enumerate(present):
                groups_v[m].append(o[idx*i:(idx+1)*i])
                groups_f[m].append(g[idx*i:(idx+1)*i])
            o = o[num_modality*i:]
            g = g[num_modality*i:]

        stats = {}
        for m in present:
            v_norms = torch.cat(groups_v[m], dim=0).norm(dim=-1)
            f_norms = torch.cat(groups_f[m], dim=0).norm(dim=-1)
            stats[m] = {
                'V_norm_mean': v_norms.mean().item(),
                'V_norm_std': v_norms.std().item(),
                'F_norm_mean': f_norms.mean().item(),
                'F_norm_std': f_norms.std().item(),
            }
        return stats

    def forward(self, a, v, l, dia_len, qmask, epoch, collect_diagnostics=False):
        qmask = torch.cat([qmask[:x,i,:] for i,x in enumerate(dia_len)],dim=0)
        spk_idx = torch.argmax(qmask, dim=-1)
        spk_emb_vector = self.speaker_embeddings(spk_idx)
        if self.use_speaker:
            if 'l' in self.modals:
                l += spk_emb_vector
        if self.use_position:
            if 'l' in self.modals:
                l = self.l_pos(l, dia_len)
            if 'a' in self.modals:
                a = self.a_pos(a, dia_len)
            if 'v' in self.modals:
                v = self.v_pos(v, dia_len)
        if self.use_modal:  
            emb_idx = torch.LongTensor([0, 1, 2]).cuda()
            emb_vector = self.modal_embeddings(emb_idx)

            if 'a' in self.modals:
                a += emb_vector[0].reshape(1, -1).expand(a.shape[0], a.shape[1])
            if 'v' in self.modals:
                v += emb_vector[1].reshape(1, -1).expand(v.shape[0], v.shape[1])
            if 'l' in self.modals:
                l += emb_vector[2].reshape(1, -1).expand(l.shape[0], l.shape[1])


        hyperedge_index, edge_index, features, batch, hyperedge_type1 = self.create_hyper_index(a, v, l, dia_len, self.modals)
        x1 = self.fc1(features)  
        weight = self.hyperedge_weight[0:hyperedge_index[1].max().item()+1]
        EW_weight = self.EW_weight[0:hyperedge_index.size(1)]

        edge_attr = self.hyperedge_attr1*hyperedge_type1 + self.hyperedge_attr2*(1-hyperedge_type1)
        out = x1
        for ll in range(self.num_L):
            out = getattr(self,'hyperconv%d' %(ll+1))(out, hyperedge_index, weight, edge_attr, EW_weight, dia_len)             
        if self.use_residue:
            out1 = torch.cat([features, out], dim=-1)                                   
        #out1 = self.reverse_features(dia_len, out1)                                     

        #---------------------------------------
        gnn_edge_index, gnn_features = self.create_gnn_index(a, v, l, dia_len, self.modals)
        gnn_out = x1
        for kk in range(self.num_K):
            gnn_out = gnn_out + getattr(self,'conv%d' %(kk+1))(gnn_out,gnn_edge_index)

        if collect_diagnostics:
            self.last_repr_norms = self._compute_modality_norms(out, gnn_out, dia_len)

        out2 = torch.cat([out,gnn_out], dim=1)
        if self.use_residue:
            out2 = torch.cat([features, out2], dim=-1)

        present = [m for m in ['l', 'a', 'v'] if m in self.modals]
        E_dict = self.group_by_modality(dia_len, out2)

        if self.pmr_tap == 'post_graph':
            self.last_pmr_repr = {m: E_dict[m] for m in present}

        if self.afw is not None:
            E_dict, afw_loss = self.afw(E_dict, dia_len)
            self.last_afw_loss = afw_loss
        else:
            self.last_afw_loss = None

        if self.pmr_tap == 'post_afw':
            self.last_pmr_repr = {m: E_dict[m] for m in present}

        self.last_modality_dims = {m: E_dict[m].size(1) for m in present}
        out1 = torch.cat([E_dict[m] for m in present], dim=-1)
        #---------------------------------------
        return out1
        


    def create_hyper_index(self, a, v, l, dia_len, modals):
        self_loop = False
        num_modality = len(modals)
        node_count = 0
        edge_count = 0
        batch_count = 0
        index1 =[]
        index2 =[]
        tmp = []
        batch = []
        edge_type = torch.zeros(0).cuda()
        in_index0 = torch.zeros(0).cuda()
        hyperedge_type1 = []

        mod_tensors = {'l': l, 'a': a, 'v': v}
        present = [m for m in ['l', 'a', 'v'] if m in modals]
        assert len(present) == num_modality

        temp = 0
        for i in dia_len:
            nodes = list(range(i*num_modality))
            nodes = [j + node_count for j in nodes]
            nodes_by_mod = {m: nodes[idx*i:(idx+1)*i] for idx, m in enumerate(present)}

            for m in present:
                index1 = index1 + nodes_by_mod[m]
            for _ in range(i):
                index1 = index1 + [nodes_by_mod[m][_] for m in present]
            for _ in range(i+num_modality):
                if _ < num_modality:
                    index2 = index2 + [edge_count]*i
                else:
                    index2 = index2 + [edge_count]*num_modality
                edge_count = edge_count + 1

            feats_this = [mod_tensors[m][temp:temp+i] for m in present]
            features_temp = torch.cat(feats_this, dim=0)
            if node_count == 0:
                features = features_temp
            else:
                features = torch.cat([features, features_temp], dim=0)
            temp = temp+i

            Gnodes=[]
            for m in present:
                Gnodes.append(nodes_by_mod[m])
            for _ in range(i):
                Gnodes.append([nodes_by_mod[m][_] for m in present])
            for ii, _ in enumerate(Gnodes):
                perm = list(permutations(_,2))
                tmp = tmp + perm
            batch = batch + [batch_count]*i*num_modality
            batch_count = batch_count + 1
            hyperedge_type1 = hyperedge_type1 + [1]*i + [0]*num_modality

            node_count = node_count + i*num_modality

        index1 = torch.LongTensor(index1).view(1,-1)
        index2 = torch.LongTensor(index2).view(1,-1)
        hyperedge_index = torch.cat([index1,index2],dim=0).cuda() 
        if self_loop:
            max_edge = hyperedge_index[1].max()
            max_node = hyperedge_index[0].max()
            loops = torch.cat([torch.arange(0,max_node+1,1).repeat_interleave(2).view(1,-1),
                            torch.arange(max_edge+1,max_edge+1+max_node+1,1).repeat_interleave(2).view(1,-1)],dim=0).cuda()
            hyperedge_index = torch.cat([hyperedge_index, loops], dim=1)

        edge_index = torch.LongTensor(tmp).T.cuda()
        batch = torch.LongTensor(batch).cuda()

        hyperedge_type1 = torch.LongTensor(hyperedge_type1).view(-1,1).cuda()

        return hyperedge_index, edge_index, features, batch, hyperedge_type1

    def group_by_modality(self, dia_len, features):
        """Split flat, dialogue-major node features (one row per utterance per
        present modality) back into a per-modality dict {m: [N_utts, dim]}."""
        present = [m for m in ['l', 'a', 'v'] if m in self.modals]
        num_modality = len(present)
        groups = {m: [] for m in present}
        for i in dia_len:
            for idx, m in enumerate(present):
                groups[m].append(features[idx*i:(idx+1)*i])
            features = features[num_modality*i:]
        return {m: torch.cat(groups[m], dim=0) for m in present}


    def create_gnn_index(self, a, v, l, dia_len, modals):
        self_loop = False
        num_modality = len(modals)
        node_count = 0
        batch_count = 0
        index =[]
        tmp = []

        mod_tensors = {'l': l, 'a': a, 'v': v}
        present = [m for m in ['l', 'a', 'v'] if m in modals]

        temp = 0
        for i in dia_len:
            nodes = list(range(i*num_modality))
            nodes = [j + node_count for j in nodes]
            nodes_by_mod = {m: nodes[idx*i:(idx+1)*i] for idx, m in enumerate(present)}

            for m in present:
                index = index + list(permutations(nodes_by_mod[m],2))
            Gnodes=[]
            for _ in range(i):
                Gnodes.append([nodes_by_mod[m][_] for m in present])
            for ii, _ in enumerate(Gnodes):
                tmp = tmp +  list(permutations(_,2))

            feats_this = [mod_tensors[m][temp:temp+i] for m in present]
            features_temp = torch.cat(feats_this, dim=0)
            if node_count == 0:
                features = features_temp
            else:
                features = torch.cat([features, features_temp], dim=0)
            temp = temp+i
            node_count = node_count + i*num_modality
        edge_index = torch.cat([torch.LongTensor(index).T,torch.LongTensor(tmp).T],1).cuda()

        return edge_index, features
