import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from ..utils import *
from daakg.sampling import typed_sampling


class Decoder(nn.Module):
    def __init__(self, name, params):
        super(Decoder, self).__init__()
        self.print_name = name
        if name.startswith("[") and name.endswith("]"):
            self.name = name[1:-1]
        else:
            self.name = name

        p = 1 if params["train_dist"] == "manhattan" else 2
        transe_sp = True if params["train_dist"] == "normalize_manhattan" else False
        self.feat_drop = params["feat_drop"]
        self.k = params["k"]
        self.alpha = params["alpha"]
        self.margin = params["margin"]
        self.boot = params["boot"]

        if self.name == "mtranse_align":
            self.func = Alignment(p=p, dim=params["dim"])
        elif self.name == "transe":
            self.func = TransE(p=p, feat_drop=self.feat_drop, transe_sp=transe_sp)
        elif self.name == "rotate":
            self.func = RotatE(p=p, feat_drop=self.feat_drop, dim=params["dim"], params=self.margin)
        else:
            raise NotImplementedError("bad decoder name: " + self.name)
        
        if params["sampling"] == "T":
            # self.sampling_method = multi_typed_sampling
            self.sampling_method = typed_sampling
        elif params["sampling"] == "N":
            self.sampling_method = nearest_neighbor_sampling
        elif params["sampling"] == "R":
            self.sampling_method = random_sampling
        elif params["sampling"] == ".":
            self.sampling_method = None
        # elif params["sampling"] == "SLEF-DESIGN":
        #     self.sampling_method = SLEF-DESIGN_sampling
        else:
            raise NotImplementedError("bad sampling method: " + self.sampling_method)

        if hasattr(self.func, "loss"):
            self.loss = self.func.loss
        else:
            self.loss = nn.MarginRankingLoss(margin=self.margin)
        if hasattr(self.func, "mapping"):
            self.mapping = self.func.mapping

    def forward(self, ins_emb, rel_emb, sample):
        if type(ins_emb) == tuple:
            ins_emb, weight = ins_emb
            rel_emb_ = torch.matmul(rel_emb, weight)
        else:
            rel_emb_ = rel_emb
        func = self.func if self.sampling_method else self.func.only_pos_loss
        if self.name in ["align", "mtranse_align"]:
            return func(ins_emb[sample[:, 0]], ins_emb[sample[:, 1]])
        elif self.name == "n_r_align":
            nei_emb, ins_emb = ins_emb, rel_emb
            return func(ins_emb[sample[:, 0]], ins_emb[sample[:, 1]], nei_emb[sample[:, 0]], nei_emb[sample[:, 1]])
        # elif self.name == "SLEF-DESIGN":
        #     '''SLEF-DESIGN: special decoder forward'''
        else:
            return func(ins_emb[sample[:, 0]], rel_emb_[sample[:, 1]], ins_emb[sample[:, 2]])

    def __repr__(self):
        return '{}(name={}): {}'.format(self.__class__.__name__, self.print_name, self.func.__repr__())


class Alignment(nn.Module):
    def __init__(self, p, dim, orth=False):
        super(Alignment, self).__init__()
        self.p = p
        self.weight = nn.Parameter(torch.Tensor(dim, dim))
        self.orth = orth
        if self.orth:
            nn.init.orthogonal_(self.weight)
            self.I = nn.Parameter(torch.eye(dim), requires_grad=False)

    def forward(self, e1, e2):
        return - torch.norm(torch.matmul(e1, self.weight) - e2, p=self.p, dim=1)
    
    def mapping(self, emb):
        return torch.matmul(emb, self.weight)

    def only_pos_loss(self, e1, e2):
        if self.p == 1:
            map_loss = torch.sum(torch.abs(torch.matmul(e1, self.weight) - e2), dim=1).sum()
        else:
            map_loss = torch.sum(torch.pow(torch.matmul(e1, self.weight) - e2, 2), dim=1).sum()
        orthogonal_loss = torch.pow(torch.matmul(self.weight, self.weight.t()) - self.I, 2).sum(dim=1).sum(dim=0)
        return map_loss + orthogonal_loss

    def __repr__(self):
        return '{}(mode={})'.format(self.__class__.__name__, self.mode)


class TransE(nn.Module):
    def __init__(self, p, feat_drop, transe_sp=False):
        super(TransE, self).__init__()
        self.p = p
        self.feat_drop = feat_drop
        self.transe_sp = transe_sp

    def forward(self, e1, r, e2):
        e1 = F.dropout(e1, p=self.feat_drop, training=self.training)
        if self.transe_sp:
            pred = - F.normalize(e1 + r - e2, p=2, dim=1).sum(dim=1)
        else:
            pred = - torch.norm(e1 + r - e2, p=self.p, dim=1)    
        return pred
    
    def only_pos_loss(self, e1, r, e2):
        e1 = F.dropout(e1, p=self.feat_drop, training=self.training)
        if self.p == 1:
            return torch.sum(torch.abs(e1 + r - e2), dim=1).sum()
        else:
            return torch.sum(torch.pow(e1 + r - e2, 2), dim=1).sum()
    
    def uncertainty(self, e1, r, e2):
        return 0
    
    def projection(self, e1, r, e2):
        return r


class RotatE(nn.Module):
    def __init__(self, p, feat_drop, dim, params=None):
        super(RotatE, self).__init__()
        # self.p = p
        self.feat_drop = feat_drop
        self.margin = params
        self.rel_range = (self.margin + 2.0) / (dim / 2)
        self.pi = 3.14159265358979323846

    def forward(self, e1, r, e2):    
        e1 = F.dropout(e1, p=self.feat_drop, training=self.training)
        re_head, im_head = torch.chunk(e1, 2, dim=1)
        re_tail, im_tail = torch.chunk(e2, 2, dim=1)
        r = r / (self.rel_range / self.pi)
        re_relation = torch.cos(r)
        im_relation = torch.sin(r)
        re_score = re_head * re_relation - im_head * im_relation
        im_score = re_head * im_relation + im_head * re_relation
        re_score = re_score - re_tail
        im_score = im_score - im_tail
        score = torch.stack([re_score, im_score], dim=0)
        pred = score.norm(dim=0).sum(dim=-1)
        return pred
    
    def loss(self, pos_score, neg_score, target):
        return - (F.logsigmoid(self.margin - pos_score) + F.logsigmoid(neg_score - self.margin)).mean()
    
    def uncertainty(self, e1, r, e2):
        return 0
    
    def projection(self, e1, r, e2):
        re_head, im_head = torch.chunk(e1, 2, dim=1)
        re_relation = torch.cos(r)
        im_relation = torch.sin(r)
        re_score = re_head * re_relation - im_head * im_relation
        im_score = re_head * im_relation + im_head * re_relation
        re_score = re_score - re_head
        im_score = im_score - im_head
        score = torch.stack([re_score, im_score], dim=0)
        return score


class MLP(nn.Module):
    def __init__(self, act=torch.relu, hiddens=[], l2_norm=False):
        super(MLP,self).__init__()
        self.hiddens = hiddens
        self.fc_layers = nn.ModuleList()
        self.num_layers = len(self.hiddens) - 1
        self.activation = act
        self.l2_norm = l2_norm
        for i in range(self.num_layers):
            self.fc_layers.append(nn.Linear(self.hiddens[i], self.hiddens[i+1]))

    def forward(self, e):
        for i, fc in enumerate(self.fc_layers):
            if self.l2_norm:
                e = F.normalize(e, p=2, dim=1)
            e = fc(e)
            if i != self.num_layers-1:
                e = self.activation(e)
        return e