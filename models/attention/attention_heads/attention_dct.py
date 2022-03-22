# -*- coding: utf-8 -*-
# ---------------------


import torch
import torch.nn as nn
from math import sqrt

from utils.dct import dct, idct
from utils.dct import create_dct
from conf import Conf

class DCTAttention(nn.Module):
    Q_dct = None

    def __init__(self, cnf):
        # type: (Conf) -> ()
        super().__init__()
        self.cnf = cnf

        # Attention stuff
        self.head_dim = cnf.attention.head_dim
        atn_softmax = nn.Softmax(dim=-1)
        self.add_module('atn_softmax', atn_softmax)

        # DCT stuff
        self.max_n = cnf.attention.dct.get("maxN", None)
        self.max_m = cnf.attention.dct.get("maxM", None)

        # Initialize class variable
        if DCTAttention.Q_dct is None and self.max_n is not None:
            DCTAttention.Q_dct = create_dct(n=self.max_n, m=self.max_m).to(cnf.device)

    def forward(self, Q, K, V, mask=None):

        Q = Q / sqrt(sqrt(self.head_dim))
        K = K / sqrt(sqrt(self.head_dim)) * mask[:, None, :, None]

        # Dct: (BS, NH, SL, E) -> (BS,NH, sl_, E)
        pad = 0
        if self.max_n is not None:
            Q = torch.matmul(self.Q_dct, Q)
            K = torch.matmul(self.Q_dct, K)
            V = torch.matmul(self.Q_dct, V * mask[:, None, :, None])
        else:
            pad = max(0, Q.shape[-2] - self.max_m)
            Q = dct(Q, dim=-2)[..., :self.max_m, :]
            K = dct(K, dim=-2)[..., :self.max_m, :]
            V = dct(V * mask[:, None, :, None], dim=-2)[..., :self.max_m, :]

        # (BS, NH, SL, E) * (BS,NH,E,sl_)
        energy = torch.matmul(Q, torch.transpose(K, -2, -1))
        attention = self.atn_softmax(energy)

        # E*V -> (bs. h, q, k) * (bs, h, v, head_dim) -> (bs, h, q, head_dim)
        # (note: k=v always!)
        if self.max_n is not None:
            x = torch.matmul(torch.matmul(self.Q_dct.t(), attention), V)
        else:
            x = torch.matmul(attention, V)
            x = torch.nn.ConstantPad1d((0, pad), 0)(x.transpose(-1, -2)).transpose(-1, -2)
            x = idct(x, dim=-2)

        return x
