# -*- coding: utf-8 -*-
# ---------------------


import torch
import torch.nn as nn
from math import sqrt

from utils.dct import create_dct, idct_2d, dct_2d
from conf import Conf

class DCTAttentionIdeal(nn.Module):
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
        # self.factor = self.max_n / self.max_m

        # Initialize class variable
        if DCTAttentionIdeal.Q_dct is None and self.max_n is not None:
            DCTAttentionIdeal.Q_dct = create_dct(n=self.max_n, m=self.max_m).to(cnf.device)

    def forward(self, Q, K, V, mask=None):

        Q = Q / sqrt(sqrt(self.head_dim))
        K = K / sqrt(sqrt(self.head_dim)) * mask[:, None, :, None]

        # Dct: (BS, NH, SL, E) -> (BS,NH, sl_, E)
        if self.max_n is not None:
            V = torch.matmul(self.Q_dct, V * mask[:, None, :, None])
        else:
            V = V * mask[:, None, :, None]

        # (BS, NH, SL, E) * (BS,NH,E,sl_)
        atn = torch.matmul(Q, torch.transpose(K, -2, -1))
        atn = self.atn_softmax(atn)
        if self.max_n is not None:
            atn_dct = torch.matmul(torch.matmul(self.Q_dct, atn), self.Q_dct.t())
            # E*V -> (bs. h, q, k) * (bs, h, v, head_dim) -> (bs, h, q, head_dim)
            # (note: k=v always!)
            x = torch.matmul(torch.matmul(self.Q_dct.t(), atn_dct), V)
        else:
            atn_dct = dct_2d(atn)
            atn_dct[:, :, self.max_m:, self.max_m:] *= 0
            x = idct_2d(atn_dct)
            x = torch.matmul(x, V)

        return x
