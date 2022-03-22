# -*- coding: utf-8 -*-
# ---------------------

import math
import copy
import torch
import torch.nn as nn
from torch.autograd import Variable

def clones(module, N):
    # Produce N identical layers.
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def len_to_mask(lenghts, cls_token=False):
    # type: (torch.tensor, bool) -> torch.tensor
    if cls_token:
        lenghts += 1

    max_len = max(lenghts.view(-1))
    mask = torch.arange(max_len)[None, ...] < lenghts[... , None]

    return mask.unsqueeze(-2)

class PositionalEncoding(nn.Module):
    """
    https://nlp.seas.harvard.edu/2018/04/03/attention.html
    """
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, s, pos=None):
        if pos is None:
            r = Variable(self.pe[:, :s],requires_grad=False)
        else:
            r = Variable(self.pe[:, pos], requires_grad=False)[0]

        return r
