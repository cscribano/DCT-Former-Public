# -*- coding: utf-8 -*-
# ---------------------

"""
Rethinking Attention with Performers

Implementation based upon:
https://github.com/mlpen/Nystromformer/blob/main/reorganized_code/encoders/backbones/efficient_attentions/attention_performer.py
"""


import torch
import torch.nn as nn
import math
from performer_pytorch import FastAttention

from conf import Conf

class PerformerAttention(nn.Module):
    def __init__(self, cnf):
        # type: (Conf) -> ()
        super().__init__()

        self.head_dim = cnf.attention.head_dim
        self.rp_dim = cnf.attention.rp_dim
        self.kernel_type = cnf.attention.kernel_type
        if self.kernel_type == "relu":
            self.attn_fn = FastAttention(dim_heads = self.head_dim, nb_features = self.rp_dim,
                                         causal = False, kernel_fn = nn.ReLU())
        elif self.kernel_type == "exp":
            self.attn_fn = FastAttention(dim_heads = self.head_dim, nb_features = self.rp_dim,
                                         causal = False, kernel_fn = torch.exp)

    def forward(self, Q, K, V, mask):
        if mask is not None:
            return self.attn_fn(
                Q / math.sqrt(math.sqrt(self.head_dim)),
                K / math.sqrt(math.sqrt(self.head_dim)) * mask[:, None, :, None],
                V * mask[:, None, :, None])
        else:
            return self.attn_fn(
                Q / math.sqrt(math.sqrt(self.head_dim)),
                K / math.sqrt(math.sqrt(self.head_dim)),
                V )
    def extra_repr(self):
        return f'rp_dim={self.rp_dim}, kernel_type={self.kernel_type}'
