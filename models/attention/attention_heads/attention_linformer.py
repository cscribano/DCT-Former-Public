# -*- coding: utf-8 -*-
# ---------------------

"""
Linformer: Self-Attention with Linear Complexity, Wang et.al

Implementation based upon:
https://github.com/mlpen/Nystromformer/blob/main/reorganized_code/encoders/backbones/efficient_attentions/attention_linformer.py
"""

import torch
import torch.nn as nn
import math

from conf import Conf

class LinformerAttention(nn.Module):

    def __init__(self, cnf):
        # type: (Conf) -> ()
        super().__init__()

        self.num_head = cnf.attention.n_heads
        self.head_dim = cnf.attention.head_dim
        self.linformer_k = cnf.attention.linformer_k
        self.square = cnf.attention.get("square", False)
        self.seq_len = cnf.attention.dataset_N

        atn_softmax = nn.Softmax(dim=-1)
        self.add_module('atn_softmax', atn_softmax)

        E = cnf.model.get("projection_matrix", None)
        if E is not None:
            self.E = E
        else:
            projection_matrix = nn.Parameter(torch.Tensor(self.num_head, self.linformer_k, self.seq_len))
            torch.nn.init.normal_(projection_matrix, std=0.02)
            self.E = projection_matrix

    def forward(self, Q, K, V, mask):
        if mask is None:
            K = torch.matmul(self.E, K)
            V = torch.matmul(self.E, V)
        else:
            K = torch.matmul(self.E.float(), K * mask[:, None, :, None])
            V = torch.matmul(self.E.float(), V * mask[:, None, :, None])

        dot = torch.matmul(Q, torch.transpose(K, -2, -1))
        if self.square:
            dot = torch.matmul(self.E, dot)
        dot = dot / math.sqrt(self.head_dim)

        attn = self.atn_softmax(dot)

        X = torch.matmul(attn, V)
        if self.square:
            X = torch.matmul(self.E.transpose(-1,-2).float(), X)

        return X

    def extra_repr(self):
        return f'linformer_k={self.linformer_k}'
