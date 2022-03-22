# -*- coding: utf-8 -*-
# ---------------------

"""
Courtesy of:
https://github.com/mlpen/Nystromformer/blob/main/reorganized_code/encoders/backbones/efficient_attentions/attention_linear.py
"""

import torch
import torch.nn as nn
import math

from conf import Conf

class LinearAttention(nn.Module):

    def __init__(self, cnf):
        # type: (Conf) -> ()
        super().__init__()

    def forward(self, Q, K, V, mask):
        Q = (nn.functional.elu(Q) + 1) / math.sqrt(math.sqrt(Q.size(2)))
        K = (nn.functional.elu(K) + 1) * mask[:, None, :, None] / math.sqrt(math.sqrt(K.size(2)))
        V = V * mask[:, None, :, None]

        X = torch.matmul(Q, torch.matmul(torch.transpose(K, -2, -1), V))

        return X