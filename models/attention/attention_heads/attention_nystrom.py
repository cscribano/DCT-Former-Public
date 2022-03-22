# -*- coding: utf-8 -*-
# ---------------------

"""
Nystformer: A Nystrom-based Algorithm for Approximating Self-Attention

Implementation based upon:
https://github.com/mlpen/Nystromformer/blob/main/reorganized_code/encoders/backbones/efficient_attentions/attention_nystrom.py
"""

import torch
import torch.nn as nn
import math

from conf import Conf

class NystromAttention(nn.Module):
    def __init__(self, cnf):
        # type: (Conf) -> ()

        super().__init__()

        self.head_dim = cnf.attention.head_dim
        self.num_head = cnf.attention.n_heads

        self.num_landmarks = cnf.attention.num_landmarks
        self.seq_len = cnf.dataset.max_seq_len

        self.use_conv = cnf.attention.get("conv_kernel_size", None) is not None
        if self.use_conv:
            self.conv = nn.Conv2d(
                in_channels = self.num_head, out_channels = self.num_head,
                kernel_size = (cnf.attention.conv_kernel_size, 1), padding = (cnf.attention.conv_kernel_size // 2, 0),
                bias = False,
                groups = self.num_head)

    def forward(self, Q, K, V, mask=None):

        if mask is not None:
            Q = Q * mask[:, None, :, None] / math.sqrt(math.sqrt(self.head_dim))
            K = K * mask[:, None, :, None] / math.sqrt(math.sqrt(self.head_dim))
        else:
            Q = Q / math.sqrt(math.sqrt(self.head_dim))
            K = K / math.sqrt(math.sqrt(self.head_dim))

        if self.num_landmarks == self.seq_len:
            attn = torch.nn.functional.softmax(torch.matmul(Q, K.transpose(-1, -2)) - 1e9 * (1 - mask[:, None, None, :]), dim = -1)
            X = torch.matmul(attn, V)
        else:
            Q_landmarks = Q.reshape(-1, self.num_head, self.num_landmarks, self.seq_len // self.num_landmarks, self.head_dim).mean(dim = -2)
            K_landmarks = K.reshape(-1, self.num_head, self.num_landmarks, self.seq_len // self.num_landmarks, self.head_dim).mean(dim = -2)

            kernel_1 = torch.nn.functional.softmax(torch.matmul(Q, K_landmarks.transpose(-1, -2)), dim = -1)
            kernel_2 = torch.nn.functional.softmax(torch.matmul(Q_landmarks, K_landmarks.transpose(-1, -2)), dim = -1)
            if mask is not None:
                kernel_3 = torch.nn.functional.softmax(torch.matmul(Q_landmarks, K.transpose(-1, -2)) - 1e9 * (1 - mask[:, None, None, :]), dim = -1)
            else:
                kernel_3 = torch.nn.functional.softmax(torch.matmul(Q_landmarks, K.transpose(-1, -2)), dim = -1)

            X = torch.matmul(torch.matmul(kernel_1, self.iterative_inv(kernel_2)), torch.matmul(kernel_3, V))

        if self.use_conv:
            X += self.conv(V * mask[:, None, :, None])

        return X

    def iterative_inv(self, mat, n_iter = 6):
        I = torch.eye(mat.size(-1), device = mat.device)
        K = mat
        V = 1 / (torch.max(torch.sum(torch.abs(K), dim = -2)) * torch.max(torch.sum(torch.abs(K), dim = -1))) * K.transpose(-1, -2)
        for _ in range(n_iter):
            KV = torch.matmul(K, V)
            V = torch.matmul(0.25 * V, 13 * I - torch.matmul(KV, 15 * I - torch.matmul(KV, 7 * I - KV)))
        return V

    def extra_repr(self):
        return f'num_landmarks={self.num_landmarks}, seq_len={self.seq_len}'
