# -*- coding: utf-8 -*-
# ---------------------

"""
Reformer: The Efficient Transformer

Implementation based upon:
https://github.com/mlpen/Nystromformer/blob/main/reorganized_code/encoders/backbones/efficient_attentions/attention_reformer.py
"""


import torch
import torch.nn as nn
from transformers.models.reformer.modeling_reformer import LSHSelfAttention, ReformerConfig

from conf import Conf

class LSHAttention(LSHSelfAttention):
    def __init__(self, cnf, query, key, value):
        # type: (Conf, torch.tensor, torch.tensor, torch.tensor) -> ()

        self.num_hash = cnf.attention.num_hash
        reformer_config = ReformerConfig()
        reformer_config.attention_head_size = cnf.attention.head_dim
        reformer_config.num_attention_heads = cnf.attention.num_heads
        reformer_config.attn_layers = ["lsh"]
        reformer_config.num_hashes = cnf.attention.num_hash
        reformer_config.is_decoder = False
        reformer_config.max_position_embeddings = cnf.model.max_seq_len
        reformer_config.hidden_size = cnf.attention.head_dim #config["dim"]
        super().__init__(reformer_config)
        self.query_key.weight = query.weight
        self.value.weight = value.weight

    def forward(self, X, mask):
        return super().forward(hidden_states = X, attention_mask = mask).hidden_states

    def extra_repr(self):
        return f'num_hash={self.num_hash}'