# -*- coding: utf-8 -*-
# ---------------------

import torch
import torch.nn as nn

from conf import Conf
from models.attention import MultiHeadAttention, DCT_MHSA_Naive

class EncoderBlock(nn.Module):

    def __init__(self, cnf):
        # type: (Conf) -> ()

        super().__init__()
        self.cnf = cnf

        emb_dim = cnf.attention.emb_dim
        heads = cnf.attention.n_heads
        dropout = cnf.attention.dropout

        assert emb_dim % heads == 0, "The embeddings dim must be divisible by heads number"

        attention_class = cnf.model.get("attention_class", "MultiHeadAttention")
        """
        print("------------------------------------------")
        print(f"----USING ATTENTION {attention_class}----")
        print("------------------------------------------")
        """

        self.attention_layer = eval(attention_class)(cnf)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.atn_norm = nn.LayerNorm(emb_dim)

        hid_dim = cnf.model.transformer_hid_dim
        self.linear_block = torch.nn.Sequential(
            nn.Linear(emb_dim, hid_dim),
            nn.GELU(),
            nn.Linear(hid_dim, emb_dim),
        )
        self.linear_norm = nn.LayerNorm(emb_dim)


    def forward(self, input, mask=None):
        """
        :param input: Embedding for source sequence (padded, batch first): (BS, seq_len, emb_dim)
        :param mask: Mask to avoid attending to padding: (BS, seq_len, 1)
        :return: (BS, seq_len, emb_dim)
        """

        # Sub-layer 1
        # Compute Multi head-attention and apply residual connection
        atn = self.attention_layer(input, mask=mask)
        atn = self.dropout1(atn)
        # In Encoder self-attention Query, Key and Value are the same shape
        atn = self.atn_norm(atn+input)

        # Sub-layer 2
        lin = self.linear_block(atn)
        lin = self.dropout2(lin)
        lin = self.linear_norm(lin+atn)


        return lin

class TransformerEncoder(nn.Module):

    def __init__(self, cnf):
        # type: (Conf) -> ()

        super().__init__()
        self.cnf = cnf

        n_blocks = cnf.model.get("encoder_blocks", 6)

        '''
        The encoder is made of N=6 identical blocks
        '''
        self.encoder_blocks = nn.ModuleList([EncoderBlock(cnf) for _ in range(n_blocks)])

    def forward(self, x, mask=None, pos=None):
        """
        :param x: Embedding for source sequence (padded, batch first): (BS, seq_len, emb_dim)
        :param mask: Mask to avoid attending to padding: (BS, seq_len, 1)
        :return: (BS, seq_len, emb_dim)
        """

        for b in self.encoder_blocks:
            x = b(x, mask)

        return x


if __name__ == '__main__':

    cnf = Conf(exp_name='default')

    s_emb = torch.ones(32, 12, 512, dtype=torch.float32).to('cuda')  # (BS, SL, emb_dim)
    e = TransformerEncoder(cnf).to('cuda')

    enc_out = e(s_emb)

    print(enc_out.shape,)
