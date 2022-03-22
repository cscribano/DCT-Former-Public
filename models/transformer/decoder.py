# -*- coding: utf-8 -*-
# ---------------------

import torch
import torch.nn as nn

from conf import Conf
from models.attention import MultiHeadAttention

class DecoderBlock(nn.Module):

    def __init__(self, cnf):
        # type: (Conf) -> ()

        super().__init__()
        self.cnf = cnf

        emb_dim = cnf.attention.emb_dim
        heads = cnf.attention.n_heads
        dropout = cnf.attention.dropout

        #if enc_emb_dim is None:
        #    enc_emb_dim = emb_dim
        #assert (emb_dim % heads == 0 and enc_emb_dim % heads == 0)

        assert emb_dim % heads == 0, "The embeddings dim must be divisible by heads number"

        # 1-st sub-layer: (masked) self-attention)
        self.self_attention_layer = MultiHeadAttention(cnf)
        self.self_atn_dropout = nn.Dropout(dropout)
        self.self_atn_norm = nn.LayerNorm(emb_dim)

        # 2-nd sub-layer: encoder-decoder attention
        self.attention_layer = MultiHeadAttention(cnf)
        self.dropout1 = nn.Dropout(dropout)
        self.atn_norm = nn.LayerNorm(emb_dim)

        hid_dim = cnf.model.transformer_hid_dim
        # 3-rd sub-layer: Feed Forward(s)
        self.linear_block = nn.Sequential(
            nn.Linear(emb_dim, hid_dim),
            nn.GELU(),
            nn.Linear(hid_dim, emb_dim),
        )
        self.dropout2 = nn.Dropout(dropout)
        self.linear_norm = nn.LayerNorm(emb_dim)

    def forward(self, input, encoder_out, mask=None, encoder_mask=None):
        """
        :param input:
        :param encoder_out:
        :param mask:
        :param encoder_mask:
        :return:
        """

        # Sub-layer 1
        # Compute Multi head-self attention and apply residual connection
        self_atn = self.self_attention_layer(input, mask=mask)
        self_atn = self.self_atn_dropout(self_atn)
        # In self-attention Query, Key and Value are the same thing
        self_atn = self.self_atn_norm(self_atn+input)

        '''
        Sub-layer 2: Encoder-Decoder attention
        Here attention Query is the Decoder state, aka the output from self-attention
        Key and Value is the decoder output.
        '''
        ed_atn = self.attention_layer(self_atn, mask=encoder_mask, encoder_out=encoder_out)
        ed_atn = self.dropout1(ed_atn)
        # In Encoder self-attention Query, Key and Value are the same thing
        ed_atn = self.atn_norm(ed_atn+self_atn)

        # Sub-layer 3
        lin = self.linear_block(ed_atn)
        lin = self.dropout2(lin)
        lin = self.linear_norm(lin+ed_atn)

        return lin


class TransformerDecoder(nn.Module):

    def __init__(self, cnf):
        # type: (Conf) -> ()

        super().__init__()

        n_blocks = cnf.model.get("decoder_blocks", 6)
        emb_dim = cnf.attention.emb_dim
        output_dim = cnf.model.get("output_dim", None)

        #self.pos_embedding = PositionalEncoding(emb_dim, 0.0, max_len)

        '''
        The Decoder is also made of N=6 identical blocks
        '''
        self.decoder_blocks = nn.ModuleList([DecoderBlock(cnf) for _ in range(n_blocks)])

        if output_dim is not None:
            self.output = nn.Linear(emb_dim, output_dim)
        else:
            self.output = None

    def forward(self, trg, encoder_out, mask=None, encoder_mask=None, pos=None):

        # this is identical to encoder
        # x = self.pos_embedding(trg, pos)
        x = trg

        for db in self.decoder_blocks:
            x = db(x, encoder_out, mask, encoder_mask)

        if self.output is not None:
            x = self.output(x)

        return x

if __name__ == '__main__':

    cnf = Conf(exp_name='default')

    e_out = torch.ones(32, 12, 512, dtype=torch.float32).to('cuda')
    s_emb = torch.ones(32, 32, 512, dtype=torch.float32).to('cuda')  # (BS, SL, emb_dim)
    e = TransformerDecoder(cnf).to('cuda')

    dec_out = e(s_emb, e_out)

    print(dec_out.shape,)
