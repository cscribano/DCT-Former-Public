# -*- coding: utf-8 -*-
# ---------------------
from typing import Tuple, Any

import os
import torch
import torch.nn as nn

from conf import Conf
from models import BaseModel, TrainableModel
from models.transformer import TransformerEncoder
from transformers import BertLMHeadModel, BertConfig
from transformers.models.bert.modeling_bert import BertLMHeadModel


class Embeddings(nn.Module):
    def __init__(self, cnf):
        # type: (Conf) -> ()
        super().__init__()

        self.word_embeddings = nn.Embedding(cnf.model.vocab_size, cnf.attention.emb_dim)
        self.position_embeddings = nn.Embedding(cnf.model.max_seq_len, cnf.attention.emb_dim)
        self.token_type_embeddings = nn.Embedding(cnf.model.max_seq_len, cnf.attention.emb_dim)

        nn.init.normal_(self.word_embeddings.weight, std=0.02)
        nn.init.normal_(self.position_embeddings.weight, std=0.02)
        nn.init.normal_(self.position_embeddings.weight, std=0.02)

        self.register_buffer("position_ids", torch.arange(cnf.model.max_seq_len).expand((1, -1)))

        self.norm = nn.LayerNorm(cnf.attention.emb_dim)
        self.dropout = nn.Dropout(p=cnf.model.dropout)

    def forward(self, input_ids, token_type_ids, position_ids=None,  past_key_values_length=0):
        # type: (torch.Tensor, torch.Tensor, torch.Tensor, int) -> torch.Tensor

        if position_ids is None:
            seq_length = input_ids.shape[1]
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        X_token = self.word_embeddings(input_ids)
        X_seq = self.token_type_embeddings(token_type_ids)
        X_pos = self.token_type_embeddings(position_ids)

        X = X_token + X_pos + X_seq
        X = self.dropout(self.norm(X))

        return X


class BERTogna(BaseModel):
    def __init__(self, cnf):
        super().__init__(cnf)
        self.cnf = cnf

        self.embedding_layer = Embeddings(cnf)
        self.encoder = TransformerEncoder(cnf)

        # Pooler
        self.dense = nn.Linear(cnf.attention.emb_dim, cnf.model.pooler_dim)
        self.activation = nn.Tanh()

        self.apply(self._init_weights)

    def forward(self, tokens, attention_mask, token_type_ids, *args, **kwargs):
        # type: (torch.Tensor, torch.Tensor, torch.Tensor, *str, **int) -> Tuple[Any, Any]

        x = self.embedding_layer(tokens, token_type_ids)
        x = self.encoder(x, attention_mask)

        first_token_tensor = x[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)

        return x, pooled_output

class MLMHead(nn.Module):
    def __init__(self, cnf, embeddings):
        # type: (Conf, torch.nn.Module) -> ()

        super().__init__()

        embedding_dim = cnf.attention.emb_dim
        dim = cnf.model.get("dim", embedding_dim)
        self.sparse_predict = cnf.experiment.get("sparse_mask_prediction", True)

        self.dense = nn.Linear(dim, embedding_dim)
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(embedding_dim, eps=1e-12)
        self.mlm_class = nn.Linear(embedding_dim, cnf.model.vocab_size, bias=False)

        self.mlm_class.weight = embeddings.weight
        self.bias = nn.Parameter(torch.zeros(cnf.model.vocab_size))
        if not self.sparse_predict:
            self.mlm_class.bias = self.bias

    def forward(self, X, masked_token_indexes):

        if self.sparse_predict:
            if masked_token_indexes is not None:
                X = X.view(-1, X.shape[-1])[
                    masked_token_indexes
                ]

        X = self.act(self.dense(X))
        X = self.norm(X)
        scores = self.mlm_class(X)

        if not self.sparse_predict:
            scores = torch.index_select(
                scores.view(-1, scores.shape[-1]), 0, masked_token_indexes
            )

        return scores

class BERTogna_LMPrediction(TrainableModel):

    def __init__(self, cnf):
        super().__init__(cnf, BERTogna, {})

        # MLM Head
        self.mlm_head = MLMHead(self.cnf, self.backbone.embedding_layer.word_embeddings)

        # Loss
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)

        # Load weights of a previous experiment
        finetune_ck = self.cnf.experiment.get("finetune_ck", None)

        if finetune_ck is not None:
            print("===============================================")
            print(f"[MODEL] Loading weights from {finetune_ck}....")
            print("===============================================")

            pt_path = os.path.join(cnf.project_log_path, finetune_ck,
                                   'training_ck', 'mp_rank_00_model_states.pt')

            # Load weights
            ck = torch.load(pt_path, map_location={'cuda:%d' % 0: cnf.device})
            backbone_ck = ck["module"]
            self.load_state_dict(backbone_ck, strict=True)
        else:
            self.apply(self._init_weights)

    def forward(self, sample, infer=False):
        # type: (list, bool) -> tuple

        input_ids = sample[1].to(self.cnf.device)
        token_type_ids = sample[3].to(self.cnf.device)
        attention_mask = sample[2].to(self.cnf.device)
        masked_lm_labels = sample[4].to(self.cnf.device)

        # BERT EMbedding
        sequence_output = self.backbone(tokens=input_ids,
                                        token_type_ids=token_type_ids,
                                        attention_mask=attention_mask)[0]

        masked_token_indexes = torch.nonzero((masked_lm_labels + 1).view(-1), as_tuple=False).view(
            -1
        )
        prediction_scores = self.mlm_head(sequence_output, masked_token_indexes)

        target = torch.index_select(masked_lm_labels.view(-1), 0, masked_token_indexes)
        prediction_scores = prediction_scores.view(-1, self.cnf.model.vocab_size)

        if masked_lm_labels is None or infer:
            return (prediction_scores, target)
        else:
            target = torch.index_select(masked_lm_labels.view(-1), 0, masked_token_indexes)
            masked_lm_loss = self.criterion(prediction_scores, target)

            outputs = (masked_lm_loss,)
            return outputs

    def val_loss(self, sample):
        # type: (tuple) -> torch.Tensor
        return self.forward(sample)[0], None

    def test_step(self, sample, **kwargs):
        logits, target = self.forward(sample, True)

        return (logits, target)


if __name__ == '__main__':

    from pytorch_memlab import MemReporter

    # Modeling
    cnf = Conf(exp_name='armstrong/pretrain_small', log=False)
    #cnf.attention.dct.n_scale_method = "sqrt"

    model = BERTogna_LMPrediction(cnf).to(cnf.device)
    sample = torch.load('../evaluation/sample.pt')

    reporter = MemReporter(model)
    out = model(sample)
    reporter.report(verbose=True)
