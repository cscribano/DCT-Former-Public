# -*- coding: utf-8 -*-
# ---------------------
import os

import torch
import torch.nn as nn
from transformers import BatchEncoding, AutoTokenizer, BertConfig, AutoModel
from transformers.models.bert.modeling_bert import BertModel

from conf import Conf
from models import TrainableModel, BERTogna

class BERTogna_ImDbClassification(TrainableModel):

    def __init__(self, cnf, pretrain_ck=None):
        super().__init__(cnf, BERTogna, {})

        self.dropout = nn.Dropout(cnf.model.classifier_dropout)
        self.classifier = nn.Linear(cnf.model.pooler_dim, 1)

        # Load backbone weights
        if pretrain_ck is not None:

            # Check if pretrain_ck is in logdir
            pt_path = os.path.join(cnf.project_log_path, pretrain_ck)
            if not os.path.isfile(pt_path):
                # Otherwise check if is abspath
                pt_path = pretrain_ck

            # Load weights
            ck = torch.load(pt_path, map_location={'cuda:%d' % 0: cnf.device})
            backbone_ck = ck["module"]
            backbone_ck = {k[9:]: v for k, v in backbone_ck.items() if k[:8] == "backbone"}

            ft_lf = self.cnf.attention.type is not 'LinformerAttention' and self.cnf.experiment.is_finetune is True
            if ft_lf:
                backbone_ck = {k:v for k, v in backbone_ck.items() if 'atn_head.E' not in k}
                self.backbone.load_state_dict(backbone_ck, strict=False)
            else:
                self.backbone.load_state_dict(backbone_ck, strict=True)

            print("==== LOADED FINETUNE CK ====")
        else:
            self.apply(self._init_weights)

        # Loss
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, sample, y=None):
        # type: (BatchEncoding, torch.tensor) -> tuple

        tokens, labels = sample

        input_ids = tokens[0].to(self.cnf.device)
        attention_mask = tokens[1].to(self.cnf.device)
        token_type_ids = tokens[2].to(self.cnf.device)

        if labels is not None:
            labels = labels.to(self.cnf.device)

        # BERT* Inference
        _, logits = self.backbone(tokens=input_ids,
                                        token_type_ids=token_type_ids,
                                        attention_mask=attention_mask)

        logits = self.classifier(self.dropout(logits))

        # Loss
        if labels is not None:
            loss = self.criterion(logits, labels)
        else:
            loss = None

        return (loss, logits)

    def val_loss(self, sample):
        # type: (tuple) -> torch.Tensor
        return self.forward(sample)[0], None

    def test_step(self, sample, **kwargs):

        logits = self.forward(sample)[1]
        return (logits, )


if __name__ == '__main__':

    s1 = "[CLS] the cat is on the table, where is the cat?"
    s2 = "[CLS] The quick brown fox jumps over the lazy dog"

    y_true = torch.tensor([0,1], dtype=torch.torch.float32).reshape(-1, 1).cuda()

    # Modeling
    cnf = Conf(exp_name='default')
    model = BERTogna_ImDbClassification(cnf).to(cnf.device)

    print(model)

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(pytorch_total_params)

    y = model(([s1,s2], y_true))


