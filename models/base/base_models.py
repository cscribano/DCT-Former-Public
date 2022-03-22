# -*- coding: utf-8 -*-
# ---------------------

import torch
from torch import nn
from abc import ABCMeta
from abc import abstractmethod

from conf import Conf

class BaseModel(nn.Module, metaclass=ABCMeta):

    def __init__(self, cnf):
        # type: (Conf) -> None
        super().__init__()

        self.cnf = cnf

    def forward(self, *args, **kwargs):
        # type: (*str, **int) -> torch.Tensor
        """
        Defines the computation performed at every call.
        Should be overridden by all subclasses.
        """
        ...

    def load_weights(self, state_dict):
        # type: (dict) -> None

        # This is useful to deal with models saved with DDP wrapper
        restore_kv = {key.replace("module.", ""): state_dict[key] for key in
                      state_dict.keys()}
        self.load_state_dict(restore_kv, strict=True)

    @property
    def is_cuda(self):
        # type: () -> bool
        """
        :return: `True` if the model is on Cuda; `False` otherwise
        """
        return next(self.parameters()).is_cuda


    def requires_grad(self, flag):
        # type: (bool) -> None
        """
        :param flag: True if the model requires gradient, False otherwise
        """
        for p in self.parameters():
            p.requires_grad = flag

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.cnf.model.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.cnf.model.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class TrainableModel(BaseModel, metaclass=ABCMeta):
    def __init__(self, cnf, backbone_cls, backbone_args):
        # type: (Conf, BaseModel, dict) -> None

        super().__init__(cnf)
        self.cnf = cnf
        if backbone_cls is not None:
            self.backbone = backbone_cls(cnf, **backbone_args)
        else:
            self.backbone = None

        # Dataset
        self.train_set = (None, {'arg1': None, "argN": None}, None)
        self.val_set = (None, {'arg1': None, "argN": None})

    def model_forward(self, *args, **kwargs):  # type: (*str, **int) -> torch.Tensor
        return self.backbone.forward(*args, **kwargs)

    @abstractmethod
    def forward(self, sample, y=None):
        # type: (tuple, torch.tensor) -> torch.Tensor
        """
        Return a computed value that can be used for backward pass computation
        Should be overridden by all subclasses.
        :param x: Data returned from Dataloader
        :param y: (optional) precomputed model's output, avoid explicit call to backbone.forward()
        :return: Computed Loss
        """
        ...

    @abstractmethod
    def val_loss(self, sample):
        # type: (tuple) -> torch.Tensor
        """
        Return a computed value that will be used for model evaluation after each epoch.
        This value won't be used for gradient computation.
        :param x: Data returned from Dataloader
        :return: Computed Validation score
        """
        ...

    @abstractmethod
    def test_step(self, sample, **kwargs):
        # type: (tuple, **str) -> any
        """
        Perform an inference step and return the desired output
        used for model evaluation.
        :param sample: A batch of data
        :return: model inference result
        """
        ...

