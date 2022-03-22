# -*- coding: utf-8 -*-
# ---------------------

import torch.optim as optim
from models import *
from dataset import *

def parse_model(conf):
	# type: (Conf) -> tuple

	model_dict = conf.model
	try:
		model = (eval(model_dict["name"]), model_dict["args"])
	except:
		raise NotImplementedError(f"Model configuration {model_dict['name']} is not supported or valid")

	return model

def parse_dataset(conf, conf_str='train_dataset'):
	# type: (Conf, str) -> tuple

	ds_dict = conf.dataset[conf_str]

	ds = eval(ds_dict["name"])(conf, **ds_dict["args"])
	loader_args = ds_dict.get("loader_args", {})
	collate_fn = ds_dict.get("collate_fn", None)

	if collate_fn is not None:
		collate_fn = eval(collate_fn)

	return (ds, loader_args, collate_fn)

def parse_optimizer(conf, model):
	# type: (Conf, torch.nn.Module) -> torch.optim

	param_optimizer = list(model.named_parameters())
	param_optimizer = [n for n in param_optimizer if "pooler" not in n[0]]

	decay = conf.optimizer.get("weight_decay", 0.0)
	no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
	optimizer_grouped_parameters = [
		{
			"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
			"weight_decay": decay,
		},
		{
			"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
			"weight_decay": 0.0,
		},
	]

	# Instantiate optimizer
	optimizer = eval(f"optim.{conf.optimizer['name']}")(params=optimizer_grouped_parameters, **conf.optimizer["args"])

	return optimizer, optimizer_grouped_parameters
