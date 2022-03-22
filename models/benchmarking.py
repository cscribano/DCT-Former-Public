# -*- coding: utf-8 -*-
# ---------------------
import click
import torch
import numpy as np
from time import time

from conf import Conf
from models import BERTogna

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

@click.command()
@click.option('--exp_name', type=str, default='dct')
@click.option('--n1', type=int, default=None)
@click.option('--n2', type=int, default=None)
@click.option('--bs', type=int, default=None)
def main(exp_name=None, n1=None, n2=None, bs=None):
	# if `exp_name` is None,
	# ask the user to enter it
	if exp_name is None:
		exp_name = input('>> experiment name: ')

	# --exp_name=dct --n1=128 --n2=64 --bs=128
	conf_dict = {
		# model name: (seq_len, configurable, scale)
		"dct": ("cnf.attention.dct.maxN", "cnf.attention.dct.maxM", 1 / 4),
		"linformer": ("cnf.attention.dataset_N", "cnf.attention.linformer_k", 1 / 8),
		"performer": ("cnf.attention.dataset_N", "cnf.attention.rp_dim", 1 / 8),
		"nystrom": ("cnf.dataset.max_seq_len", "cnf.attention.num_landmarks", 1 / 8),
	}

	# Define configuration
	if n1 is not None and n2 is not None and bs is not None:
		sl_bs = {n1: bs}
	else:
		sl_bs = {128: 256, 512: 32, 1024: 16, 4096: 1}
	cd = conf_dict.get(exp_name, None)

	for seq_len, batch_size in sl_bs.items():

		cnf = Conf(exp_name=f"benchmarking/{exp_name}", log=False)
		if cd is not None:
			if n2 is None:
				exec(f"{cd[0]} = {seq_len}")
				exec(f"{cd[1]} = int({seq_len}*{cd[2]})")
			else:
				exec(f"{cd[0]} = {seq_len}")
				exec(f"{cd[1]} = {n2}")

		# Instantiate model
		model = BERTogna(cnf).eval()
		num_iter = 50

		time_list = []
		torch.cuda.reset_peak_memory_stats()
		model = model.cuda()

		for _ in range(num_iter):
			# Prepare sample data
			input_ids = torch.randint(0, cnf.model.vocab_size, (batch_size, seq_len)).long().to(cnf.device)
			token_type_ids = torch.zeros(batch_size, seq_len).long().to(cnf.device)
			attention_mask = None

			torch.cuda.synchronize()
			t0 = time()

			# Inference
			_, _ = model(tokens=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

			torch.cuda.synchronize()
			t1 = time()
			time_list.append((t1 - t0) / batch_size)

		per_inst_time_avg = np.mean(time_list) * 1000
		per_inst_time_std = np.std(time_list) * 1000
		memory_per_inst = torch.cuda.max_memory_allocated() / 1024 / 1024 / batch_size

		results = {
			"batch_size": batch_size,
			"sequence_lenght": seq_len,
			"per_inst_time_avg (ms)": round(per_inst_time_avg, 3),
			"per_inst_time_std (ms)": round(per_inst_time_std, 3),
			"memory_per_inst (MB)": round(memory_per_inst, 3),
		}

		print(results)

		del model
		torch.cuda.empty_cache()

if __name__ == '__main__':
	main()
