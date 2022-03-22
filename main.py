# -*- coding: utf-8 -*-
# ---------------------

import os
import torch
import torch.distributed as dist
import signal
import logging
from conf import Conf

import time
import click
import torch.backends.cudnn as cudnn
import deepspeed as ds

from trainer_pretrain import Trainer_PT
from trainer_finetune import Trainer_FT

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

@click.command()
@click.option('--exp_name', type=str, default=None)
@click.option('--local_rank', type=int, default=0)
@click.option('--resume', type=bool, default=False)
@click.option('--mode', type=click.Choice(['run', 'test'],
										  case_sensitive=False), default="run")
@click.option('--test_ck', type=click.Choice(['last', 'best'], case_sensitive=False), default="last")
@click.option('--conf_file_path', type=str, default=None)
@click.option('--seed', type=int, default=None)
def main(exp_name, local_rank, resume, mode, test_ck, conf_file_path, seed):
	# type: (str, int, bool, str, str, str, int) -> None

	assert torch.backends.cudnn.enabled, "Running without cuDNN is discouraged"

	# if `exp_name` is None,
	# ask the user to enter it
	if exp_name is None:
		exp_name = input('>> experiment name: ')

	# if `exp_name` contains '!',
	# `log_each_step` becomes `False`
	log_each_step = True
	if '!' in exp_name:
		exp_name = exp_name.replace('!', '')
		log_each_step = False

	# if `exp_name` contains a '@' character,
	# the number following '@' is considered as
	# the desired random seed for the experiment
	split = exp_name.split('@')
	if len(split) == 2:
		seed = int(split[1])
		exp_name = split[0]

	if mode == "test":
		resume = True
	cnf = Conf(conf_file_path=conf_file_path, seed=seed,
			   exp_name=exp_name, resume=resume, log_each_step=log_each_step)

	global Trainer
	Trainer = Trainer_FT if cnf.finetune else Trainer_PT

	# Setup logging
	logging.basicConfig(
		format='[%(asctime)s] [p%(process)s] [%(pathname)s:%(lineno)d] [%(levelname)s] %(message)s',
		level=logging.INFO,
	)

	print(f'\n▶ Starting Experiment \'{exp_name}\' [seed: {cnf.seed}]')

	cnf_attrs = vars(cnf)
	for k in cnf_attrs:
		s = f'{k} : {cnf_attrs[k]}'
		logging.info(s)

	# Assuming 1 process == 1 GPU
	if not cnf.slurm:
		"""
		python -m torch.distributed.launch --nproc_per_node=<N> --master_addr="127.0.0.1" --master_port=1234 main.py --exp_name=<exp_name>
		deepspeed main.py --exp_name=<exp_name> --seed=6969
		"""
		DDP_Trainer(local_rank, cnf, mode, test_ck)
	else:
		rank = int(os.environ["SLURM_PROCID"])
		DDP_Trainer(rank, cnf, mode, test_ck)

	signal.signal(signal.SIGINT, cleanup)
	signal.signal(signal.SIGTERM, cleanup)

def init_process(backend='nccl'):
	""" Initialize the distributed environment. """
	"""
	WARNING: those variables are automatically set when calling torch.distributed.launch...
	os.environ['MASTER_ADDR'] = host
	os.environ['MASTER_PORT'] = str(port)
	"""

	print("============NODE================")
	print(
	os.environ.get('RANK', ""),
	os.environ.get('WORLD_SIZE', ""),
	os.environ.get('MASTER_ADDR', ""),
	os.environ.get('MASTER_PORT', ""),
	os.environ.get('LOCAL_RANK', ""),
	)
	print("================================")

	ds.init_distributed(backend, auto_mpi_discovery=True)

def init_process_slurm(rank, size, gpu_id, jobid, backend='nccl'):
	# type: (int, int, int, int, str) -> None

	hostfile = f"dist_url.{jobid}.txt"

	if rank == 0:
		dist_url = "tcp://{}:{}".format(Conf.HOSTNAME, Conf.PORT)
		with open(hostfile, "w") as f:
			f.write(dist_url)
	else:
		while not os.path.exists(hostfile):
			time.sleep(1)
		with open(hostfile, "r") as f:
			dist_url = f.read()

	print(f"{dist_url}")

	#     required_env = ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK"]
	os.environ['RANK'] = str(rank)
	os.environ['WORLD_SIZE'] = str(size)
	os.environ['MASTER_ADDR'] = dist_url.split(":")[-1]
	os.environ['MASTER_PORT'] = dist_url.split(":")[-2][2:]
	os.environ['LOCAL_RANK'] = str(gpu_id)

	ds.init_distributed(backend, init_method=dist_url, auto_mpi_discovery=False)


def cleanup():
	dist.destroy_process_group()

def DDP_Trainer(rank, cnf, mode, test_ck):
	# type: (int, Conf, str, str) -> None

	if cnf.slurm:
		init_process_slurm(rank, cnf.world_size, cnf.gpu_id, cnf.jobid)
	else:
		init_process()

	cnf.setup_device_id(rank)

	print(
		f"Rank {rank + 1}/{cnf.world_size} process initialized.\n"
	)

	trainer = Trainer(cnf, rank)

	if mode == "run":
		trainer.run()
	else:
		trainer.test(modes=(mode, ), load_best=test_ck == "best")

if __name__ == '__main__':
	main()
