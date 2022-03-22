# -*- coding: utf-8 -*-
# ---------------------

import math
from datetime import datetime
from time import time
import json

import numpy as np
import torch
from utils import save_git_stuff
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
import deepspeed as ds

# from evaluation import inference_on_test
from conf import Conf
from models.provider import parse_model, parse_optimizer, parse_dataset
from evaluation import ft_inference_on_test


class Trainer_FT(object):

	def __init__(self, cnf, rank):
		# type: (Conf, int) -> Trainer_FT

		self.cnf = cnf
		self.rank = rank
		self.epoch = 0
		self.log_path = cnf.exp_log_path

		# init model and optimizer
		trainable, trainable_args = parse_model(cnf)

		# Retrieve dataset class and arguments
		training_set, trainloader_args, collate_fn = parse_dataset(cnf, "train_dataset")
		self.val_set, valloader_args, _ = parse_dataset(cnf, "val_dataset")

		# init validation loader
		val_sampler = DistributedSampler(self.val_set, shuffle=False)

		self.val_loader = DataLoader(
			dataset=self.val_set, sampler=val_sampler,
			collate_fn=collate_fn, **valloader_args
		)

		# Instantiate trainable object wrapping DeepSpeed model
		model = trainable(self.cnf, **trainable_args.todict())  # type: TrainableModel

		# Retrieve optimizer, parameters and scheduler
		optimizer, parameters = parse_optimizer(cnf, model)

		self.model, self.optimizer, self.train_loader, _ = ds.initialize(config=cnf.deepspeed.todict(), model=model,
																	model_parameters=parameters,
																	optimizer=optimizer,
																	training_data=training_set, collate_fn=collate_fn)

		# bug in ds: https://github.com/microsoft/DeepSpeed/pull/1391
		self.train_len = len(self.train_loader) / self.train_loader.batch_size

		self.val_losses = []

		if self.rank == 0:
			# init logging stuffs
			print(f'tensorboard --logdir={cnf.project_log_path.abspath()}\n')
			self.sw = SummaryWriter(self.log_path)
			self.train_losses = []

		# starting values
		self.best_val_loss = None

		# possibly load checkpoint
		self.load_ck()

	def load_ck(self):
		"""
		load training checkpoint
		"""
		ck_path = self.log_path / 'training_ck'
		if ck_path.exists():

			_, ck = self.model.load_checkpoint(ck_path, tag="")
			print(f'[loading checkpoint \'{ck_path}\']')
			self.epoch = ck['epoch']
			self.best_val_loss = ck["best_val_loss"]

	def save_ck(self,  is_valid=False, is_best=False):
		"""
		save training checkpoint
		"""

		ck_dir = "valid_ck" if is_valid else "training_ck"
		tag = "best" if is_best else ""

		ck = {
			'epoch': self.epoch + 1,
			'best_val_loss': self.best_val_loss,
		}

		self.model.save_checkpoint(self.log_path / ck_dir, client_state=ck,
								    tag=tag, save_latest=False)

		torch.cuda.synchronize()
		dist.barrier()

	#-------------------------
	# TRAINING
	#-------------------------

	def train_step(self, sample):
		"""
		Run a single training step
		"""

		loss = self.model.forward(sample)[0]
		self.model.backward(loss)

		self.model.step()
		return loss.item()

	def train(self):
		"""
		train model for one epoch on the Training-Set.
		"""
		start_time = time()
		self.model.train()

		times = []
		t = time()

		for step, sample in enumerate(self.train_loader):

			l = self.train_step(sample)

			if self.model.is_gradient_accumulation_boundary() and self.rank == 0:
				self.train_losses.append(l)

				# Display progress
				progress = (step + 1) / self.train_len
				progress_bar = ('█' * int(50 * progress)) + ('┈' * (50 - int(50 * progress)))
				times.append(time() - t)
				if self.cnf.log_each_step or (not self.cnf.log_each_step and progress == 1):
					print(
						'\r[{}] Epoch {:0{e}d}.{:0{s}d}: │{}│ {:6.2f}% │ Loss: {:.6f} │ ↯: {:5.2f} step/s'.format(
							datetime.now().strftime("%m-%d@%H:%M"), self.epoch, step + 1,
							progress_bar, 100 * progress,
							np.mean(self.train_losses), 1 / np.mean(times[-100:]),
							e=math.ceil(math.log10(self.cnf.epochs)),
							s=math.ceil(math.log10(self.train_len)),
																				), end='')
				t = time()

		if self.rank == 0:
			# log average loss of this epoch
			mean_epoch_loss = np.mean(self.train_losses)
			self.sw.add_scalar(tag='train_loss', scalar_value=mean_epoch_loss, global_step=self.epoch)
			self.train_losses = []

			# log epoch duration
			print(f' │ T: {time() - start_time:.2f} s')

	#-------------------------
	# VALIDATION
	#-------------------------

	def val_step(self, sample):
		"""
		Run a single validation step
		"""
		val_loss, _ = self.model.module.val_loss(sample)
		torch.cuda.synchronize()

		dist.all_reduce(val_loss)
		total_loss = val_loss / dist.get_world_size()
		total_loss = total_loss.mean().item()

		return total_loss, _

	def validate(self):
		"""
		Validate model on the Validation-Set
		"""

		self.model.eval()

		t = time()
		with torch.no_grad():
			for step, sample in enumerate(self.val_loader):
				val_loss, _ = self.val_step(sample)
				self.val_losses.append(val_loss)

		# log average loss on validation set
		mean_val_loss = np.mean(self.val_losses)
		self.val_losses = []
		print(f'\t● AVG Loss on VAL-set: {mean_val_loss:.6f} │ T: {time() - t:.2f} s')

		if self.rank == 0:
			self.sw.add_scalar(tag='val_loss', scalar_value=mean_val_loss, global_step=self.epoch)

		dist.barrier()

		# Save best validation result for finetuning
		if self.best_val_loss is None or mean_val_loss < self.best_val_loss:
			self.best_val_loss = mean_val_loss
			self.save_ck(is_best=True)

	#-------------------------
	# TESTING
	#-------------------------

	def test(self, modes=("val", ), load_best=False):

		print("[TRAINER]: Started test")
		model = self.model.module  # type: BaseModel

		if load_best:
			best_pth = self.log_path/'best.pth'
			if best_pth.exists():
				best = torch.load(best_pth, map_location={'cuda:%d' % 0: self.cnf.device})
				model.load_weights(best)
				print("[WARNING]: Loaded model's best checkpoint")
			else:
				print("[WARNING]: Best checkpoint does not exists, running on train checkpoint..")

		for mode in modes:
			ft_inference_on_test(self.cnf, self.val_set, model, mode, rank=self.rank)

	def run(self):
		"""
		start model training procedure (train > validation > checkpoint > repeat)
		"""

		# Store exact experiment configuration
		if self.rank == 0:
			save_git_stuff(self.cnf.exp_log_path)
			hparams_file = self.cnf.exp_log_path / "configuration.json"
			with open(hparams_file, "w+") as handle:
				json.dump(obj=self.cnf.y.todict(), fp=handle, indent=2)

		# Begin training loop
		for _ in range(self.epoch, self.cnf.epochs):

			# Single training epoch
			self.train()

			# if not self.train_all and (self.epoch % self.cnf.val_epoch_step == 0):
			if self.epoch % self.cnf.val_epoch_step == 0:
				self.validate()

			if self.epoch % self.cnf.ck_epoch_step == 0:
				self.save_ck()

			self.epoch += 1

		# --exp_name=paper_finetune/finetune_dct_32 --mode=test --conf_file_path=./log/DCT-Former/paper_finetune/finetune_dct_32.2022.2.16.10.34.25.sm4sx1zj
		dist.barrier()

		if self.cnf.rank == 1:
			print("[TRAINER]: Computing test result...")
			self.test()

		print("[TRAINER]: Train completed.")
