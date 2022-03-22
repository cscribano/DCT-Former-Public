# -*- coding: utf-8 -*-
# ---------------------

import math
from datetime import datetime
from time import time
from timeit import default_timer as get_now
import json

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
import deepspeed as ds

from conf import Conf
from dataset import DistPreTrainingDataset, PretrainValidationDataset
from models import TrainableModel
from models.provider import  parse_model, parse_optimizer
from utils.misc import save_git_stuff, is_time_to_exit
from utils.schedulers import get_scheduler
from evaluation import pt_inference_on_test


class Trainer_PT(object):

	def __init__(self, cnf, rank):
		# type: (Conf, int) -> Trainer_PT

		self.cnf = cnf
		self.rank = rank
		self.epoch = 0
		self.log_path = cnf.exp_log_path
		self.cnf.experiment.exp_start_marker = get_now()

		# init model
		trainable, trainable_args = parse_model(cnf)

		# Instantiate trainable object wrapping DeepSpeed model
		model = trainable(self.cnf)  # type: TrainableModel

		# Retrieve optimizer, parameters and scheduler
		optimizer, parameters = parse_optimizer(cnf, model)
		lr_scheduler = get_scheduler(cnf, optimizer)

		self.model, self.optimizer, _, self.lr_scheduler = ds.initialize(config=cnf.deepspeed.todict(), model=model,
																	model_parameters=parameters,
																	optimizer=optimizer,
																	lr_scheduler=lr_scheduler)

		self.bs_per_gpu = self.model.train_micro_batch_size_per_gpu()
		self.fp16 = self.model.fp16_enabled()

		# Init dataloaders
		self.train_data_provider = DistPreTrainingDataset(cnf, self.model.train_micro_batch_size_per_gpu())
		self.valid_data_provider = PretrainValidationDataset(cnf)

		self.val_losses = []

		if self.rank == 0:
			# init logging stuffs
			print(f'tensorboard --logdir={cnf.project_log_path.abspath()}\n')
			self.sw = SummaryWriter(self.log_path)
			self.train_losses = []

		# State variables
		self.best_val_loss = None
		self.global_data_samples = 0
		self.global_steps = 0

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
			td = ck["residual_train_time"]
			self.cnf.experiment.exp_start_marker = get_now() - td
			self.global_steps = ck.get("last_global_steps", 0)
			self.global_data_samples = ck["global_data_samples"]


	def save_ck(self,  is_valid=False, is_last=False, is_best=False):
		"""
		save training checkpoint
		"""

		ck_dir = "valid_ck" if is_valid else "training_ck"
		tag = "best" if is_best else "last" if is_last else ""

		ck = {
			'epoch': self.epoch + 1,
			'best_val_loss': self.best_val_loss,
			'exp_start_marker': self.cnf.experiment.exp_start_marker,
			'residual_train_time': get_now() - self.cnf.experiment.exp_start_marker,
			'global_data_samples': self.global_data_samples,
			'last_global_steps': self.global_steps
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

		# Increment counter and prepare next batch
		self.global_data_samples += self.bs_per_gpu * dist.get_world_size()
		self.train_data_provider.prefetch_batch()

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
		self.scale_counter_at_1 = 0

		t = time()
		dataset_iterator, total_length = self.train_data_provider.get_shard(self.epoch)
		self.train_data_provider.prefetch_shard(self.epoch + 1)

		for step, batch_index in enumerate(dataset_iterator):

			"""
			if step >= len(dataset_iterator) - 1:
				# skip last batch
				continue
			"""

			try:
				sample = self.train_data_provider.get_batch(batch_index)
				l = self.train_step(sample)

				if self.model.is_gradient_accumulation_boundary():

					# Increment Update steps counter
					self.global_steps += 1

					# -----
					# HACK: add to scale counter if stuck at scale 1 (to detect possible NaN (diverged model))
					if self.fp16 and self.optimizer.cur_scale == 1:
						self.scale_counter_at_1 += 1
						print(f"[INFO] Optimizer scale=={self.scale_counter_at_1}")

					if self.scale_counter_at_1 >= 100:
						print("[WARNING] Optimizer scale==1 counter has been reached")
						del sample
						break
					# ----

					if self.rank == 0:
						self.train_losses.append(l)

						# Display progress
						progress = (step + 1) / len(dataset_iterator)
						progress_bar = ('█' * int(50 * progress)) + ('┈' * (50 - int(50 * progress)))
						times.append(time() - t)

						if self.cnf.log_each_step or (not self.cnf.log_each_step and progress == 1):
							print(
								'\r[{}] Epoch {:0{e}d}.{:0{s}d}: │{}│ {:6.2f}% │ Loss: {:.6f} │ ↯: {:5.2f} step/s'.format(
									datetime.now().strftime("%m-%d@%H:%M"), self.epoch, step + 1,
									progress_bar, 100 * progress,
									np.mean(self.train_losses), 1 / np.mean(times[-100:]),
									e=math.ceil(math.log10(self.cnf.epochs)),
									s=math.ceil(math.log10(len(dataset_iterator))),
																						), end='')
					t = time()

			except StopIteration:
				continue

			del sample

		torch.cuda.synchronize()
		dist.barrier(self.model.data_parallel_group)

		self.train_data_provider.release_shard(self.epoch)

		if self.rank == 0:
			# log average loss of this epoch
			mean_epoch_loss = np.mean(self.train_losses)
			self.sw.add_scalar(tag='train_loss', scalar_value=mean_epoch_loss, global_step=self.epoch)
			llr = self.lr_scheduler.get_last_lr()
			self.sw.add_scalar(tag='lr', scalar_value=llr[0] if type(llr) == list else llr, global_step=self.epoch)
			self.sw.add_scalar(tag='steps', scalar_value=self.global_steps, global_step=self.epoch)
			self.sw.add_scalar(tag='samples', scalar_value=self.global_data_samples, global_step=self.epoch)

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

	def validate(self, save=True):
		"""
		Validate model on the Validation-Set
		"""

		self.model.eval()

		val_dataset = self.valid_data_provider.get_validation_set(self.epoch)
		val_loader = DataLoader(val_dataset, sampler=DistributedSampler(val_dataset),
								**self.cnf.dataset.val_dataset.loader_args)

		t = time()
		with torch.no_grad():
			for step, sample in enumerate(val_loader):
				val_loss, _ = self.val_step(sample)
				self.val_losses.append(val_loss)

		# log average loss on validation set
		mean_val_loss = np.mean(self.val_losses)
		self.val_losses = []
		print(f'\t● AVG Loss on VAL-set: {mean_val_loss:.6f} │ T: {time() - t:.2f} s')

		if self.rank == 0 and save:
			self.sw.add_scalar(tag='val_loss', scalar_value=mean_val_loss, global_step=self.epoch)

		del val_loader
		del val_dataset
		dist.barrier()

		# Save best validation result for finetuning
		if save and (self.best_val_loss is None or mean_val_loss < self.best_val_loss):
			self.best_val_loss = mean_val_loss
			self.save_ck(is_best=True)

	def test(self, *args, **kwargs):

		print("[TRAINER]: Started test")
		model = self.model.module  # type: BaseModel
		val_dataset = self.valid_data_provider.get_validation_set(0)

		pt_inference_on_test(self.cnf, val_dataset, model, rank=self.rank)

	def run(self):
		"""
		start model training procedure (train > validation > checkpoint > repeat)
		"""

		# Store exact experiment configuration
		if self.rank == 0 and not self.cnf.resume:
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

			# Check for stopping conditions
			# TODO: if enough time passed, and the validation loss is not low enough, stop the run
			early_stop = self.scale_counter_at_1 >= 100
			done_training = is_time_to_exit(get_now(), self.global_steps, self.cnf)

			if early_stop or done_training:
				print("[TRAINER] Reached a stop condition")
				self.validate()
				self.save_ck(is_last=True)
				break

			self.epoch += 1
		# --exp_name=test --mode=test --conf_file_path=../LOGS/PAPER/pretrain_nystrom_32.2022.2.25.15.43.49.5wixa7a2
		dist.barrier()
		print("[TRAINER]: Train completed.")
