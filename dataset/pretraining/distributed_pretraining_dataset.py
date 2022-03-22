# coding=utf-8
# Copyright 2021 Intel Corporation. All rights reserved.
# code taken from commit: 35b4582486fe096a5c669b6ca35a3d5c6a1ec56b
# https://github.com/microsoft/DeepSpeedExamples/tree/master/bing_bert
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import random
from dataset.pretraining.bert_dataset_provider import BertDatasetProviderInterface
from concurrent.futures import ProcessPoolExecutor
from enum import IntEnum

import h5py
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler

from conf import Conf


class BatchType(IntEnum):
    RANKING_BATCH = 0
    QP_BATCH = 1
    PRETRAIN_BATCH = 2


def torch_long(x):
    return torch.LongTensor(x)


def map_to_torch(encoding):
    encoding = torch_long(encoding)
    encoding.requires_grad_(False)
    return encoding


# Workaround because python functions are not picklable
class WorkerInitObj(object):
    def __init__(self, seed):
        self.seed = seed

    def __call__(self, id):
        np.random.seed(seed=self.seed + id)
        random.seed(self.seed + id)


def create_pretraining_dataset(
    input_file,
    max_predictions_per_seq,
    num_workers,
    train_batch_size,
    worker_init,
    data_sampler,
    no_nsp=False,
    num_replicas=1,
    rank=0,
    epoch=0,
):
    train_data = pretraining_dataset(
        input_file=input_file, max_predictions_per_seq=max_predictions_per_seq, no_nsp=no_nsp
    )
    sampler_instance = data_sampler(
        train_data, num_replicas=num_replicas, rank=rank, drop_last=True
    )
    sampler_instance.set_epoch(epoch)
    train_dataloader = DataLoader(
        train_data,
        sampler=sampler_instance,
        batch_size=train_batch_size,
        num_workers=num_workers,
        worker_init_fn=worker_init,
        pin_memory=True,
    )
    return train_dataloader, len(train_data)


class pretraining_dataset(Dataset):
    def __init__(self, input_file, max_predictions_per_seq, no_nsp=False):
        self.input_file = input_file
        self.max_predictions_per_seq = max_predictions_per_seq
        f = h5py.File(input_file, "r")
        keys = [
            "input_ids",
            "input_mask",
            "segment_ids",
            "masked_lm_positions",
            "masked_lm_ids",
            "next_sentence_labels",
        ]
        self.no_nsp = no_nsp
        if no_nsp:
            keys.remove("next_sentence_labels")
        self.inputs = [np.asarray(f[key][:]) for key in keys]
        f.close()

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.inputs[0])

    def __getitem__(self, index):
        [input_ids, input_mask, segment_ids, masked_lm_positions, masked_lm_ids] = [
            torch.from_numpy(input[index].astype(np.int64))
            for _, input in enumerate(self.inputs[:5])
        ]
        masked_lm_labels = torch.ones(input_ids.shape, dtype=torch.long) * -1
        index = self.max_predictions_per_seq
        # store number of  masked tokens in index
        padded_mask_indices = torch.nonzero((masked_lm_positions == 0), as_tuple=False)
        if len(padded_mask_indices) != 0:
            index = padded_mask_indices[0].item()
        masked_lm_labels[masked_lm_positions[:index]] = masked_lm_ids[:index]

        if self.no_nsp:
            return [
                map_to_torch([BatchType.PRETRAIN_BATCH]),
                input_ids,
                input_mask,
                segment_ids,
                masked_lm_labels,
            ]
        else:
            next_sentence_labels = torch.from_numpy(
                np.asarray(self.inputs[-1][index].astype(np.int64))
            )
            return [
                map_to_torch([BatchType.PRETRAIN_BATCH]),
                input_ids,
                input_mask,
                segment_ids,
                next_sentence_labels,
                masked_lm_labels,
            ]


class ValidationDataset:
    def __init__(self, args):
        if args.local_rank == -1:
            self.global_rank = 0
            self.world_size = 1
        else:
            self.global_rank = dist.get_rank()
            self.world_size = dist.get_world_size()

        # Initialize dataset files
        dataset_path = args.dataset_path
        self.dataset_files = [
            os.path.join(dataset_path, f)
            for f in os.listdir(dataset_path)
            if os.path.isfile(os.path.join(dataset_path, f)) and "test" in f
        ]
        assert (
            len(self.dataset_files) > 0
        ), "No validation files found, make sure *valid_*.hdf5 file exist in dataset path"
        self.dataset_files.sort()
        self.num_files = len(self.dataset_files)
        if self.global_rank == 0:
            print(f"[INFO] ValidationDataset - Initialization:  num_files = {self.num_files}")
        self.max_predictions_per_seq = args.max_predictions_per_seq
        self.no_nsp = args.no_nsp

    def get_validation_set(self, index):
        file_index = index % self.num_files
        input_file = self.dataset_files[file_index]
        validation_data = pretraining_dataset(
            input_file=input_file,
            max_predictions_per_seq=self.max_predictions_per_seq,
            no_nsp=self.no_nsp,
        )
        print(f"[INFO] ValidationDataset - shard {file_index} - length {len(validation_data)}")
        return validation_data


class PreTrainingDataset(BertDatasetProviderInterface):
    def __init__(self, cnf, bs_per_gpu, data_prefix="train"):
        # type: (Conf, int, str) -> None

        self.num_workers = cnf.dataset.num_workers
        self.max_predictions_per_seq = cnf.experiment.max_predictions_per_seq
        assert data_prefix in ["train", "test"], "data_prefix must be [train|test]"

        self.gradient_accumulation_steps = cnf.deepspeed.gradient_accumulation_steps
        self.train_micro_batch_size_per_gpu = bs_per_gpu #args.train_micro_batch_size_per_gpu

        if cnf.rank == -1:
            self.global_rank = 0
            self.world_size = 1
        else:
            self.global_rank = dist.get_rank()
            self.world_size = dist.get_world_size()

        # Initialize dataset files
        dataset_path = cnf.dataset.data_root
        self.dataset_files = [
            os.path.join(dataset_path, f)
            for f in os.listdir(dataset_path)
            if os.path.isfile(os.path.join(dataset_path, f)) and data_prefix in f
        ]
        assert len(self.dataset_files) > 0, "No train/test*.hdf5 files found in given dataset path"
        if data_prefix == "train":
            self.dataset_files.sort()
        random.shuffle(self.dataset_files)
        self.num_files = len(self.dataset_files)
        self.data_sampler = RandomSampler if cnf.rank == -1 else DistributedSampler

        self.worker_init = WorkerInitObj(cnf.seed + cnf.local_rank)
        self.dataset_future = None
        self.pool = ProcessPoolExecutor(1)

        if self.global_rank == 0:
            print(f"[INFO] PreTrainingDataset - Initialization:  num_files = {self.num_files}")
        self.no_nsp = True #args.no_nsp

    def get_shard(self, epoch):
        if self.dataset_future is None:
            data_file = self._get_shard_file(epoch)
            self.train_dataloader, sample_count = create_pretraining_dataset(
                input_file=data_file,
                max_predictions_per_seq=self.max_predictions_per_seq,
                num_workers=self.num_workers,
                train_batch_size=self.train_micro_batch_size_per_gpu,
                worker_init=self.worker_init,
                data_sampler=self.data_sampler,
                no_nsp=self.no_nsp,
                num_replicas=self.world_size,
                rank=self.global_rank,
                epoch=epoch,
            )
        else:
            self.train_dataloader, sample_count = self.dataset_future.result(timeout=None)

        return self.train_dataloader, sample_count

    def release_shard(self, epoch):
        del self.train_dataloader

    def prefetch_shard(self, epoch):
        data_file = self._get_shard_file(epoch)
        self.dataset_future = self.pool.submit(
            create_pretraining_dataset,
            data_file,
            self.max_predictions_per_seq,
            self.num_workers,
            self.train_micro_batch_size_per_gpu,
            self.worker_init,
            self.data_sampler,
            self.no_nsp,
            self.world_size,
            self.global_rank,
            epoch,
        )

    def get_batch(self, batch_iter):
        return batch_iter

    def prefetch_batch(self):
        pass

    def _get_shard_file(self, shard_index):
        file_index = self._get_shard_file_index(shard_index, self.global_rank)
        return self.dataset_files[file_index % self.num_files]

    def _get_shard_file_index(self, shard_index, global_rank):
        global_rank = 0
        if dist.is_initialized() and self.world_size > self.num_files:
            remainder = self.world_size % self.num_files
            file_index = (shard_index * self.world_size) + global_rank + (remainder * shard_index)
        else:
            file_index = shard_index * self.world_size + global_rank

        return file_index % self.num_files

if __name__ == '__main__':
    cnf = Conf(exp_name='pretrain_tiny')

    # Init dist stuff (required)
    if not 'MASTER_ADDR' in os.environ:
        os.environ['MASTER_ADDR'] = Conf.HOSTNAME
        os.environ['MASTER_PORT'] = str(Conf.PORT)

    dist.init_process_group('nccl', world_size=1, rank=0)

    # Dataloader
    index = 0 # Epoch counter

    pretrain_dataset_provider = PreTrainingDataset(cnf, 4)
    dataset_iterator, total_length = pretrain_dataset_provider.get_shard(index)

    pretrain_dataset_provider.prefetch_shard(index + 1)

    for batch_index_number, batch_index in enumerate(dataset_iterator):
        try:
            batch = pretrain_dataset_provider.get_batch(batch_index)

            input_ids = batch[1]
            attention_mask = batch[2]
            token_type_ids = batch[3]
            masked_lm_labels = batch[4]

            torch.save(pretrain_dataset_provider.get_batch(batch_index), "../../evaluation/sample.pt")
            break

        except StopIteration:
            pass

    print("Finished")

