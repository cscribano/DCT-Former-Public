# -*- coding: utf-8 -*-
# ---------------------

import os
import glob
from typing import *

import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
from torch.nn.utils.rnn import pad_sequence

from conf import Conf
from dataset.utils import denoise_text


class ImDb_sentiment(Dataset):

    def __init__(self, cnf, mode='train', precomputed=True):
        # type: (Conf, str, bool) -> None
        """
        param cnf: configuration object
        param mode: dataset split, train or test
        param precomputed: Return result of tokenizer computed offline
        """
        self.cnf = cnf
        self.mode = mode
        self.precomputed = precomputed

        sentiments = ['pos', 'neg']
        assert mode in ["train", "test"]

        data_root = cnf.dataset.data_root
        split_directory = os.path.join(data_root, mode)

        # https://pytorchnlp.readthedocs.io/en/latest/_modules/torchnlp/datasets/imdb.html#imdb_dataset
        samples = []
        labels = []
        for i, sentiment in enumerate(sentiments):
            for filename in glob.iglob(os.path.join(split_directory, sentiment, '*.txt')):
                with open(filename, 'r', encoding="utf-8") as f:
                    text = f.readline()

                samples.append(denoise_text(text))
                labels.append(i)

        self.labels = torch.tensor(labels, dtype=torch.float32).reshape(-1, 1)
        self.samples = samples

        if precomputed:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower=True)

    def __len__(self):
        # type: () -> int
        return len(self.labels)

    def __getitem__(self, i):
        # type: (int) -> Tuple[str, torch.Tensor]

        x = self.samples[i]
        y = self.labels[i]

        if self.precomputed:
            tokens = self.tokenizer.encode_plus(x, padding=True, truncation=True,
                                                       max_length=self.cnf.dataset.max_seq_len, return_tensors="pt")

            input_ids = tokens['input_ids']
            attention_mask = tokens['attention_mask']
            token_type_ids = tokens['token_type_ids']

            x = [input_ids, attention_mask, token_type_ids]

        return x, y

def collate_fn_pad(samples):
    x = pad_sequence([torch.stack(s[0], dim=-1)[0] for s in samples]).transpose(0,2)
    # pad = 1024 - x.shape[-1]
    # x = torch.nn.ConstantPad1d((0, pad), 0)(x)
    y = torch.stack([s[1] for s in samples])
    return (x,y)

def collate_fn_pad_fixed(samples):
    x = pad_sequence([torch.stack(s[0], dim=-1)[0] for s in samples]).transpose(0,2)
    pad = 1024 - x.shape[-1]
    x = torch.nn.ConstantPad1d((0, pad), 0)(x)
    y = torch.stack([s[1] for s in samples])
    return (x,y)

if __name__ == '__main__':

    from torch.utils.data import DataLoader
    import numpy as np
    from tqdm import tqdm

    cnf = Conf(exp_name='armstrong/small_bertogna_imdb')
    ds = ImDb_sentiment(cnf, **cnf.dataset.train_dataset.args)

    loader = DataLoader(
        dataset=ds, batch_size=1, num_workers=1, collate_fn=collate_fn_pad
    )

    lenghts = []
    for i in tqdm(ds):
        x, _ = i
        lenghts.append(x[0].shape[-1])

    lenghts = np.array(lenghts)
    print(lenghts.mean())
    print(min(lenghts), max(lenghts))
