# -*- coding: utf-8 -*-
# ---------------------

import os

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score
import pandas as pnd

from conf import Conf
from dataset import collate_fn_pad
from models import BaseModel
from models.provider import parse_dataset

def pt_inference_on_test(cnf, dataset, model, rank=0):
    # type: (Conf, Dataset, BaseModel, int) -> None

    """
    Run inference on the test/validation set
    :param dataset: a Dataset object that inherits from TestableModel
    :param model: object of a class that inherit from TestDataset
    :return: None
    """

    assert cnf.world_size == 1, "Distributed testing is not supported"
    model.eval()

    # Create dataloader object
    dataloader = DataLoader(
        dataset=dataset, batch_size=256, num_workers=8, shuffle=False
    )

    preds = []
    targets = []
    for index, sample in tqdm(enumerate(dataloader), total=len(dataloader)):

        with torch.no_grad():
            y_pred, y_true = model.test_step(sample)
            preds.append(y_pred.argmax(dim=-1).cpu().numpy())
            targets.append(y_true.cpu().numpy())

    targets = np.concatenate(targets)
    preds = np.concatenate(preds)

    print(accuracy_score(targets, preds))


def ft_inference_on_test(cnf, dataset, model, mode="test", rank=0):
    # type: (Conf, Dataset, BaseModel, str, int) -> None

    """
    Run inference on the test/validation set
    :param dataset: a Dataset object that inherits from TestableModel
    :param model: object of a class that inherit from TestDataset
    :return: None
    """

    #assert cnf.world_size == 1, "Distributed testing is not supported"
    _, _, collate_fn = parse_dataset(cnf)
    model.eval()

    # Create dataloader object
    dataloader = DataLoader(
        dataset=dataset, batch_size=64, num_workers=8, shuffle=False, collate_fn=collate_fn
    )

    preds = []
    targets = []
    for index, sample in tqdm(enumerate(dataloader), total=len(dataloader)):
        with torch.no_grad():
            y_pred = model.test_step(sample)[0]
            pred = torch.sigmoid(y_pred[:, 0]) > 0.5
            preds.append(pred.cpu().numpy())
            targets.append(sample[1].cpu().numpy())

    targets = np.concatenate(targets)
    preds = np.concatenate(preds).astype(np.int64)

    print(classification_report(targets, preds))

    """
    # Save experiment results
    cls = classification_report(targets, preds, labels=[0, 1], output_dict=True)
    cls={f"{k2}_[{k1}]" : [cls[k1][k2]] for k1 in cls.keys() for k2 in cls[k1].keys()}
    cls["Experiment"] = cnf.exp_name

    df = pnd.DataFrame(cls)

    results_file = os.path.join(cnf.project_log_path, "all_experiments.csv")
    if not os.path.isfile(results_file):
        df.to_csv(results_file, mode='a', index=True, header=list(df.columns))
    else:
        df.to_csv(results_file, mode='a', index=True, header=None)
    """


