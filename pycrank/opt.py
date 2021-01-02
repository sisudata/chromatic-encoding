"""
Embedding-aware optimization loops that allow mixed sparse/dense.
"""

from time import time
import sys

from tqdm import tqdm
from sklearn.metrics import log_loss, accuracy_score
from sklearn.metrics import roc_auc_score
from scipy.special import expit
import torch
import numpy as np

from .utils import PEAK_MEM_KEYS
from .sparse_adam import SparseAdamAMS

def train(model, train_data_loader, val_data_loader, device, config, callback, tqdm=None):
    """
    Initialize a binary classification model with the specified name
    and train for the specified number of epochs.

    Call back with a dictionary containing various keys:

     * val_loss - validation log loss
     * train_loss - average epoch training loss
     * train_time - time spent on epoch
     * epoch_i - epoch index, 0-indexed.
     * pycrank.utils.PEAK_MEM_KEYS - various memory keys

    Expects configuration with hyperparameters:

     * epochs - number of epochs
     * lr - learning rate
     * wd - weight decay
    """
    criterion = torch.nn.BCEWithLogitsLoss()
    dense_optimizer = torch.optim.AdamW(
        [p for _, p in model.dense_parameters()], lr=config["lr"], weight_decay=config["wd"], amsgrad=True)
    sparse_optimizer = SparseAdamAMS(
        params=[p for _, p in model.sparse_parameters()], lr=config["lr"])

    for epoch_i in range(config["epochs"]):
        epoch_name = 'epoch {:2d} of {}'.format(epoch_i + 1, config["epochs"])
        tqdm_desc = '{} {}'.format(tqdm, epoch_name) if tqdm else None
        loss, epoch_t, epoch_mem = train_epoch(model, dense_optimizer, sparse_optimizer, train_data_loader, criterion, device, tqdm_desc)
        _, tloss, _, _ = test(model, val_data_loader, device)

        callback(dict(epoch_i=epoch_i, train_loss=loss, val_loss=tloss, train_time=epoch_t, **{k: epoch_mem.get(k) for k in PEAK_MEM_KEYS}))


def train_epoch(model, dense_optimizer, sparse_optimizer, data_loader, criterion, device, tqdm_desc):
    """
    Trains one epoch.

    Returns (average loss, train time, memory stats if cuda)
    """
    model.train()
    total_loss = 0
    nex = 0
    if device.type == "cuda":
        torch.cuda.reset_max_memory_allocated(device)
    t = time()
    for i, (fields, target) in enumerate(
            tqdm(data_loader,
                 ncols=80,
                 desc=(tqdm_desc or ''),
                 mininterval=(60 * 10),
                 leave=False, disable=(not tqdm_desc))):
        fields, target = fields.to(device), target.to(device)
        loss = criterion(model(fields), target.float())
        loss.backward()
        dense_optimizer.step()
        sparse_optimizer.step()
        dense_optimizer.zero_grad()
        sparse_optimizer.zero_grad()
        total_loss += loss.item()
        nex += 1
    t = time() - t
    memstats = torch.cuda.memory_stats(device) if device.type == "cuda" else {}
    return total_loss / nex, t, memstats


def test(model, data_loader, device):
    """
    Evaluates one pass through the given dat aloader.

    Returns (accuracy, log loss, auc, accuracy of best constant)
    Assumes model is already on device and data isn't.
    """
    model.eval()
    targets, predicts = list(), list()
    with torch.no_grad():
        for fields, target in data_loader:
            fields, target = fields.to(device), target.to(device)
            y = model(fields)
            targets.extend(target.tolist())
            predicts.extend(y.tolist())

    best_const = max(accuracy_score(targets, np.zeros(len(targets))), accuracy_score(targets, np.ones(len(targets))))

    return accuracy_score(targets, [x > 0 for x in predicts]), log_loss(targets, expit(predicts)), roc_auc_score(targets, predicts), best_const
