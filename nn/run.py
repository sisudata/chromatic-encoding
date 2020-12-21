#!/usr/bin/env python
# Expects 9 command line parameters:
#
# train.data train.indices train.indptr train.y
# test.data test.indices test.indptr test.y
# json-output-file
#
# Trains a neural net, logging to stdout, and saving resutls in a final json
#
# Uses the env vars below for their obvious purposes.

import os
from time import time

MODELNAME = os.environ.get('MODELNAME')
DATASET = os.environ.get('DATASET')
ENCODING = os.environ.get('ENCODING')
CUDA = os.environ.get('CUDA_VISIBLE_DEVICES')


import sys
from .utils import binprefix

json_out = sys.argv[9]

t = time()
train_X, train_y = binprefix(*sys.argv[1:5])
t = time() - t
print('loaded train binary csr in {:7.4f} sec'.format(t))

t = time()
test_X, test_y = binprefix(*sys.argv[5:9])
t = time() - t
print('loaded test binary csr in {:7.4f} sec'.format(t))

ncol = max(train_X.shape[1], test_X.shape[1])

from .utils import rpad

train_X = rpad(train_X, ncol)
test_X = rpad(test_X, ncol)

import numpy as np

t = time()
Xm = train_X.max(axis=0).toarray()
tXm = test_X.max(axis=0).toarray()
field_dims = np.maximum(Xm.ravel(), tXm.ravel()) + 1
t = time() - t
print('computed field dims in {:7.4f} sec'.format(t))

import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
import tqdm

from torchfm.dataset.sps import SparseDataset

def train(model, optimizer, data_loader, criterion, device, epoch_name, disable_tqdm=False):
    model.train()
    total_loss = 0
    nex = 0
    t = time()
    for i, (fields, target) in enumerate(tqdm.tqdm(data_loader, desc=epoch_name, leave=False, disable=(disable_tqdm or None))):
        fields, target = fields.to(device), target.to(device)
        y = model(fields)
        loss = criterion(y, target.float())
        model.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        nex += 1
    t = time() - t
    return total_loss / nex, t


from sklearn.metrics import log_loss, accuracy_score
from sklearn.metrics import roc_auc_score

def test(model, data_loader, device, disable_tqdm=False):
    model.eval()
    targets, predicts = list(), list()
    with torch.no_grad():
        for fields, target in tqdm.tqdm(data_loader, desc='eval', leave=False, disable=(disable_tqdm or None)):
            fields, target = fields.to(device), target.to(device)
            y = model(fields)
            targets.extend(target.tolist())
            predicts.extend(y.tolist())

    best_const = max(accuracy_score(targets, np.zeros(len(targets))), accuracy_score(targets, np.ones(len(targets))))

    return accuracy_score(targets, [x > 0.5 for x in predicts]), log_loss(targets, predicts), roc_auc_score(targets, predicts), best_const

from .utils import get_model

def runall(train_dataset,
           val_dataset,
           test_dataset,
           device,
           epoch,
           learning_rate,
           batch_size,
           weight_decay,
           disable_tqdm=False):
    emit = {}

    emit["num_train"] = len(train_dataset)
    emit["num_val"] = len(val_dataset)
    emit["num_test"] = len(test_dataset)
    emit["modelname"] = MODELNAME
    emit["dataset"] = DATASET
    emit["encoding"] = ENCODING
    emit["device"] = device
    print(emit)

    device = torch.device(device)

    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0)
    val_data_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=0)
    model = get_model(MODELNAME, train_dataset).to(device)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    train_epoch_losses = []
    val_epoch_losses = []
    train_time = 0
    for epoch_i in range(epoch):
        epoch_name = 'epoch {:4d} of {}'.format(epoch_i + 1, epoch)
        loss, epoch_t = train(model, optimizer, train_data_loader, criterion, device, epoch_name, disable_tqdm)
        train_time += epoch_t
        _, vloss, _, _ = test(model, val_data_loader, device, disable_tqdm)
        print('epoch:', 1 + epoch_i, 'of', epoch,
              'train: logloss: {:7.4f}'.format(loss),
              'val  : logloss: {:7.4f}'.format(vloss))
        sys.stdout.flush()

        train_epoch_losses.append(loss)
        val_epoch_losses.append(vloss)

    emit["train_epoch_logloss"] = train_epoch_losses
    emit["val_epoch_logloss"] = val_epoch_losses

    emit["val_acc"], emit["val_logloss"], emit["val_auc"], emit["val_acc_best_const"] = test(model, val_data_loader, device, disable_tqdm)
    print('val   auc:', 'log: {:7.4f} acc: {:7.4f} auc: {:7.4f} acc best const: {:7.4f}'
          .format(emit["val_logloss"], emit["val_acc"], emit["val_auc"], emit["val_acc_best_const"]))
    sys.stdout.flush()

    emit["test_acc"], emit["test_logloss"], emit["test_auc"], emit["test_acc_best_const"] = test(model, test_data_loader, device, disable_tqdm)
    print('test  auc:', 'log: {:7.4f} acc: {:7.4f} auc: {:7.4f} acc best const: {:7.4f}'
          .format(emit["test_logloss"], emit["test_acc"], emit["test_auc"], emit["test_acc_best_const"]))
    sys.stdout.flush()
    emit["train_acc"], emit["train_logloss"], emit["train_auc"], emit["train_acc_best_const"] = test(model, train_data_loader, device, disable_tqdm)
    print('train auc:', 'log: {:7.4f} acc: {:7.4f} auc: {:7.4f} acc best const: {:7.4f}'
          .format(emit["train_logloss"], emit["train_acc"], emit["train_auc"], emit["train_acc_best_const"]))
    sys.stdout.flush()

    emit["train_sec"] = t
    emit['num_params'] = sum(p.numel() for p in model.parameters() if p.requires_grad)

    memstats = torch.cuda.memory_stats(device) if device.type == "cuda" else {}

    keys = [
        "allocated_bytes.all.peak",
        "reserved_bytes.all.peak",
        "active_bytes.all.peak",
        "inactive_split_bytes.all.peak",
    ]

    for k in keys:
        emit[k] = memstats.get(k)

    return emit

import warnings
warnings.filterwarnings(
    "ignore", message=r".*CUDA initialization: Found no NVIDIA driver.*")

if CUDA:
    device = "cuda" # max 1 gpu anyway
else:
    device = "cpu"

val_ix = train_X.shape[0] * 8 // 10
train_dataset = SparseDataset(train_X[:val_ix, :], train_y[:val_ix], field_dims)
val_dataset = SparseDataset(train_X[val_ix:, :], train_y[val_ix:], field_dims)
test_dataset = SparseDataset(test_X, test_y, field_dims)

emit = runall(
    train_dataset,
    val_dataset,
    test_dataset,
    device,
    # was 100, dropped to 10 to wait less, seemed reasonable given
    # no early stopping and LR tuning to achieve lower timings variance
    epoch=10,
    learning_rate=1e-3, # TODO tune this
    batch_size=2048,
    weight_decay=0.01, # TODO tune this
    disable_tqdm=False)


import json

with open(json_out, 'w', encoding='utf-8') as f:
    json.dump(emit, f, ensure_ascii=False, indent=4, sort_keys=True)
# TODO add auc
