#!/usr/bin/env python
# Expects 10 command line parameters:
#
# train.data train.indices train.indptr train.y
# test.data test.indices test.indptr test.y
# json-output-file hyper-train-file
#
# Trains a neural net, logging to stdout, and saving results in a final json
# and hyper tuning in a pdf.
#
# Uses the env vars below, see repo README for details.

from multiprocessing import cpu_count

import os
from time import time
import warnings
warnings.filterwarnings(
    "ignore", message=r".*CUDA initialization: Found no NVIDIA driver.*")

MODELNAME = os.environ.get('MODELNAME')
DATASET = os.environ.get('DATASET')
ENCODING = os.environ.get('ENCODING')
TRUNCATE = os.environ.get('TRUNCATE')

import numpy as np
from scipy import sparse as sps

def binprefix(data, indices, indptr, y):
    files = [data, indices, indptr, y]
    dtypes = [np.uint32, np.uint32, np.uint64, np.uint32]
    data, indices, indptr, y = (
        np.fromfile(f, dtype=d) for f, d in zip(files, dtypes))
    return sps.csr_matrix((data, indices, indptr)), y

def rpad(X, col):
    assert X.getformat() == 'csr'
    return sps.csr_matrix((X.data, X.indices, X.indptr), shape=(X.shape[0], col))

from torchfm.model.afi import AutomaticFeatureInteractionModel
from torchfm.model.afm import AttentionalFactorizationMachineModel
from torchfm.model.dcn import DeepCrossNetworkModel
from torchfm.model.dfm import DeepFactorizationMachineModel
from torchfm.model.ffm import FieldAwareFactorizationMachineModel
from torchfm.model.fm import FactorizationMachineModel
from torchfm.model.fnfm import FieldAwareNeuralFactorizationMachineModel
from torchfm.model.fnn import FactorizationSupportedNeuralNetworkModel
from torchfm.model.lr import LogisticRegressionModel
from torchfm.model.ncf import NeuralCollaborativeFiltering
from torchfm.model.nfm import NeuralFactorizationMachineModel
from torchfm.model.pnn import ProductNeuralNetworkModel
from torchfm.model.wd import WideAndDeepModel
from torchfm.model.xdfm import ExtremeDeepFactorizationMachineModel
from torchfm.model.afn import AdaptiveFactorizationNetwork
from torchfm.model.hofm import HighOrderFactorizationMachineModel

def get_model(name, field_dims):
    """
    Parameter settings copied directly from pytorch-fm repo
    """
    if name == 'lr':
        return LogisticRegressionModel(field_dims)
    elif name == 'fm':
        return FactorizationMachineModel(field_dims, embed_dim=16)
    elif name == 'hofm':
        return HighOrderFactorizationMachineModel(field_dims, order=3, embed_dim=16)
    elif name == 'ffm':
        return FieldAwareFactorizationMachineModel(field_dims, embed_dim=4)
    elif name == 'fnn':
        return FactorizationSupportedNeuralNetworkModel(field_dims, embed_dim=16, mlp_dims=(16, 16), dropout=0.2)
    elif name == 'wd':
        return WideAndDeepModel(field_dims, embed_dim=16, mlp_dims=(16, 16), dropout=0.2)
    elif name == 'ipnn':
        return ProductNeuralNetworkModel(field_dims, embed_dim=16, mlp_dims=(16,), method='inner', dropout=0.2)
    elif name == 'opnn':
        return ProductNeuralNetworkModel(field_dims, embed_dim=16, mlp_dims=(16,), method='outer', dropout=0.2)
    elif name == 'dcn':
        return DeepCrossNetworkModel(field_dims, embed_dim=16, num_layers=3, mlp_dims=(16, 16), dropout=0.2)
    elif name == 'nfm':
        return NeuralFactorizationMachineModel(field_dims, embed_dim=64, mlp_dims=(64,), dropouts=(0.2, 0.2))
    elif name == 'ncf':
        raise ValueError('only movielens')
    elif name == 'fnfm':
        return FieldAwareNeuralFactorizationMachineModel(field_dims, embed_dim=4, mlp_dims=(64,), dropouts=(0.2, 0.2))
    elif name == 'dfm':
        return DeepFactorizationMachineModel(field_dims, embed_dim=16, mlp_dims=(16, 16), dropout=0.2)
    elif name == 'xdfm':
        return ExtremeDeepFactorizationMachineModel(
            field_dims, embed_dim=16, cross_layer_sizes=(16, 16), split_half=False, mlp_dims=(16, 16), dropout=0.2)
    elif name == 'afm':
        return AttentionalFactorizationMachineModel(field_dims, embed_dim=16, attn_size=16, dropouts=(0.2, 0.2))
    elif name == 'afi':
        return AutomaticFeatureInteractionModel(
             field_dims, embed_dim=16, atten_embed_dim=64, num_heads=2, num_layers=3, mlp_dims=(400, 400), dropouts=(0, 0, 0))
    elif name == 'afn':
        return AdaptiveFactorizationNetwork(
            field_dims, embed_dim=16, LNN_dim=1500, mlp_dims=(400, 400, 400), dropouts=(0, 0, 0))
    else:
        raise ValueError('unknown model name: ' + name)

import torch
from sklearn.metrics import roc_auc_score
import tqdm

from time import time

PEAK_MEM_KEYS = [
    "allocated_bytes.all.peak",
    "reserved_bytes.all.peak",
    "active_bytes.all.peak",
    "inactive_split_bytes.all.peak",
]

def train(model, optimizer, data_loader, criterion, device, epoch_name, disable_tqdm=False):
    model.train()
    total_loss = 0
    nex = 0
    if device.type == "cuda":
        torch.cuda.reset_max_memory_allocated(device)
    t = time()
    for i, (fields, target) in enumerate(tqdm.tqdm(data_loader, ncols=80, desc=epoch_name, leave=False, disable=(disable_tqdm or None))):
        fields, target = fields.to(device), target.to(device)
        y = model(fields)
        loss = criterion(y, target.float())
        model.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        nex += 1
    t = time() - t
    memstats = torch.cuda.memory_stats(device) if device.type == "cuda" else {}
    return total_loss / nex, t, memstats


from sklearn.metrics import log_loss, accuracy_score
from sklearn.metrics import roc_auc_score

def test(model, data_loader, device, disable_tqdm=False):
    model.eval()
    targets, predicts = list(), list()
    with torch.no_grad():
        for fields, target in tqdm.tqdm(data_loader, ncols=80, desc='eval', leave=False, disable=(disable_tqdm or None)):
            fields, target = fields.to(device), target.to(device)
            y = model(fields)
            targets.extend(target.tolist())
            predicts.extend(y.tolist())

    best_const = max(accuracy_score(targets, np.zeros(len(targets))), accuracy_score(targets, np.ones(len(targets))))

    return accuracy_score(targets, [x > 0.5 for x in predicts]), log_loss(targets, predicts), roc_auc_score(targets, predicts), best_const

import random, warnings

def seed_all(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    warnings.filterwarnings(
        "ignore", message=r".*CUDA initialization: Found no NVIDIA driver.*")

import ray

# if not set, creates a local ray
ray.init(address=os.environ.get('RAY_ADDRESS'))

LOCAL = not os.environ.get('RAY_ADDRESS')

import sys

json_out = sys.argv[9]
pdf_out = sys.argv[10]

t = time()
train_X, train_y = binprefix(*sys.argv[1:5])
t = time() - t
print('loaded train binary csr in {:7.4f} sec'.format(t))

t = time()
test_X, test_y = binprefix(*sys.argv[5:9])
t = time() - t
print('loaded test binary csr in {:7.4f} sec'.format(t))

NCOL = max(train_X.shape[1], test_X.shape[1])

train_X = rpad(train_X, NCOL)
test_X = rpad(test_X, NCOL)

import numpy as np

t = time()
Xm = train_X.max(axis=0).toarray()
tXm = test_X.max(axis=0).toarray()
field_dims = np.maximum(Xm.ravel(), tXm.ravel()) + 1
t = time() - t
print('computed field dims in {:7.4f} sec'.format(t))

from torch.utils.data import DataLoader
from torchfm.dataset.sps import SparseDataset
from ray import tune

# from .utils import get_model, train, test, PEAK_MEM_KEYS, seed_all
import torch

def train_wrapper(config, train_X=None, train_y=None, field_dims=None, epochs=None, batch_size=None):
    seed_all(1234)
    if LOCAL:
        ix = np.random.choice(train_X.shape[0], batch_size * 4)
        train_X = train_X[ix, :]
        train_y = train_y[ix]
    val_ix = train_X.shape[0] * 8 // 10

    train_dataset = SparseDataset(train_X[:val_ix, :], train_y[:val_ix], field_dims)
    val_dataset = SparseDataset(train_X[val_ix:, :], train_y[val_ix:], field_dims)
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0)
    val_data_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        torch.set_num_threads(cpu_count())
        device = torch.device("cpu")
    model = get_model(MODELNAME, field_dims).to(device)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=config["lr"], weight_decay=config["wd"])
    train_time = 0

    for epoch_i in range(epochs):
        epoch_name = 'epoch {:4d} of {}'.format(epoch_i + 1, epochs)
        loss, epoch_t, epoch_mem = train(model, optimizer, train_data_loader, criterion, device, epoch_name, disable_tqdm=True)
        train_time += epoch_t
        _, tloss, _, _ = test(model, val_data_loader, device, disable_tqdm=True)

        tune.report(train_loss=loss, val_loss=tloss, train_time=train_time, **{k: epoch_mem.get(k) for k in PEAK_MEM_KEYS})

seed_all(1234)
params = {
    "train_X": train_X,
    "train_y": train_y,
    "field_dims": field_dims,
    "epochs": 20,
    "batch_size": 256,
}
search_space = {
    "lr": tune.loguniform(1e-5, 1e-2),
    "wd": tune.loguniform(1e-5, 1e-2),
}
resources = {
    'cpu': cpu_count() if LOCAL else 0,
    'gpu': 0 if LOCAL else 1,
}

# try to resume first
for resume in [True, False]:
    try:
        analysis = tune.run(
            tune.with_parameters(train_wrapper, **params),
            config=search_space,
            num_samples=(4 if LOCAL else 10),
            name=f"{MODELNAME}-{DATASET}-{ENCODING}-{TRUNCATE}",
            resources_per_trial=resources,
            metric="val_loss",
            queue_trials=True,
            scheduler=tune.schedulers.ASHAScheduler(
                max_t=params["epochs"],
                grace_period=max(params["epochs"] // 5, 1)),
            resume=resume,
            local_dir="nn/data/ray_experiments",
            max_failures=2,
            mode="min")
        break
    except ValueError as e:
        if 'Called resume when no checkpoint exists in local directory.' in str(e):
            continue
        else:
            raise



BEST_CONFIG = dict(analysis.best_config)
print('best config', BEST_CONFIG)
BATCH_SIZE = params["batch_size"]
EPOCHS = params["epochs"]

from matplotlib import pyplot as plt

fig, axs = plt.subplots(2)
dfs = analysis.trial_dataframes
for d in dfs.values():
    d.train_loss.plot(ax=axs[0])
    d.val_loss.plot(ax=axs[1])
axs[0].set_title("hpo train loss")
axs[1].set_title("hpo val loss")
for ax in axs.flat:
    ax.set(xlabel='epoch', ylabel='log loss')
    ax.label_outer()
fig.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1))
plt.savefig(pdf_out,  bbox_inches='tight')

@ray.remote(num_cpus=resources['cpu'], num_gpus=resources['gpu'])
def retrain_and_test(datasets, field_dims):
    seed_all(1234)
    (train_X, train_y, test_X, test_y) = datasets
    if LOCAL:
        ix = np.random.choice(train_X.shape[0], BATCH_SIZE * 4)
        train_X = train_X[ix, :]
        train_y = train_y[ix]
        ix = np.random.choice(test_X.shape[0], BATCH_SIZE * 2)
        test_X = test_X[ix, :]
        test_y = test_y[ix]

    train_dataset = SparseDataset(train_X, train_y, field_dims)
    test_dataset = SparseDataset(test_X, test_y, field_dims)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        torch.set_num_threads(cpu_count())
        device = torch.device("cpu")

    emit = {}
    emit["num_train"] = len(train_dataset)
    emit["num_test"] = len(test_dataset)
    emit["modelname"] = MODELNAME
    emit["dataset"] = DATASET
    emit["encoding"] = ENCODING
    emit["device"] = str(device.type)
    emit["truncate"] = NCOL # == TRUNCATE unless 0, in which case == ncolors
    print(emit)

    train_data_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=0)
    test_data_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=0)
    model = get_model(MODELNAME, field_dims).to(device)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=BEST_CONFIG["lr"], weight_decay=BEST_CONFIG["wd"])

    train_time = 0
    train_mem = {}
    train_epoch_losses = []
    test_epoch_losses = [] # only for debug, note val used above for hpo

    for epoch_i in range(EPOCHS):
        epoch_name = 'epoch {:4d} of {}'.format(epoch_i + 1, EPOCHS)
        loss, epoch_t, epoch_mem = train(model, optimizer, train_data_loader, criterion, device, epoch_name, disable_tqdm=True)
        train_time += epoch_t
        train_mem = {k: max(train_mem.get(k, 0), epoch_mem.get(k, 0)) for k in PEAK_MEM_KEYS}
        _, tloss, _, _ = test(model, test_data_loader, device, disable_tqdm=True)
        print('epoch:', 1 + epoch_i, 'of', EPOCHS,
              'train: logloss: {:7.4f}'.format(loss),
              'val  : logloss: {:7.4f}'.format(tloss))
        sys.stdout.flush()

        train_epoch_losses.append(loss)
        test_epoch_losses.append(tloss)

    emit["train_epoch_logloss"] = train_epoch_losses
    emit["test_epoch_logloss"] = test_epoch_losses

    emit["train_acc"], emit["train_logloss"], emit["train_auc"], emit["train_acc_best_const"] = test(model, train_data_loader, device, disable_tqdm=True)
    print('train:', 'log: {:7.4f} acc: {:7.4f} auc: {:7.4f} acc best const: {:7.4f}'
          .format(emit["train_logloss"], emit["train_acc"], emit["train_auc"], emit["train_acc_best_const"]))
    sys.stdout.flush()

    emit["test_acc"], emit["test_logloss"], emit["test_auc"], emit["test_acc_best_const"] = test(model, test_data_loader, device, disable_tqdm=True)
    print('test :', 'log: {:7.4f} acc: {:7.4f} auc: {:7.4f} acc best const: {:7.4f}'
          .format(emit["test_logloss"], emit["test_acc"], emit["test_auc"], emit["test_acc_best_const"]))
    sys.stdout.flush()

    emit["train_sec"] = t
    emit['num_params'] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    for k in PEAK_MEM_KEYS:
        emit[k] = train_mem.get(k)

    return emit

datasets = (train_X, train_y, test_X, test_y)

emit = ray.get(retrain_and_test.remote(datasets, field_dims))

import json

with open(json_out, 'w', encoding='utf-8') as f:
    json.dump(emit, f, ensure_ascii=False, indent=4, sort_keys=True)
