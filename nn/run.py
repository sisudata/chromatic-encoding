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

MODELNAME = os.environ.get('MODELNAME')
DATASET = os.environ.get('DATASET')
ENCODING = os.environ.get('ENCODING')
TRUNCATE = os.environ.get('TRUNCATE')

import numpy as np
from scipy import sparse as sps
import pycrank
import pycrank.utils
import pycrank.opt
import torch
from sklearn.metrics import roc_auc_score
from time import time
import random, warnings
import ray

# if not set, creates a local ray
ray.init(address=os.environ.get('RAY_ADDR'))

LOCAL = not os.environ.get('RAY_ADDR')

import sys

s3src = sys.argv[1]
json_out = sys.argv[2]
pdf_out = sys.argv[3]

import boto3
s = boto3.Session()
access_key = s.get_credentials().access_key
region_name = s.region_name or "us-west-2"
secret_key = s.get_credentials().secret_key
awscreds = f"AWS_ACCESS_KEY_ID={access_key}"
awscreds += f" AWS_DEFAULT_REGION={region_name}"
awscreds += f" AWS_SECRET_ACCESS_KEY={secret_key}"

import subprocess

etd = f"{ENCODING}_{TRUNCATE}_{DATASET}"
DET = f"{DATASET}.{ENCODING}.{TRUNCATE}"
CMD = f'''
{awscreds} aws s3 cp {s3src} {etd}.tar >/dev/null 2>/dev/null
tar xf {etd}.tar
rm {etd}.tar {etd}.{{train,test}}{{.original,}}.svm.*.zst
zstd -f -d -q --rm {DET}.bin.tar.zst
tar xf {DET}.bin.tar
rm {DET}.bin.tar
rm {DET}.jsonl
'''


def mkdata(datadir):
    subprocess.check_call(['/bin/bash', '-c', CMD], cwd=datadir)
    sparse_suffixes = ['data','indices','indptr','y']
    with pycrank.utils.timeit('loading train csr'):
        train_X, train_y = pycrank.utils.load_binary_csr(
            *[f"{datadir}/{DET}.train.{x}" for x in sparse_suffixes])

    with pycrank.utils.timeit('loading test csr'):
        test_X, test_y = pycrank.utils.load_binary_csr(
            *[f"{datadir}/{DET}.test.{x}" for x in sparse_suffixes])

    ncol = max(train_X.shape[1], test_X.shape[1])
    train_X = pycrank.utils.rpad(train_X, ncol)
    test_X = pycrank.utils.rpad(test_X, ncol)

    with pycrank.utils.timeit('computing field dims'):
        Xm = train_X.max(axis=0).toarray()
        tXm = test_X.max(axis=0).toarray()
        field_dims = np.maximum(Xm.ravel(), tXm.ravel())

    datasets = (train_X, train_y, test_X, test_y, field_dims, ncol)
    return datasets

from torch.utils.data import DataLoader
from pycrank.sps import SparseDataset
from ray import tune
import tempfile

def mktmp():
    from pathlib import Path
    home = str(Path.home())
    return tempfile.TemporaryDirectory(dir=home, prefix='ray-job-')

def train_wrapper(config, epochs=None, batch_size=None):
    with mktmp() as tmpdir:
        (train_X, train_y, _, _, field_dims, _) = mkdata(tmpdir)
        pycrank.utils.seed_all(1234)
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

        model = pycrank.utils.default_model(MODELNAME, field_dims).to(device)
        config["epochs"] = epochs
        pycrank.opt.train(
            model, train_data_loader, val_data_loader, device, config, lambda d: tune.report(**d), f'{DATASET} {ENCODING} {TRUNCATE} hpo')


pycrank.utils.seed_all(1234)
params = {
    "epochs": 10,
    "batch_size": 2048,
}
search_space = {
    "lr": tune.loguniform(1e-5, 1e-2),
    "wd": tune.loguniform(1e-5, 1e-2),
} # keys expected by pycrank.opt.train
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
            num_samples=12,
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
    except ValueError as exc:
        if 'Called resume when no checkpoint exists in local directory.' in str(exc):
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
configs = analysis.get_all_configs()
for trial_key, trial_df in dfs.items():
    if len(trial_df) == 0:
        continue
    label = ' '.join('{}={}'.format(k, configs[trial_key][k]) for k in search_space)
    trial_df.train_loss.plot(ax=axs[0], label='train {}'.format(label))
    trial_df.val_loss.plot(ax=axs[1], label='val {}'.format(label))
axs[0].set_title("hpo train loss")
axs[1].set_title("hpo val loss")
for ax in axs.flat:
    ax.set(xlabel='epoch', ylabel='log loss')
    ax.label_outer()
    ax.set(yscale="log")
fig.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1))
plt.savefig(pdf_out,  bbox_inches='tight')

@ray.remote(num_cpus=resources['cpu'], num_gpus=resources['gpu'])
def retrain_and_test():
    with mktmp() as tmpdir:
        (train_X, train_y, test_X, test_y, field_dims, ncol) = mkdata(tmpdir)

        pycrank.utils.seed_all(1234)
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
        emit["truncate"] = ncol # == TRUNCATE unless 0, in which case == ncolors
        print(emit)

        train_data_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=0)
        test_data_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=0)
        model = pycrank.utils.default_model(MODELNAME, field_dims).to(device)

        BEST_CONFIG["epochs"] = EPOCHS
        train_time = 0
        train_mem = {}
        train_epoch_losses = []
        test_epoch_losses = [] # only for debug, note val used above for hpo
        def callback(d):
            nonlocal train_time, train_mem, train_epoch_losses, test_epoch_losses
            train_time += d["train_time"]
            for k in pycrank.utils.PEAK_MEM_KEYS:
                train_mem[k] = max(train_mem.get(k) or 0, d.get(k) or 0)
            print('epoch:', 1 + d["epoch_i"], 'of', EPOCHS,
                  'train: logloss: {:7.4f}'.format(d["train_loss"]),
                  'val  : logloss: {:7.4f}'.format(d["val_loss"]))
            sys.stdout.flush()
            train_epoch_losses.append(d["train_loss"])
            test_epoch_losses.append(d["val_loss"])
        pycrank.opt.train(model, train_data_loader, test_data_loader, device, BEST_CONFIG, callback, f'final {DATASET} {ENCODING} {TRUNCATE}')

        emit["config"] = BEST_CONFIG
        emit["train_epoch_logloss"] = train_epoch_losses
        emit["test_epoch_logloss"] = test_epoch_losses

        emit["train_acc"], emit["train_logloss"], emit["train_auc"], emit["train_acc_best_const"] = pycrank.opt.test(model, train_data_loader, device)
        print('train:', 'log: {:7.4f} acc: {:7.4f} auc: {:7.4f} acc best const: {:7.4f}'
              .format(emit["train_logloss"], emit["train_acc"], emit["train_auc"], emit["train_acc_best_const"]))
        sys.stdout.flush()

        emit["test_acc"], emit["test_logloss"], emit["test_auc"], emit["test_acc_best_const"] = pycrank.opt.test(model, test_data_loader, device)
        print('test :', 'log: {:7.4f} acc: {:7.4f} auc: {:7.4f} acc best const: {:7.4f}'
              .format(emit["test_logloss"], emit["test_acc"], emit["test_auc"], emit["test_acc_best_const"]))
        sys.stdout.flush()

        emit["train_sec"] = train_time
        emit['num_params'] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        for k in pycrank.utils.PEAK_MEM_KEYS:
            emit[k] = train_mem.get(k)

        return emit

emit = ray.get(retrain_and_test.remote())

import json

with open(json_out, 'w', encoding='utf-8') as f:
    json.dump(emit, f, ensure_ascii=False, indent=4, sort_keys=True)
