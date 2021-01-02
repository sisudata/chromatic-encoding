#!/usr/bin/env python
# Expects 9 command line parameters:
#
# train.data train.indices train.indptr train.y
# test.data test.indices test.indptr test.y
# json-output-file
#
# Uses the env vars below, see repo README for details.

from multiprocessing import cpu_count

import os, sys
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

s3src = sys.argv[1]
json_out = sys.argv[2]

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
set -e
{awscreds} aws s3 cp {s3src} {etd}.tar >/dev/null 2>/dev/null
tar xf {etd}.tar
rm {etd}.tar {etd}.{{train,test}}{{.original,}}.svm.*.zst
zstd -f -d -q --rm {DET}.bin.tar.zst
tar xf {DET}.bin.tar
rm {DET}.bin.tar
rm {DET}.jsonl
'''


def mkdata(datadir):
    try:
        subprocess.run(
            ['/bin/bash', '-c', CMD],
            universal_newlines=True,
            capture_output=True,
            check=True,
            cwd=datadir)
    except subprocess.CalledProcessError as e:
        raise ValueError('stdout:\n{}\nstderr:\n{}\n'.format(e.stdout, e.stderr)) from e
    sparse_suffixes = ['data','indices','indptr','y']
    try:
        with pycrank.utils.timeit('loading train csr'):
            train_X, train_y = pycrank.utils.load_binary_csr(
                *[f"{datadir}/{DET}.train.{x}" for x in sparse_suffixes])

        with pycrank.utils.timeit('loading test csr'):
            test_X, test_y = pycrank.utils.load_binary_csr(
                *[f"{datadir}/{DET}.test.{x}" for x in sparse_suffixes])
    except Exception as exc:
        raise ValueError(
            'datadir: {} contents: {}'.format(
                datadir, os.listdir(datadir))) from exc
    ncol = max(train_X.shape[1], test_X.shape[1])
    train_X = pycrank.utils.rpad(train_X, ncol)
    test_X = pycrank.utils.rpad(test_X, ncol)

    with pycrank.utils.timeit('computing field dims'):
        Xm = train_X.max(axis=0).toarray()
        tXm = test_X.max(axis=0).toarray()
        field_dims = np.maximum(Xm.ravel(), tXm.ravel())

    with pycrank.utils.timeit('offset uniq-ing'):
        # for column c, offsets[c] is the total number of features from
        # all earlier columns. So offsets[c] + 1-indexed data of column c
        # is a unique feature value in [1, total num features] and the c-th
        # column occupies range [1 + offsets[c], field_dims[c] + offsets[c]]
        # zeros must never appear in the underlying CSR arrays so they are not
        # represented except as sentinels for the entire array, e.g., the first
        # 0.
        offsets = np.insert(np.cumsum(field_dims, dtype=np.uint32), 0, 0)
        # uniqued contains the original data in the specified ranges
        uniqued = offsets[train_X.indices] + train_X.data
        uniqued_test = offsets[test_X.indices] + test_X.data

    with pycrank.utils.timeit('bincount'):
        counts = np.bincount(uniqued)
        assert counts[0] == 0, counts[0] # everything was 1-indexed

    with pycrank.utils.timeit('binsort'):
        tops = np.argsort(counts)[::-1]

    ncut = np.sum(counts >= 2)
    winners = tops[:ncut]
    cut = counts[winners[-1]]
    print('freq cutoff', ncut, 'at count',
          cut, 'of', train_X.shape[0], file=sys.stderr)

    with pycrank.utils.timeit('mask compute'):
        keep = np.zeros(len(counts), dtype=bool)
        keep[winners] = True
        excl0 = keep[0]
        keep[0] = False # in case len(tops) < ncut, 0 feature might be included
        keep_uniqued = keep[uniqued]
        keep_uniqued_test = keep[uniqued_test]

    with pycrank.utils.timeit('remap'):
        # find new offsets such that the retained features can be remapped to
        # compact intervals.
        # uniqued feature f -> monotone_features[f]
        # is the new 1-indexed feature value (or the most recent on of lower index).
        monotone_features = np.cumsum(keep, dtype=np.uint32)
        assert monotone_features[0] == 0
        assert monotone_features[-1] == len(winners) - excl0
        # recall the offset <-> uniqued relationship
        winner_features_per_column = np.diff(monotone_features[offsets]) #[offsets[1:]] - monotone_features[offsets[:-1] + 1] - keep[offsets[:-1] + 1]
        base_col = np.cumsum(winner_features_per_column)
        base_col = np.roll(base_col, 1)
        assert base_col[0] == len(winners) - excl0, (base_col[0], len(winners) - excl0)
        base_col[0] = 0
        base_feature = np.repeat(base_col, np.diff(offsets))
        # note counts has a dummy 0 feature
        assert offsets[-1] + 1 == len(counts)
        assert len(base_feature) + 1 == len(counts)
        monotone_features[1:] -= base_feature # still 1-indexed
        remap = monotone_features

    with pycrank.utils.timeit('new indptr'):
        new_indptr = np.insert(np.cumsum(keep_uniqued, dtype=np.uint64), 0, 0)
        new_indptr = new_indptr[train_X.indptr]
        new_indptr_test = np.insert(np.cumsum(keep_uniqued_test, dtype=np.uint64), 0, 0)
        new_indptr_test = new_indptr_test[test_X.indptr]

    with pycrank.utils.timeit('new mats'):
        new_indices = train_X.indices[keep_uniqued]
        new_data = remap[uniqued[keep_uniqued]]
        new_indices_test = test_X.indices[keep_uniqued_test]
        new_data_test = remap[uniqued_test[keep_uniqued_test]]

    train_X = sps.csr_matrix((new_data, new_indices, new_indptr), shape=train_X.shape)
    test_X = sps.csr_matrix((new_data_test, new_indices_test, new_indptr_test), shape=test_X.shape)

    with pycrank.utils.timeit('computing trunc field dims'):
        Xm = train_X.max(axis=0).toarray()
        tXm = test_X.max(axis=0).toarray()
        field_dims = np.maximum(Xm.ravel(), tXm.ravel())
        assert np.all(field_dims == winner_features_per_column), (
            np.sum(field_dims != winner_features_per_column),
            field_dims[field_dims != winner_features_per_column][:10],
            winner_features_per_column[field_dims != winner_features_per_column][:10],
            np.flatnonzero(field_dims != winner_features_per_column)[:10])


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

def mkmodel(field_dims):
    if MODELNAME == 'wd':
        from pycrank.wd import WideAndDeepModel
        return WideAndDeepModel(
            field_dims,
            embed_dim=32,
            mlp_dims=(128, 128),
            dropout=0.5)
    raise ValueError('unknown model name: ' + MODELNAME)

pycrank.utils.seed_all(1234)
params = {
    "epochs": 10,
    "batch_size": 2048,
    "lr": 1e-2,
    "wd": 1e-4,
}
resources = {
    'cpu': cpu_count() if LOCAL else 0,
    'gpu': 0 if LOCAL else 1,
}

BATCH_SIZE = params["batch_size"]
EPOCHS = params["epochs"]

@ray.remote(num_cpus=resources['cpu'], num_gpus=resources['gpu'])
def retrain_and_test():
    with mktmp() as tmpdir:
        (train_X, train_y, test_X, test_y, field_dims, ncol) = mkdata(tmpdir)

        pycrank.utils.seed_all(1234)
        if LOCAL:
            ix = np.random.choice(train_X.shape[0], params["batch_size"] * 4)
            train_X = train_X[ix, :]
            train_y = train_y[ix]
            ix = np.random.choice(test_X.shape[0], params["batch_size"] * 2)
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

        train_data_loader = DataLoader(train_dataset, batch_size=params["batch_size"], num_workers=0)
        test_data_loader = DataLoader(test_dataset, batch_size=params["batch_size"], num_workers=0)
        model = mkmodel(field_dims).to(device)

        train_time = 0
        train_mem = {}
        train_epoch_losses = []
        test_epoch_losses = [] # only for debug, note val used above for hpo
        def callback(d):
            nonlocal train_time, train_mem, train_epoch_losses, test_epoch_losses
            train_time += d["train_time"]
            for k in pycrank.utils.PEAK_MEM_KEYS:
                train_mem[k] = max(train_mem.get(k) or 0, d.get(k) or 0)
            print('epoch:', 1 + d["epoch_i"], 'of', params["epochs"],
                  'train: logloss: {:7.4f}'.format(d["train_loss"]),
                  'test  : logloss: {:7.4f}'.format(d["val_loss"]))
            sys.stdout.flush()
            train_epoch_losses.append(d["train_loss"])
            test_epoch_losses.append(d["val_loss"])
        pycrank.opt.train(model, train_data_loader, test_data_loader, device, params, callback, f'final {DATASET} {ENCODING} {TRUNCATE}')

        emit["params"] = params
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
