#!/usr/bin/env python3
#
# script for evaluating lgbm runtime
#
# python lgbm.py nthreads path-to-manifest.json
#
# manifest json file is expected to have at least the following:
# { "train": "path-to-train-file",
#   "valid": "path-to-valid-file",
#   "train_rows": expected number of training rows,
#   "train_cols": expected number of training cols,
#   "valid_rows": expected number of validation rows,
#   "valid_cols": expected number of validation cols,
#   "categorical": [list of indices of categorical columns] }
#
# files can have 'bin' suffixes (flat float4 array, categoricals
# just count as unique floats, scalar value unimportant, when
# you reshape array in C-order with rows and an extra column then
# the first column is the target, the next one is the 0-th feature)
#
# or they can have 'svm' suffixes (classical svmlight fmt)

import sys

assert len(sys.argv) == 3, sys.argv

nthreads = int(sys.argv[1])
manifest = int(sys.argv[2])

import json
with open(manifest, 'r') as f:
    manifest = json.load(f)

train = manifest["train"]
valid = manifest["valid"]
train_rows = manifest["train_rows"]
train_cols = manifest["train_cols"]
valid_rows = manifest["valid_rows"]
valid_cols = manifest["valid_cols"]
categorical = manifest["categorical"]

assert train_cols == valid_cols, (train_cols, valid_cols)
cols = train_cols

from svmlight_loader_install import load_svmlight_file
import numpy as np

def load(fn):
    assert fn.endswith('.svm') or fn.endswith('.bin'), fn
    if fn.endswith('.svm'):
        return load_svmlight_file(fn)

    yX = np.fromfile(fn, np.float32).reshape(-1, cols + 1)
    return yX[:, 0], yX[:, 1:]

from time import time
import lightgbm as lgb
import warnings

t = time()

y, X = load(train)
assert X.shape[0] == train_rows, (X.shape, train_rows)

dataset = lgb.Dataset(data=X, label=y, params={'verbose': -1})

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message='Using categorical_feature in Dataset.')
    warnings.filterwarnings("ignore", message='.*categorical_feature in Dataset is overridden.*')

    gbm = lgb.train(
        params = {
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'num_threads': nthreads,
            'verbose': -1,
            'metric': 'binary_logloss'},
        train_set=dataset,
        verbose_eval=False,
        categorical_feature=categorical)

t = time() - t

from sklearn.metrics import roc_auc_score, log_loss

def evaluate(y, X):
    out = gbm.predict(X, verbose_eval=False)
    auc = roc_auc_score(y, out)
    acc = ((out >= 0.5) == y).mean()
    logloss = log_loss(y, out)
    return auc, acc, logloss

emit = {}
emit["train_auc"], emit["train_acc"], emit["train_logloss"] = evaluate(y, X)

y, X = load(valid)
assert X.shape[0] == valid_rows, (X.shape, train_rows)
emit["auc"], emit["acc"], emit["test_logloss"] = evaluate(y, X)

emit["train_examples"] = train_rows
emit["learner"] = "LGBM" if train.endswith('.svm') else "CL+LGBM"
emit["budget"] = cols
emit["test_examples"] = valid_rows
emit["train_sec"] = t

print(json.dumps(emit))
