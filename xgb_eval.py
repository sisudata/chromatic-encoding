# python script for xgboost evaluation
# assumes clone into /tmp of
# https://github.com/mblondel/svmlight-loader

# Usage: python xgb_eval.py model-file load-X-y-py-code
# uses exec() on the last variable to init a fn
# called load() which should return the X, y matrix for train
#
# writes results to <model-file>.json, which should already
# exist from the training script xgb.py

import sys
import os

if "/tmp/svmlight-loader" not in sys.path:
	sys.path.append("/tmp/svmlight-loader")
	sys.path.append("/tmp/svmlight-loader/build")
	from svmlight_loader import load_svmlight_file

assert os.path.exists("/tmp/svmlight-loader")

model_file = sys.argv[1]

json_file = model_file + '.json'

import json

with open(json_file, 'r') as f:
    metrics_and_params = json.load(f)

loadcode = sys.argv[2]

import xgboost as xgb
import numpy as np
from time import time

params = metrics_and_params.copy()
params.pop("load")
params.pop("train")
params.pop("num_round")

exec(loadcode)
X, y = load()
dtest = xgb.DMatrix(X, label=y)

gb = xgb.Booster(params, model_file = model_file)
preds = gb.predict(dtest, ntree_limit=metrics_and_params["num_round"])

metrics_and_params['acc'] = (dtest.get_label() == (preds >= 0.5)).mean()

from sklearn.metrics import roc_auc_score

metrics_and_params['auc'] = roc_auc_score(dtest.get_label(), preds)

with open(json_file, 'w') as f:
    json.dump(metrics_and_params, f)
