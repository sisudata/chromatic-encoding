# python script for xgboost training
# assumes clone into /tmp of
# https://github.com/mblondel/svmlight-loader
#  max-depth  0 -> use default
# num-rounds = num trees. 0 -> use default

# Usage: python xgb.py nthreads max-depth num-round outfile load-X-y-py-code
# uses exec() on the last variable to init a fn
# called load() which should return the X, y matrix for train
# writes params, timing to outfile.json


import sys

import os
if "/tmp/svmlight-loader" not in sys.path:
	sys.path.append("/tmp/svmlight-loader")
	sys.path.append("/tmp/svmlight-loader/build")
	from svmlight_loader import load_svmlight_file

assert os.path.exists("/tmp/svmlight-loader")

nthreads = int(sys.argv[1])
maxdepth = int(sys.argv[2])
numround = int(sys.argv[3])
outfile = sys.argv[4]
loadcode = sys.argv[5]

import xgboost as xgb
import numpy as np
from time import time

param = {
    'objective': 'binary:logistic',
    'nthread': nthreads,
}
if maxdepth:
    param['max_depth'] = maxdepth
if not numround:
    numround = 10 # per docs

t = time()
exec(loadcode)
X, y = load()
df = xgb.DMatrix(X, label=y)
load = time() - t
print('load', load)
print('train X {} y {}'.format(X.shape, y.shape))

t = time()
gbm = xgb.train(param, df, numround)
train = time() - t
print('train', train)

gbm.save_model(outfile)
param['load'] = load
param['train'] = train
param['num_round'] = numround
import json
with open(outfile + '.json', 'w') as f:
    json.dump(param, f)
