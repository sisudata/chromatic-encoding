#!/usr/bin/env python
# Given two arguments <labelfile> <predfile>
# where the predictions are raw scores (before logistic transformation)
# newline delimited
# outputs two lines to stdout
# "accuracy": 1 - <0-1 loss>,
# "logloss": <logistic loss>,
# "auc": <auc loss>

import sys
import numpy as np
from scipy.special import expit
from sklearn.metrics import log_loss, accuracy_score, roc_auc_score

labelfile = sys.argv[1]
predfile = sys.argv[2]

labels = np.loadtxt(labelfile)
preds = np.loadtxt(predfile)

assert len(labels) == len(preds)
labels = labels > 0

print('"accuracy":', accuracy_score(labels, preds > 0), ",")
preds = expit(preds)
print('"logloss":', log_loss(labels, preds), ",")
print('"auc":', roc_auc_score(labels, preds), ",")
