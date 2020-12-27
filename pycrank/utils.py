"""
Training utility functions.
"""

import warnings
import random
from contextlib import contextmanager
import sys
from time import time

import torch
import numpy as np
from scipy import sparse as sps

from .wd import WideAndDeepModel

@contextmanager
def timeit(name, preprint=True):
    """Enclose a with-block with to debug-print out block runtime"""
    t = time()
    if preprint:
        print(name, end='')
        sys.stdout.flush()
    yield
    t = time() - t
    if not preprint:
        print(name, end='')
    print(' took {:0.1f} seconds'.format(t))

def default_model(name, field_dims):
    """
    Generates a pytorch model based on default parameters.
    """
    if name == 'wd':
        return WideAndDeepModel(field_dims, embed_dim=16, mlp_dims=(16, 16), dropout=0.2)

    raise ValueError('unknown model name: ' + name)

def load_binary_csr(data, indices, indptr, y):
    """
    Given native binary 1-D arrays on disc of dtypes u32, u32, u64, u32
    for a CSR matrix's data, indices, indptr arrays and the target array,
    respectively, load and return the pair (X, y).
    """
    files = [data, indices, indptr, y]
    dtypes = [np.uint32, np.uint32, np.uint64, np.uint32]
    data, indices, indptr, y = (
        np.fromfile(f, dtype=d) for f, d in zip(files, dtypes))
    nrows = np.int64(len(indptr) - 1)
    ncols = np.int64(int(indices.max()) + 1)
    if nrows < 0 or ncols < 0:
        raise ValueError('bad shape {}'.format((nrows, ncols)))
    return sps.csr_matrix((data, indices, indptr), shape=(nrows, ncols)), y

def seed_all(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    warnings.filterwarnings(
        "ignore", message=r".*CUDA initialization: Found no NVIDIA driver.*")

def rpad(X, col):
    assert X.getformat() == 'csr'
    return sps.csr_matrix((X.data, X.indices, X.indptr), shape=(X.shape[0], col))

PEAK_MEM_KEYS = [
    "allocated_bytes.all.peak",
    "reserved_bytes.all.peak",
    "active_bytes.all.peak",
    "inactive_split_bytes.all.peak",
]
