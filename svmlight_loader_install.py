# this local module is to be imported because there seem
# to be pip issues with the svmlight loader
#
# just use
#
# from svmlight_loader_install import load_svmlight_file, dump_svmlight_file
from subprocess import check_call, DEVNULL

check_call(r"""
if ! [ -d /tmp/svmlight-loader ]  ; then
cd /tmp
test -e svmlight-loader || git clone https://github.com/mblondel/svmlight-loader.git
cd svmlight-loader
make
python setup.py build
fi
""", shell=True, stdout=DEVNULL)

import sys
if '/tmp/svmlight-loader' not in sys.path:
    sys.path.append('/tmp/svmlight-loader')
    sys.path.append('/tmp/svmlight-loader/build')
from svmlight_loader import load_svmlight_file, dump_svmlight_file

import os
import numpy as np
from scipy import sparse as sps

def binprefix(svmlight):
    """ assumes user has run svm2bins on the svm file """
    assert svmlight.endswith('.svm')
    svmlight = svmlight[:len(svmlight) - len('.svm')]
    suffixes = ['.data.bin', '.indices.bin', '.indptr.bin', '.y.bin']
    for s in suffixes:
        if not os.path.exists(svmlight + s):
            raise FileNotFoundError(svmlight + s)
    dtypes = [np.float64, np.uint32, np.uint64, np.float64]
    data, indices, indptr, y = (
        np.fromfile(svmlight + s, dtype=d) for s, d in zip(suffixes, dtypes))
    return sps.csr_matrix((data, indices, indptr)), y
