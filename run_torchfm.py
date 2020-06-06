#!/usr/bin/env python3
#
# script for evaluating torch fm routines
#
# python run_torchfm.py budget dataset compress quiet(yes/n) nthreads device modeltype
#
# device can be 'cpu' 'cuda'
# on multigpu systems you can specify the devicenum as well
#
# note ffm, fnfm will not run, essentially, due to quadratic overhead of field
# to field crossings, unless you use CL.

import torch
from _torchmodel import runall

import json

# params from original py package
# cites measure legitness for model preference
# also have GB usage and util for V100 (16GB total)
# as well as test acc for kdda on sm1024 encoding (used to make sure
# nothing is super fucked with training, again I just used default
# parameters)
ed = {
    'lr': 16,
    # https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf
    'fm': 16, # 1092 cites, 1.5GB, 17%, 87.8%
    # https://arxiv.org/abs/1606.07792
    'wd': 16, # 848 cites, 1.4GB, 16%, 88.6%
    # very wide & deep experiment with raw input data: for 50M
    # features, on a V100, you need a small dimension
    'vwd': 4,
    # https://arxiv.org/abs/1708.05027
    'nfm': 64, # 325 cites, 3GB, 26%, 87.9%
    # https://www.ijcai.org/Proceedings/2017/0239.pdf
    'dfm': 16, # 342 cites, 1.8GB, 15%, 88.1%
    # https://www.csie.ntu.edu.tw/~cjlin/papers/ffm.pdf
    'ffm': 4, # 227 cites
    # https://arxiv.org/abs/1708.04617
    'afm': 16, # 189 cites
    # https://arxiv.org/abs/1601.02376
    'fnn': 16, # 166 cites, 1.3GB, 4% util, 88.7%
    # https://arxiv.org/abs/1708.05123
    'dcn': 16, # 139 cites
    # https://arxiv.org/abs/1611.00144
    'ipnn': 16, # 135 cites,
    'opnn': 16, # ibid,
    # https://arxiv.org/abs/1803.05170
    'xdfm': 111, # 111 cites
    'fnfm': 4, # 5 cites
    'afi': 16, # 4 cites, 4GB, 80% util
    'afn': 16 # 0 cites, 1GB, 40% util
    # 'ncf' -- not valid for non-cf contexts
}


def main(budget, dataset, compress, quiet, nthreads, device, modeltype):
    suffix=f'.{compress}{budget}'
    train_path = f'svms-data/{dataset}.train{suffix}.svm'
    test_path = f'svms-data/{dataset}.test{suffix}.svm'
    torch.set_num_threads(nthreads)

    assert modeltype in ed, (modeltype, list(ed))

    embed_dim = ed[modeltype]
    if modeltype == 'vwd':
        modeltype = 'wd'

    emit = runall(train_path,
              test_path,
              torch.device(device),
              model_name=modeltype,
              embed_dim=embed_dim,
              epoch=5, # originally 15, but that was for larger batch size and took too long
              learning_rate=1e-3,
              batch_size=256, # originally 2048, but I ran out of gpu mem
              weight_decay=1e-6,
                  quiet=False,
                  tqdm_quiet=True)

    emit["dataset"] = dataset
    emit["budget"] = budget
    emit["compress"] = compress

    return emit

if __file__ == "__main__":
    import sys

    assert len(sys.argv) in [8,9], sys.argv

    budget = int(sys.argv[1])
    dataset = sys.argv[2]
    compress = sys.argv[3]
    quiet = sys.argv[4] == "yes"
    nthreads = int(sys.argv[5])
    device = sys.argv[6]
    modeltype= sys.argv[7]

    emit = main(budget, dataset, compress, quiet, nthreads, device, modeltype)
    print(json.dumps(emit))
