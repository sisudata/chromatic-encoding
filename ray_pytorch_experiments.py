#!/usr/bin/env python 3
#
# python ray_pytorch_experiments.py s3src dstfile field-aware(yes/n)
#
# Runs all pytorch experiments, pulling datasets from s3 on demand
# s3src is the key prefix for .tar.zst
#
# this runs "incrementally" / so successful runs are idempotent
#
# if "field-aware" is yes, then this runs a different set of experiments,
# focussed on field-aware methods.

import traceback

import ray
ray.init()

import os
os.makedirs('svms-data', exist_ok=True) # data dir
DATA_DIR = os.path.abspath("svms-data")

import sys
assert len(sys.argv) == 4, sys.argv
s3src = sys.argv[1]
dstfile = sys.argv[2]
field_aware = sys.argv[3] == "yes"

records = []

import shutil
if os.path.exists(dstfile):
    print('NOTE: dstfile {} exists, so will re-use previous runs'.format(dstfile))

    import json
    with open(dstfile, 'r') as f:
        for line in f:
            records.append(json.loads(line))

tups = set(tuple(record[k] for k in ('budget', 'dataset', 'compress', 'learner'))
           for record in records)
print('found {} existing records'.format(len(tups)))
for tup in tups:
    print('  {}'.format(tup))

def any_match(tup):
    for record in records:
        rtup = tuple(record[k] for k in ('budget', 'dataset', 'compress', 'learner'))
        if rtup == tup:
            print('found record for {}\n'.format(tup), end='')
            return record
    print('running {}\n'.format(tup), end='')
    return None

from run_torchfm import main

def logged_main(cmd_args):
    budget, dataset, compress = cmd_args[0:3]
    learner = cmd_args[-1]
    tup = (budget, dataset, compress, learner)
    d = any_match(tup)
    if d is not None:
        return (d, tup)

    try:
        return (main(*cmd_args), tup)
    except Exception as e:
        return (e, tup)

ngpus = 1 if field_aware else 0.5

@ray.remote(num_cpus=1, num_gpus=ngpus, max_calls=1)
def runner(cmd_args):
    return logged_main(cmd_args)

@ray.remote
def cleanup(bash_glob, dependencies):
    ray.wait(dependencies, num_returns=len(dependencies), timeout=None)
    check_call(f"""bash -c 'rm -f {bash_glob}'""", shell=True)

if field_aware:
    # ['ft'] OOM or don't finish
    suffixes = ['faft']
    models = ['wd']
    budgets = [256 * 1024, 1024 * 1024]
else:
    suffixes = ['sm', 'ft', 'te', 'ht']
    budgets = [1024]
    models = ['lr', 'wd', 'fm', 'nfm', 'dfm']

def launch_grid(budget, dataset):
    futures = []
    for suffix in suffixes:
        suffix_futures = []
        for model in models:
            cputhreads = 1
            quiet = "yes"
            cuda = "cuda"
            args = [budget, dataset, suffix, quiet, cputhreads, cuda, model]
            suffix_futures.append(runner.remote(args))

        rmglob = f'{DATA_DIR}/{dataset}.{{train,test}}.{suffix}{budget}.{{data,indices,indptr,y}}.bin'
        cleanup_fut = cleanup.remote(rmglob, suffix_futures)

        futures.extend(suffix_futures)
        futures.append(cleanup_fut)
    return futures

from multiprocessing import cpu_count
assert cpu_count() > 1, cpu_count()

from subprocess import check_call, DEVNULL
import json


compress_bases = [f'fieldaware{budget}.tar.zst' for budget in budgets] if field_aware else ['binary.tar.zst']

for compress_base in compress_bases:
    compress_file = f'{DATA_DIR}/{compress_base}'
    if not os.path.exists(compress_file):
        print('downloading binary zst')
        check_call(f"""aws s3 cp --no-progress "{s3src}{compress_base}" {compress_file}""", shell=True)
    else:
        print('found local zst', compress_file)

    print('extracting binary zst')
    cpus_for_unzst = max(cpu_count() - 1, 1)
    check_call(f"""tar -I "pzstd -p {cpus_for_unzst}" -xf {compress_file}""", shell=True)

# check_call(f"""rm -f {compress_file}""", shell=True)

datasets = ['url', 'kdd12', 'kdda', 'kddb']

futures = []
for budget in budgets:
    for dataset in datasets:
        futures.extend(launch_grid(budget, dataset))

while futures:
    ready_ids, remaining_ids = ray.wait(futures)
    # Get the available object and do something with it.
    with open(dstfile, "a") as f:
        for ready_id in ready_ids:
            result = ray.get(ready_id)
            if result is None:
                continue # cleanup task
            result, tup = result
            if isinstance(result, Exception):
                ex = result
                ex = traceback.format_exception(etype=type(ex), value=ex, tb=ex.__traceback__)
                print('error ignored for task {}:\n{}\n'.format(tup, ex), end='')
                continue
            print(json.dumps(result), file=f)
            print('completed {}\n'.format(tup), end='')

    futures = remaining_ids
