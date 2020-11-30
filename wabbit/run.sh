#!/bin/bash
# Expects dataset tar files to be present in clean/data already.
# For each dataset, extracts all datasets, learns a VW learner in
# parallel, and evaluates it on the corresponding test set
#
# For each dataset writes all shard logs and spanning tree server
# log to ${dataset}.train.log and ${dataset}.test.log,
# evaluation stats to ${dataset}.json,
# and the final vowpal models to ${dataset}.model.
#
# Expects S3ROOT, DATASETS to be set.

set -euo pipefail

echo vw on $DATASETS
echo cache root "$S3ROOT"

source common.sh

to_get=""
for dataset in $DATASETS ; do
    if \
        ! cache_read ${dataset}.json || \
        ! cache_read ${dataset}.train.log || \
        ! cache_read ${dataset}.test.log || \
        ! cache_read ${dataset}.model ; then
        to_get="${to_get}${dataset} "
    fi
done


spanning_tree >wabbit/data/spanning_tree.out 2>wabbit/data/spanning_tree.err
trap "killall spanning_tree && rm -f wabbit/data/spanning_tree.{out,err}" EXIT

for dataset in $to_get ; do
    cp clean/data/${dataset}.tar wabbit/data

    echo '{' > wabbit/data/${dataset}.json
    
    pushd wabbit/data >/dev/null
    bash ../vw.sh ${dataset} >> ${dataset}.json
    popd >/dev/null

    echo '"train": {' >> wabbit/data/${dataset}.json
    python wabbit/logloss.py \
           wabbit/data/${dataset}.train.label \
           wabbit/data/${dataset}.train.pred >> wabbit/data/${dataset}.json
    echo '},' >> wabbit/data/${dataset}.json
    echo '"test": {' >> wabbit/data/${dataset}.json
    python wabbit/logloss.py \
           wabbit/data/${dataset}.test.label \
           wabbit/data/${dataset}.test.pred >> wabbit/data/${dataset}.json
    echo '},' >> wabbit/data/${dataset}.json
    echo '}' >> wabbit/data/${dataset}.json
    rm wabbit/data/${dataset}.{train,test}.{pred,label}

    cache_write wabbit/data/${dataset}.json
    cache_write wabbit/data/${dataset}.train.log
    cache_write wabbit/data/${dataset}.test.log
    cache_write wabbit/data/${dataset}.model
done
