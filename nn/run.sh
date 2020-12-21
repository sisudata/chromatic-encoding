#!/bin/bash
# Expects S3ROOT, DATASETS, ENCODINGS, MODELNAMES to be set. Assumes that all ${dataset}.tar files
# are present in encode/data already.
#
# For given ${encoding}_${dataset}.tar, extracts the dataset, trains a neural net,
# and saves logs and result metrics to ${encoding}.${dataset}.${modelname}.{log,json}.
# TODO: save the NNs?
# TODO: variable-trunc encodings

set -euo pipefail

echo nn run on $DATASETS with $ENCODINGS on $MODELNAMES
echo cache root "$S3ROOT"

source common.sh

to_get=""
for dataset in $DATASETS ; do
    for encoding in $ENCODINGS ; do
        for modelname in $MODELNAMES ; do
            if \
                ! cache_read ${encoding}.${dataset}.${modelname}.log || \
                    ! cache_read ${encoding}.${dataset}.${modelname}.json ; then
                to_get="${to_get}${dataset}.${encoding}.${modelname} "
            fi
        done
    done
done

for dataset_encoding_modelname in $to_get ; do
    dataset_encoding="${dataset_encoding_modelname%.*}"
    modelname="${dataset_encoding_modelname##*.}"
    dataset="${dataset_encoding%.*}"
    encoding="${dataset_encoding##*.}"

    cp encode/data/${encoding}_${dataset}.tar nn/data/
    tar xf nn/data/${encoding}_${dataset}.tar -C nn/data
    
    rm nn/data/${encoding}_${dataset}.tar \
       nn/data/${encoding}_${dataset}.{train,test}{.original,}.svm.*.zst
    rm -f nn/data/${dataset}.${encoding}.jsonl
    zstd -f -d -q --rm nn/data/${dataset}.${encoding}.bin.tar.zst
    tar xf nn/data/${dataset}.${encoding}.bin.tar -C nn/data
    rm nn/data/${dataset}.${encoding}.bin.tar

    DATASET="${dataset}" ENCODING="${encoding}" MODELNAME="${modelname}" python -m nn.run \
           nn/data/${dataset}.${encoding}.train.{data,indices,indptr,y} \
           nn/data/${dataset}.${encoding}.test.{data,indices,indptr,y} \
           nn/data/${encoding}.${dataset}.${modelname}.json | tee nn/data/${encoding}.${dataset}.${modelname}.log
    
    rm nn/data/${dataset}.${encoding}.{train,test}.{data,indices,indptr,y}

    # cache_write nn/data/${encoding}.${dataset}.${modelname}.log
    # cache_write nn/data/${encoding}.${dataset}.${modelname}.json
done
