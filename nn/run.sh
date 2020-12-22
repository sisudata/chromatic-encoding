#!/bin/bash
# Expects S3ROOT, DATASETS, ENCODINGS, TRUNCATES, MODELNAMES to be set. Assumes that all ${dataset}.tar files
# are present in encode/data already.
#
# For given ${encoding}_${truncate}_${dataset}.tar, extracts the dataset, trains a neural net,
# and saves logs and result metrics to ${dataset}.${encoding}.${truncate}.${modelname}.{log,json}.

set -euo pipefail

echo nn run on $DATASETS with $ENCODINGS truncated to $TRUNCATES on $MODELNAMES
echo cache root "$S3ROOT"

source common.sh

to_get=""
for dataset in $DATASETS ; do
    for encoding in $ENCODINGS ; do
        for modelname in $MODELNAMES ; do
            for truncate in $TRUNCATES ; do
                if [[ $truncate -eq 0 ]] && [ "$encoding" != "ce" ]; then
                    continue
                fi
                if \
                    ! cache_read ${dataset}.${encoding}.${truncate}.${modelname}.log || \
                        ! cache_read ${dataset}.${encoding}.${truncate}.${modelname}.json ; then
                to_get="${to_get}${dataset}.${encoding}.${truncate}.${modelname} "
                fi
            done
        done
    done
done

for dataset_encoding_truncate_modelname in $to_get ; do
    dataset_encoding_truncate="${dataset_encoding_truncate_modelname%.*}"
    modelname="${dataset_encoding_truncate_modelname##*.}"
    dataset_encoding="${dataset_encoding_truncate%.*}"
    truncate="${dataset_encoding_truncate##*.}"
    dataset="${dataset_encoding%.*}"
    encoding="${dataset_encoding##*.}"

    cp encode/data/${encoding}_${truncate}_${dataset}.tar nn/data/
    tar xf nn/data/${encoding}_${truncate}_${dataset}.tar -C nn/data
    
    rm nn/data/${encoding}_${truncate}_${dataset}.tar \
       nn/data/${encoding}_${truncate}_${dataset}.{train,test}{.original,}.svm.*.zst
    zstd -f -d -q --rm nn/data/${dataset_encoding_truncate}.bin.tar.zst
    tar xf nn/data/${dataset_encoding_truncate}.bin.tar -C nn/data
    rm nn/data/${dataset_encoding_truncate}.bin.tar

    if [[ $truncate -eq 0 ]]; then
        true_truncate=$(grep "ncolors" nn/data/${dataset_encoding_truncate}.jsonl | head -1 | jq .ncolors)
    else
        true_truncate="$truncate"
    fi
    DATASET="${dataset}" ENCODING="${encoding}" TRUNCATE="${true_truncate}" MODELNAME="${modelname}" python \
           -m nn.run \
           nn/data/${dataset_encoding_truncate}.train.{data,indices,indptr,y} \
           nn/data/${dataset_encoding_truncate}.test.{data,indices,indptr,y} \
           nn/data/${dataset_encoding_truncate_modelname}.json | tee nn/data/${dataset_encoding_truncate_modelname}.log
    
    rm nn/data/${dataset_encoding_truncate}.{train,test}.{data,indices,indptr,y}
    rm nn/data/${dataset_encoding_truncate}.jsonl

    cache_write nn/data/${dataset_encoding_truncate_modelname}.log
    cache_write nn/data/${dataset_encoding_truncate_modelname}.json
done
