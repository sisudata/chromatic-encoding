#!/bin/bash
# Expects S3ROOT, DATASETS, ENCODINGS, TRUNCATES, MODELNAMES to be set. Assumes that all ${dataset}.tar files
# are present in encode/data already.
#
# For given ${encoding}_${truncate}_${dataset}.tar, extracts the dataset, trains a neural net,
# and saves logs and result metrics to ${dataset}.${encoding}.${truncate}.${modelname}.tar
#
# which contains a .log logfile, .json result file, .pdf HPO viz all of the same basename.

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
                if ! cache_read ${dataset}.${encoding}.${truncate}.${modelname}.tar ; then
                    to_get="${to_get}${dataset}.${encoding}.${truncate}.${modelname} "
                elif [ "${1:-}" = "--force" ] ; then
                    force "${dataset}.${encoding}.${truncate}.${modelname}.tar"
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

    s3src="${S3ROOT}/encode/${encoding}_${truncate}_${dataset}.tar"
    if ! aws s3 ls "$s3src" > /dev/null 2>/dev/null; then
        echo "missing $s3src"
        exit 1
    fi

    RAY_ADDR="${RAY_ADDRESS:-}" DATASET="${dataset}" ENCODING="${encoding}" TRUNCATE="${truncate}" MODELNAME="${modelname}" python \
            nn/run.py \
            "$s3src" \
           nn/data/${dataset_encoding_truncate_modelname}.json \
           nn/data/${dataset_encoding_truncate_modelname}.pdf \
        | tee nn/data/${dataset_encoding_truncate_modelname}.log
    
    tar cf nn/data/${dataset_encoding_truncate_modelname}.tar -C nn/data \
        --remove-files ${dataset_encoding_truncate_modelname}.{log,json,pdf}
    
    cache_write ${dataset_encoding_truncate_modelname}.tar
done
