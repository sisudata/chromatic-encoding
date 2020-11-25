#!/bin/bash
# Expects data files for each dataset to be present on the local fs
# already by invocation of ./clean/run.sh
#
# Expects S3ROOT, DATASETS to be set.
#
# For each dataset in DATASETS, creates chunked files locally, caching to S3:
# ${dataset}.{train,test}{.original,}.svm.000
# where the final number indicates chunk index (files are CHUNK-line chunked,
# default 100k), and every set of chunked files is zstd-encoded and then
# tarred into ${dataset}.tar.
# `original.svm` corresponds to the unmodified data raw files
# `.svm` is the cleaned data
#
# Dataset-specific preprocessing: for consistency accross different datasets,
# all features are converted to binary. In principle, we could save
# dense features and combine them with sparse ones after chromatic encoding,
# but this saves us a large code path.
#
# url [1] - FeatureTypes file contains all real-valued feature indices,
# for which non-zero values are binned into BIN (default 100) percentile
# bins, equidepth, with negatives binned separately, and with bins
# becoming their own binary features (or fewer if there were less distinct).
#
# kdda [2] - various features are real-valued transformations of counts,
# which were similarly discretized.
#
# kddb [3] - all binary
#
# kdd12 [3] - all binary
#
# Regardless, all final binary features are re-indexed in descending
# order of frequency. Since they are all binary, values are left off
# as appearance implies a value of 1.0.
#
# [1] http://www.sysnet.ucsd.edu/projects/url/
# [2] https://pslcdatashop.web.cmu.edu/KDDCup/workshop/papers/kdd2010ntu.pdf
# [3] https://www.csie.ntu.edu.tw/~cjlin/papers/ffm.pdf

set -euo pipefail

CHUNK=${CHUNK:-100000}
BIN=${BIN:-100}

echo cleaning datasets $DATASETS
echo cache root "$S3ROOT"

source common.sh

to_get=""
for dataset in $DATASETS ; do
    if ! cache_read ${dataset}.zst ; then
        to_get="${to_get}${dataset} "
    fi
done

files=$(DATASETS="${to_get}" bash clean/datasets_to_compressed.sh)

echo "$files" \
    | parallel --no-run-if-empty --will-cite bash clean/bz2extract.sh

for dataset in $to_get ; do
    if [ "$dataset" = "urltoy" ] ; then
        bash clean/maketoy.sh
    fi
done

echo "$to_get" \
    | tr ' ' '\n' \
    | parallel --no-run-if-empty --will-cite --no-run-if-empty echo {}.{train,test}.original.svm \
    | tr ' ' '\n' \
    | parallel --no-run-if-empty --will-cite split -l $CHUNK -a 4 -d clean/data/{} clean/data/{}.

for dataset in $to_get ; do
    case $dataset in
        urltoy|url|kdda)
            BIN="$BIN" bash clean/process_mixed_dataset.sh $dataset
            ;;
        kddb|kdd12)
            bash clean/process_binary_dataset.sh $dataset
            ;;
    esac
    
    bash clean/recode_dataset.sh $dataset

    pushd clean/data >/dev/null
    all=$(echo ${dataset}.{train,test}{.original,}.svm.* \
        | tr ' ' '\n' \
        | parallel --will-cite 'zstd -q {} -o {}.zst && rm {} && echo {}.zst')
    tar cf ${dataset}.tar $all
    rm $all clean/data/${dataset}.{train,test}.original.svm
    cache_write ${dataset}.tar
    popd >/dev/null
done

echo "$DATASETS" \
    | tr ' ' '\n' \
    | parallel --will-cite "

