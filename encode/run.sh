#!/bin/bash
# Expects dataset tar files to be present in clean/data already.
# For each dataset/encoding pair, generates a CSR matrix
# ${dataset}.${encoding}.{train,test}.{data,indices,indptr,y}
# where the parameters above are all equal-length native
# arrays of type u32.
#
# Also creates newline-delimited json records for each pair
# ${dataset}.${encoding}.jsonl with some log info.
#
# Expects S3ROOT, DATASETS, ENCODINGS to be set. Assumes that all ${dataset}.tar files
# are present in clean/data already.
#
# The different ENCODINGS supported are
#
# ft - frequency truncation, just filters to the most popular TRUNCATE
# features (default 1000000).

set -euo pipefail

echo encoding $DATASETS with $ENCODINGS
echo cache root "$S3ROOT"

source common.sh

to_get=""
for dataset in $DATASETS ; do
    for encoding in $ENCODINGS ; do
        to_get="${to_get}${dataset}.${encoding} "
    done
done

for dataset_encoding in $to_get ; do
    dataset="${dataset_encoding%.*}"
    encoding="${dataset_encoding##*.}"

    cp clean/data/${dataset}.tar encode/data/
    tar xf encode/data/${dataset}.tar -C encode/data
    rm encode/data/${dataset}.{train,test}.original.svm.*.zst
    all=$(echo encode/data/${dataset}.{train,test}.svm.*.zst \
        | tr ' ' '\n' \
        | parallel --no-run-if-empty --will-cite '
            file="{}"
            zstd -f -d -q {} -o "${file%.*}"
            rm {}
            echo "${file%.*}"' \
        | sort )
    
    case $encoding in
        ft)
            echo ft.sh
            ;;
        greedy)
            train=$(echo "$all" | grep train)
            test=$(echo "$all" | grep test)
            k=1
            cargo run --release -p crank --example greedy -- --train $train --test $test --k "$k"
            echo "find"
            find encode/data/ -name "${dataset}.{train,test}.original.svm.*" -delete
            ;;
        hll)
            echo hll not supported
            exit 1
            ;;
    esac

    
done

# ENCODINGS = ft greedy

# ft: just filter ft indices, recode to :1 with awk

# GREEDy -> convert to separate cases, eg freq ordering or degree ordering

