#!/bin/bash
# For each dataset, generates an ultra-simple text file in svmlight format
# <target> <neightbor>:<edge weight>...
# where a given line contains nodes adjacent to the first <target> node
# and their edge weights.
#
# The file is outputted as ${dataset}.graph all in text format, and each
# node is only ever a <target> once. Redundant bidirectional edges are not encoded.
# Also creates a ${dataset}.jsonl with log info.
#
# Expects S3ROOT, DATASETS to be set.
# Assumes that all ${dataset}.tar files
# are present in clean/data already.

set -euo pipefail

echo generating co-occurrence graph for $DATASETS
echo cache root "$S3ROOT"

source common.sh

to_get=""
for dataset in $DATASETS ; do
    if ! cache_read ${dataset}.graph ; then
        to_get="${to_get}${dataset} "
    fi
done

for dataset in $to_get ; do
    cp clean/data/${dataset}.tar graph/data/
    tar xf graph/data/${dataset}.tar -C graph/data
    rm graph/data/${dataset}.tar graph/data/${dataset}.{train,test}.original.svm.*.zst
    rm graph/data/${dataset}.test.svm.*.zst

    all=$(echo graph/data/${dataset}.train.svm.*.zst \
        | tr ' ' '\n' \
        | parallel --no-run-if-empty --will-cite '
            file="{}"
            zstd --rm -f -d -q {} -o "${file%.*}"
            echo "${file%.*}"' \
        | sort )
    
    cargo build -p crank --release --example graph >/dev/null 2>&1
    target/release/examples/graph --files $all --out graph/data/${dataset}.graph > graph/data/${dataset}.jsonl
    rm $all
    
    cache_write graph/data/${dataset}.graph
    cache_write graph/data/${dataset}.jsonl
done

