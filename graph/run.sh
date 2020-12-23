#!/bin/bash
# For each dataset, generates an ultra-simple text file in svmlight format
# <target> <neightbor>:<edge weight>...
# where a given line contains nodes adjacent to the first <target> node
# and their edge weights.
#
# The file is outputted as ${dataset}.graph.00000 all in text format, chunked into
# files of CHUNK lines, default 10k.
#
# Any node is only ever a <target> once. Redundant bidirectional edges are not encoded.
# Also creates a ${dataset}.jsonl with log info.
#
# zstd's the graph files and then tars everything into a final ${dataset}.graph.tar
#
# Expects S3ROOT, DATASETS to be set.
# Assumes that all ${dataset}.tar files
# are present in clean/data already.

set -euo pipefail

CHUNK=${CHUNK:-10000}

echo generating co-occurrence graph for $DATASETS
echo cache root "$S3ROOT"

source common.sh

to_get=""
for dataset in $DATASETS ; do
    if ! cache_read ${dataset}.graph.tar ; then
        to_get="${to_get}${dataset} "
    elif [ "$1" = "--force" ] ; then
        force "${dataset}.graph.tar"
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

    split -l $CHUNK -a 5 -d graph/data/${dataset}.graph graph/data/${dataset}.graph.
    rm graph/data/${dataset}.graph
    all=$(find graph/data -maxdepth 1 -type f -regextype posix-extended -regex \
               "graph/data/${dataset}."'graph\.[0-9]+' -print0 \
              | parallel --will-cite -0 'zstd --rm -q {} && basename {}.zst')
    tar cf graph/data/${dataset}.graph.tar -C graph/data --remove-files $all ${dataset}.jsonl

    cache_write ${dataset}.graph.tar
done

