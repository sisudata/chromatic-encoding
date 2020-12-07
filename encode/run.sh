#!/bin/bash
# Expects dataset tar files to be present in clean/data already.
# For each dataset/encoding pair, generates a CSR matrix
# ${dataset}.${encoding}.{train,test}.{data,indices,indptr,y}
# where the parameters above are all equal-length native
# arrays of type u32.
#
# Also creates newline-delimited json records for each pair
# ${dataset}.${encoding}.jsonl with some log info and the color-encoded zstd-ed svm
# files in ${dataset}.tar (just like the clean data, the format is 
# ${encoding}_${dataset}.{train,test}{.original,}.svm.0000.zst
# noting the prefix.
#
# Expects S3ROOT, DATASETS, ENCODINGS to be set. Assumes that all ${dataset}.tar files
# are present in clean/data already.
#
# The different ENCODINGS supported are TODO
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
    rm encode/data/${dataset}.tar encode/data/${dataset}.{train,test}.original.svm.*.zst
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
        weight)
            train=$(echo "$all" | grep train)
            test=$(echo "$all" | grep test)
            k=1
            cargo build -p crank --release --example greedy >/dev/null 2>&1
            # TODO tee to redirect
            target/release/examples/greedy --train $train --test $test --k "$k" | tee encode/data/${dataset_encoding}.jsonl
            find encode/data/ -maxdepth 1 -type f -regextype posix-extended -regex \
                 "^encode/data/${dataset}."'(train|test)\.svm\.[0-9]+$' -delete
            
            pushd encode/data >/dev/null
            all=$(find . -maxdepth 1 -type f -regextype posix-extended -regex \
                 "^./${dataset}."'(train|test)\.svm\.[0-9]+_greedy_1$' -print0 \
                 | parallel --will-cite -0 '
                     file=$(basename "{}")
                     strip="${file%_greedy_'"$k"'}"
                     mv "$file" "$strip"
                     stripzst="'"${encoding}"'_${strip}.zst"
                     cat "$strip" | tr ":" "_" | zstd -q -o "${stripzst}"
                     orig="'"${encoding}"'_${strip/.svm/.original.svm}.zst"
                     zstd -q "$strip" -o "$orig"
                     echo "${orig}"
                     echo "${stripzst}"')
            tar cf ${encoding}_${dataset}.tar $all
            rm $all
            popd >/dev/null
            
            rm encode/data/${dataset}.{train,test}.svm.*
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

