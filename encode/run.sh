#!/bin/bash
# Expects dataset tar files to be present in clean/data already.
# For each dataset/encoding/truncate triplet, generates a CSR matrix
# ${dataset}.${encoding}.${truncate}.{train,test}.{data,indices,indptr,y}
# (see svm2bins docs for format of these files). These all get tar'd and zstd'd into
# ${dataset}.${encoding}.${truncate}.bin.tar.zst
#
# Also creates newline-delimited json records for each pair
# ${dataset}.${encoding}.${truncate}.jsonl with some log info and the color-encoded zstd-ed svm
# files in ${encoding}_${truncate}_${dataset}.tar (just like the clean data, the format is 
# ${encoding}_${truncate}_${dataset}.{train,test}{.original,}.svm.0000.zst
# noting the prefix.
#
# Expects S3ROOT, DATASETS, ENCODINGS, TRUNCATES to be set.
# Assumes that all ${dataset}.tar files are present in clean/data already.
# Same goes for ${dataset}.graph.tar in graph/data.
#
# The different ENCODINGS supported are
#
# ft - frequency truncate, just filters to the most popular TRUNCATE
# features (default 1000000).
#
# ce - greedy coloring approach to chromatic encoding.
# TODO: consider ce1, ce10, ce100 corresponding to different k values.
#
# The different TRUNCATES should be integers denoting the number of features
# (possibly high-cardinality) that featurization should truncate to. Note that
# only ce supports a value of 0, which means to truncate to the number of colors.

set -euo pipefail

echo encoding $DATASETS with $ENCODINGS truncating to $TRUNCATES
echo cache root "$S3ROOT"

source common.sh

to_get=""
for dataset in $DATASETS ; do
    for encoding in $ENCODINGS ; do
        for truncate in $TRUNCATES ; do
            if [[ $truncate -eq 0 ]] && [ "$encoding" != "ce" ]; then
                continue
            fi
            if ! cache_read ${encoding}_${truncate}_${dataset}.tar ; then
                to_get="${to_get}${dataset}.${encoding}.${truncate} "
            elif [ "${1:-}" = "--force" ] ; then
                force ${encoding}_${truncate}_${dataset}.tar
                to_get="${to_get}${dataset}.${encoding}.${truncate} "
            fi
        done
    done
done

for dataset_encoding_truncate in $to_get ; do
    dataset_encoding="${dataset_encoding_truncate%.*}"
    truncate="${dataset_encoding_truncate##*.}"
    dataset="${dataset_encoding%.*}"
    encoding="${dataset_encoding##*.}"

    cp clean/data/${dataset}.tar encode/data/
    tar xf encode/data/${dataset}.tar -C encode/data
    rm encode/data/${dataset}.tar encode/data/${dataset}.{train,test}.original.svm.*.zst
    all=$(echo encode/data/${dataset}.{train,test}.svm.*.zst \
        | tr ' ' '\n' \
        | parallel --no-run-if-empty --will-cite '
            file="{}"
            zstd --rm -f -d -q {} -o "${file%.*}"
            echo "${file%.*}"' \
        | sort )
    
    case $encoding in
        ft)
            train=$(echo "$all" | grep train)
            test=$(echo "$all" | grep test)

            pushd encode/data >/dev/null
            all=$(find . -maxdepth 1 -type f -regextype posix-extended -regex \
                 "./${dataset}."'(train|test)\.svm\.[0-9]+' -print0 \
                 | parallel --will-cite -0 '
                 base=$(basename "{}")
                 file="'"${encoding}_${truncate}"'_${base}"
                 orig="${file/.svm/.original.svm}"
                 awk -v TRUNCATE='"${truncate}"' -f ../ft.awk $base > $file
                 zstd --rm -f -q $base -o ${orig}.zst
                 zstd -f -q $file -o ${file}.zst
                 basename ${orig}.zst
                 basename ${file}.zst
                 mv $file $base
            ')
            touch ${dataset_encoding_truncate}.jsonl
            all=$(echo "$all" && echo ${dataset_encoding_truncate}.jsonl)
            popd >/dev/null
            # at this point the files referred to by $train and $test are frequency-truncated

            cargo build -p crank --release --example svm2bins > /dev/null 2>&1
            target/release/examples/svm2bins \
                 --train $train --test $test \
                 --train-out encode/data/${dataset_encoding_truncate}.train \
                 --test-out encode/data/${dataset_encoding_truncate}.test

            rm $train $test
            
            ;;
        ce)
            train=$(echo "$all" | grep train)
            test=$(echo "$all" | grep test)
            k=1

            tar xf graph/data/${dataset}.graph.tar -C encode/data/
            graphs=$( find encode/data -maxdepth 1 -type f -regextype posix-extended -regex \
                           "encode/data/${dataset}."'graph\.[0-9]+.zst' -print0 \
                          | parallel --will-cite -0 'zstd -f -d --rm -q {} && echo "encode/data/$(basename {} .zst)"' )            
            cargo build -p crank --release --example greedy >/dev/null 2>&1
            target/release/examples/greedy --graph $graphs --train $train --test $test --k "$k" --ncolors "$truncate" > encode/data/${dataset_encoding_truncate}.jsonl
            rm $graphs
            find encode/data/ -maxdepth 1 -type f -regextype posix-extended -regex \
                 "^encode/data/${dataset}."'(train|test)\.svm\.[0-9]+$' -delete
            
            pushd encode/data >/dev/null
            all=$(find . -maxdepth 1 -type f -regextype posix-extended -regex \
                 "./${dataset}."'(train|test)\.svm\.[0-9]+_greedy' -print0 \
                 | parallel --will-cite -0 '
                     file=$(basename "{}")
                     strip="${file%_greedy}"
                     mv "$file" "$strip"
                     stripzst="'"${encoding}_${truncate}"'_${strip}.zst"
                     cat "$strip" | tr ":" "_" | zstd -q -o "${stripzst}"
                     orig="'"${encoding}_${truncate}"'_${strip/.svm/.original.svm}.zst"
                     zstd -q "$strip" -o "$orig"
                     echo "${orig}"
                     echo "${stripzst}"')
            all=$(echo "$all" && echo ${dataset_encoding_truncate}.jsonl)
            popd >/dev/null

            # at this point $train and $test refer to the colorized svm files
            cargo build -p crank --release --example svm2bins > /dev/null 2>&1
            target/release/examples/svm2bins \
                 --train $train --test $test \
                 --train-out encode/data/${dataset_encoding_truncate}.train \
                 --test-out encode/data/${dataset_encoding_truncate}.test
            rm $train $test
            ;;
    esac

    pushd encode/data >/dev/null

    tar cf ${dataset_encoding_truncate}.bin.tar \
        --remove-files ${dataset_encoding_truncate}.{train,test}.{data,indices,indptr,y}
    zstd -q --rm ${dataset_encoding_truncate}.bin.tar -o ${dataset_encoding_truncate}.bin.tar.zst

    tar cf ${encoding}_${truncate}_${dataset}.tar --remove-files \
        $all \
        ${dataset_encoding_truncate}.bin.tar.zst
    popd >/dev/null
    
    cache_write encode/data/${encoding}_${truncate}_${dataset}.tar    
done

