#!/bin/bash
# Expects dataset tar files to be present in clean/data already.
# For each dataset/encoding pair, generates a CSR matrix
# ${dataset}.${encoding}.{train,test}.{data,indices,indptr,y} (see svm2bins docs
# for format of these files). These all get tar'd and zstd'd into
# ${dataset}.${encoding}.bin.tar.zst
#
# Also creates newline-delimited json records for each pair
# ${dataset}.${encoding}.jsonl with some log info and the color-encoded zstd-ed svm
# files in ${encoding}_${dataset}.tar (just like the clean data, the format is 
# ${encoding}_${dataset}.{train,test}{.original,}.svm.0000.zst
# noting the prefix.
#
# Expects S3ROOT, DATASETS, ENCODINGS to be set. Assumes that all ${dataset}.tar files
# are present in clean/data already.
#
# The different ENCODINGS supported are
#
# ft - frequency truncation, just filters to the most popular TRUNCATE
# features (default 1000000).
#
# weight - greedy coloring approach to chromatic encoding.

set -euo pipefail

echo encoding $DATASETS with $ENCODINGS
echo cache root "$S3ROOT"

source common.sh

to_get=""
for dataset in $DATASETS ; do
    for encoding in $ENCODINGS ; do
        if ! cache_read ${encoding}_${dataset}.tar ; then
            to_get="${to_get}${dataset}.${encoding} "
        fi
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
            zstd --rm -f -d -q {} -o "${file%.*}"
            echo "${file%.*}"' \
        | sort )
    
    case $encoding in
        ft)
            train=$(echo "$all" | grep train)
            test=$(echo "$all" | grep test)

            TRUNCATE=${TRUNCATE:-1000000}
            all=$(echo "$all" | parallel --will-cite '
                 file="{}"
                 orig="${file/.svm/.original.svm}"
                 mv $file $orig
                 awk -v TRUNCATE='"${TRUNCATE}"' -f encode/ft.awk $orig > $file
                 zstd --rm -f -q $orig -o ${orig}.zst
                 zstd -f -q $file -o ${file}.zst
                 basename ${orig}.zst
                 basename ${file}.zst
            ')
            # at this point the files referred to by $train and $test are frequency-truncated

            cargo build -p crank --release --example svm2bins > /dev/null 2>&1
            target/release/examples/svm2bins \
                 --train $train --test $test \
                 --train-out encode/data/${dataset_encoding}.train \
                 --test-out encode/data/${dataset_encoding}.test
            
            ;;
        weight)
            echo "${dataset}" "${encoding}"
            train=$(echo "$all" | grep train)
            test=$(echo "$all" | grep test)
            k=1
            cargo build -p crank --release --example greedy >/dev/null 2>&1
            target/release/examples/greedy --train $train --test $test --k "$k" > encode/data/${dataset_encoding}.jsonl
            find encode/data/ -maxdepth 1 -type f -regextype posix-extended -regex \
                 "^encode/data/${dataset}."'(train|test)\.svm\.[0-9]+$' -delete
            
            pushd encode/data >/dev/null
            all=$(find . -maxdepth 1 -type f -regextype posix-extended -regex \
                 "./${dataset}."'(train|test)\.svm\.[0-9]+_greedy_1' -print0 \
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
            popd >/dev/null

            # at this point $train and $test refer to the colorized svm files
            cargo build -p crank --release --example svm2bins > /dev/null 2>&1
            target/release/examples/svm2bins \
                 --train $train --test $test \
                 --train-out encode/data/${dataset_encoding}.train \
                 --test-out encode/data/${dataset_encoding}.test

            all=$(echo "$all" && echo encode/data/${dataset_encoding}.jsonl)
            ;;
    esac

    pushd encode/data >/dev/null

    tar cf ${dataset}.${encoding}.bin.tar \
        --remove-files ${dataset_encoding}.{train,test}.{data,indices,indptr,y}
    zstd -q --rm ${dataset}.${encoding}.bin.tar -o ${dataset}.${encoding}.bin.tar.zst

    tar cf ${encoding}_${dataset}.tar \
        $all \
        ${dataset}.${encoding}.bin.tar.zst
    rm $all \
       ${dataset}.${encoding}.bin.tar.zst
    popd >/dev/null

    rm encode/data/${dataset}.{train,test}.svm.* # all svms and zsts

    
    cache_write encode/data/${encoding}_${dataset}.tar    
done

