#!/bin/bash
# Given a compressed file expected to be in the raw data, extracts
# it to clean/data and creates clean/data/<dataset>.{train,test}.original.svm
# svmlight files based on its contents

set -euo pipefail

bz2file="$1"
basefile="${bz2file%.*}"

function local_unzip {
    prefix="$1"
    if [ -f clean/data/${prefix}.original.svm ] ; then
        exit 0
    fi
    bunzip2 -c "raw/data/$bz2file" > "clean/data/${basefile}"
    mv clean/data/${basefile} clean/data/${prefix}.original.svm
}

if [ "$bz2file" = url_combined.bz2 ] ; then
    if [ -f clean/data/url.train.original.svm ] && \
           [ -f clean/data/url.test.original.svm ] ; then
        exit 0
    fi
    bunzip2 -c "raw/data/$bz2file" > "clean/data/${basefile}"
    num_lines=$(wc -l < clean/data/url_combined)
    train=$(( 7 * $num_lines / 10 ))
    test=$(( $num_lines - $train ))
    head -n $train clean/data/url_combined \
         | sed 's/^-1/0/' \
         | sed 's/^+1/1/' \
               > clean/data/url.train.original.svm
    tail -n $test clean/data/url_combined \
         | sed 's/^-1/0/' \
         | sed 's/^+1/1/' \
               > clean/data/url.test.original.svm
    rm clean/data/url_combined
elif [ "$bz2file" = kdd12.tr.bz2 ] ; then
    local_unzip kdd12.train
elif [ "$bz2file" = kdd12.val.bz2 ] ; then
    local_unzip kdd12.test
elif [ "$bz2file" = kddb-raw-libsvm.bz2 ] ; then
    local_unzip kddb.train
elif [ "$bz2file" = kddb-raw-libsvm.t.bz2 ] ; then
    local_unzip kddb.test
elif [ "$bz2file" = kdda.bz2 ] ; then
    local_unzip kdda.train
elif [ "$bz2file" = kdda.t.bz2 ] ; then
    local_unzip kdda.test
else
    echo "invalid bz2file $bz2file" >&2
    exit 1
fi
