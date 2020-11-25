#!/bin/bash
# Recodes a dataset's files so that featues are indexed by descending
# test set frequency of occurrence. Assumes all features are binary at this
# point in the working dataset files.

set -euo pipefail

dataset="$1"

train_files=$(echo clean/data/${dataset}.train.svm.* | tr ' ' '\n')
test_files=$(echo clean/data/${dataset}.test.svm.* | tr ' ' '\n')
files=$(echo "$train_files" && echo "$test_files")

# recode by frequency into 1-indexed features
bash clean/counts.sh "$train_files" \
    | sort --parallel $(nproc) -k 2nr -k 1 \
    | awk '{ print $1 " " NR }' \
    | cargo run --quiet --release --package crank --example recode -- --files $files

echo "$files" | parallel --will-cite mv {}_recode {}

ntrain_missing=$(echo "$train_files" \
    | (parallel --will-cite grep "  " || true) \
    | wc -l)

if [ "$ntrain_missing" -gt 0 ]; then
    echo "inconsistent recoding $dataset" >&2
fi

ntest_missing=$(echo "$test_files" \
    | (parallel --will-cite grep "  " || true) \
    | wc -l)
ntest=$(bash clean/lines.sh "$test_files")
if [ "$ntest_missing" -gt 0 ]; then
    echo "warning: $ntest_missing lines with new features of $ntest in $dataset test"
fi

echo "$files" | parallel --will-cite sed -E -i "'s/  +/ /g'"

echo "$files" | parallel --will-cite bash clean/valid_svm.sh {} implicit
