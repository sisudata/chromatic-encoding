#!/bin/bash
# Processes a dataset with mixed (binary and continuous) types
# in the manner described in clean/run.sh

set -euo pipefail

dataset="$1"
if ! [ "$BIN" -ge 2 ] ; then
    echo "invalid BIN $BIN" >&2
    exit 1
fi

files=$(echo clean/data/${dataset}.{train,test}.original.svm.* | tr ' ' '\n')

echo "$files" \
    | parallel --will-cite bash clean/valid_svm.sh {} explicit

# convert all negatives to new keys ending in a dash, make clean
# dataset files
echo "$files" \
    | parallel --will-cite '
infile="{}"
outfile="${infile//.original/}"
sed "s/:-/-:/g" $infile > $outfile
'

train_files=$(echo clean/data/${dataset}.train.svm.* | tr ' ' '\n')
test_files=$(echo clean/data/${dataset}.test.svm.* | tr ' ' '\n')

# extract all features with more than BIN nonzero unique values
# get their distinct value counts
# bin on the uniqued, counted data
# convert both train and test to use binned features
MAXCARD=${BIN} bash clean/cards.sh "$train_files" \
    | bash clean/values.sh "$train_files" \
    | BIN=${BIN} parallel --will-cite -N 1 --pipe python clean/bins.py \
    | bash clean/digitize.sh "$files"
