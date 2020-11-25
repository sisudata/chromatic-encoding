#!/bin/bash
# Processes a dataset with only binary types
# in the manner described in clean/run.sh

set -euo pipefail

dataset="$1"

files=$(echo clean/data/${dataset}.{train,test}.original.svm.* | tr ' ' '\n')

# for kddb all values have constant 0.33333 valuation
# kdd12 has 0.30151 mostly, and various other crap
echo "$files" | parallel --will-cite '
infile="{}"
outfile="${infile//.original/}"
sed "s/0[.]33333/1/g" $infile \
 | sed "s/0[.]30151/1/g" \
 | sed -r "s/ ([0-9]+):( |\$)/ \1:1\2/g" \
 | sed -r "s/ ([0-9]+)( |\$)/ \1:1\2/g" > $outfile
'

files=$(echo clean/data/${dataset}.{train,test}.svm.* | tr ' ' '\n')

echo "$files" | parallel --will-cite bash clean/valid_svm.sh {} binary


