#!/bin/bash
# Downloads raw data files. Doesn't decode or rename or do anything
# remotely clever, besides cache to S3.
# Expects S3ROOT, DATASETS to be set

set -euo pipefail

echo grabbing raw for datasets $DATASETS
echo cache root "$S3ROOT"

source common.sh

urls=$(bash raw/datasets_to_urls.sh)

to_get=""
for url in $urls ; do
    fname=$(basename "$url")
    if ! cache_read "$fname" ; then
        to_get="${to_get}$url "
    fi
done


to_get=$(echo "$to_get" \
    | tr ' ' '\n' \
    | xargs -r -P $(echo "$to_get" | wc -l) -L 1 bash -c '
base=$(basename "$1")
wget -q -O raw/data/$base "$1"
echo $base
' --)

for base in $to_get ; do
    cache_write $base
done 

    
