#!/bin/bash
# Downloads raw data files. Doesn't decode or rename or do anything
# remotely clever, besides cache to S3.
# Expects S3ROOT, DATASETS to be set

set -euo pipefail

echo grabbing raw for datasets $DATASETS
echo cache root "$S3ROOT"

source common.sh

libsvmurl="https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/"

urls=""
for dataset in $DATASETS ; do
    case $dataset in
        urltoy)
            urls="${urls}${libsvmurl}url_combined.bz2 "
            ;;
        url)
            urls="${urls}${libsvmurl}url_combined.bz2 "
            ;;
        kdda)
            urls="${urls}${libsvmurl}kdda.bz2 "
            urls="${urls}${libsvmurl}kdda.t.bz2 "
            ;;
        kddb)
            urls="${urls}${libsvmurl}kddb.bz2 "
            urls="${urls}${libsvmurl}kddb.t.bz2 "
            ;;
        kdd12)
            urls="${urls}${libsvmurl}kdd12.bz2 "
            urls="${urls}${libsvmurl}kdd12.t.bz2 "
            ;;
    esac
done

urls=$(echo "$urls" | tr ' ' '\n' | sort -u)

to_get=""
for url in $urls ; do
    fname=$(basename "$url")
    if ! cache_read "$fname" ; then
        to_get="${to_get}$url "
    fi
done

echo "$to_get" | tr ' ' '\n' | grep . | xargs -P 100 -L 1 -I {} bash -c '
wget -q -O /tmp/$(basename "{}") "{}" ; cache_write /tmp/$(basename "{}")'
    
