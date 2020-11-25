#!/bin/bash
# Prints out all dataset access urls
# for every dataset in DATASETS

set -euo pipefail

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
            urls="${urls}${libsvmurl}kddb-raw-libsvm.bz2 "
            urls="${urls}${libsvmurl}kddb-raw-libsvm.t.bz2 "
            ;;
        kdd12)
            urls="${urls}${libsvmurl}kdd12.tr.bz2 "
            urls="${urls}${libsvmurl}kdd12.val.bz2 "
            ;;
        *)
            echo "unknown dataset $dataset" >&2
            exit 1
            ;;
    esac
done

echo "$urls" | tr ' ' '\n' | sort -u
