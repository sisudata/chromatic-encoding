#!/bin/bash
# Prints a line for every expected compressed
# file based on the DATASETS variable, checking that
# it's present in raw/data

set -euo pipefail

files=""
for dataset in $DATASETS ; do
    case $dataset in
        urltoy)
            test -f raw/data/url_combined.bz2
            files="${files}url_combined.bz2 "
            ;;
        url)
            test -f raw/data/url_combined.bz2
            files="${files}url_combined.bz2 "
            ;;
        kdda)
            test -f raw/data/kdda.bz2
            test -f raw/data/kdda.t.bz2
            files="${files}kdda.bz2 "
            files="${files}kdda.t.bz2 "
            ;;
        kddb)
            test -f raw/data/kddb-raw-libsvm.bz2
            test -f raw/data/kddb-raw-libsvm.t.bz2
            files="${files}kddb-raw-libsvm.bz2 "
            files="${files}kddb-raw-libsvm.t.bz2 "
            ;;
        kdd12)
            test -f raw/data/kdd12.tr.bz2
            test -f raw/data/kdd12.val.bz2
            files="${files}kdd12.tr.bz2 "
            files="${files}kdd12.val.bz2 "
            ;;
    esac
done

echo "$files" \
    | tr ' ' '\n' \
    | sort -u \
    | grep .
