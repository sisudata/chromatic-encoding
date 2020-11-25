#!/bin/bash
# Common bash function library.
#
# This is sourced in various nested scripts, but does not execute
# anything on its own. Expects S3ROOT set.
#
# Note that this makes heavy assumptions about the way this bash
# script is run, namely we assume this is being sourced from
# a bash script invoked from repo like `bash subdir/run.sh`

# Which nested 
export DIRPREFIX=$(basename $(dirname "${BASH_SOURCE[1]}"))

function cache_read {
    local fsfile="$DIRPREFIX/data/$1"
    if [ -f "$fsfile" ]; then
        echo cache hit "$fsfile"
        return 0
    fi
    local s3file="$S3ROOT/$DIRPREFIX/$1"
    if aws s3 cp "$s3file" "$fsfile" >/dev/null 2>/dev/null ; then
        echo cache hit "$s3file"
        return 0
    fi
    return 1
}

export -f cache_read

function cache_write {
    local s3file="$S3ROOT/$DIRPREFIX/$base"
    aws s3 cp "$1" "$s3file"
}

export -f cache_write


# https://stackoverflow.com/a/41962458/1779853
function seeded_random {
  seed="$1"
  openssl enc -aes-256-ctr -pass pass:"$seed" -nosalt \
    </dev/zero 2>/dev/null
}
