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
    local base=$(basename "$1")
    local fsfile="$DIRPREFIX/data/$base"
    local s3file="$S3ROOT/$DIRPREFIX/$base"
    mv "$1" $fsfile
    aws s3 cp "$fsfile" "$s3file"
}

export -f cache_write
