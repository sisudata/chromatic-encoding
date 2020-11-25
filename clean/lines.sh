#!/bin/bash
# Sums up lines in all argument files in parallel

echo "$1" \
    | parallel --will-cite wc -l \
    | cut -d" " -f1 \
    | datamash sum 1
