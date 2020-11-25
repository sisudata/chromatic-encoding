#!/bin/bash
# Assuming clean/data/url{,.t} are defined, generates clean/data/urltoy versions

if ! [ -f clean/data/urltoy.train.original.svm ]; then
    num_lines=$(wc -l < clean/data/url.train.original.svm)
    train=$(( $num_lines / 10 ))
    tail -n $train clean/data/url.train.original.svm > clean/data/urltoy.train.original.svm
fi

if ! [ -f clean/data/urltoy.test.original.svm ]; then
    num_lines=$(wc -l < clean/data/url.test.original.svm)
    train=$(( $num_lines / 10 ))
    tail -n $train clean/data/url.test.original.svm > clean/data/urltoy.test.original.svm
fi

