#!/bin/bash
# Accepts a dataset as the first argument
# Expects ${dataset}.tar to be present in the cwd
#
# Extracts data, trains a VW model against it, generating the following
# files in cwd:
# - ${dataset}.{train,test}.log - logfiles
# - ${dataset}.{train,test}.{label,pred} - labels only, predictions only
#
# deletes the tarfile.
#
# Prints to stdout
# "train_sec": <seconds>,

set -euo pipefail

dataset="$1"
jobid="$RANDOM"

# extract train

tar xf ${dataset}.tar
rm ${dataset}.tar ${dataset}.{train,test}.original.svm.*.zst
trainfiles=$( \
    find  . -maxdepth 1 -name "${dataset}.train.svm.*.zst" -type f -print0  \
        | parallel -0 --will-cite '
            zstd -f -q -d {}
            rm {}
            compressed="{}"
            extracted=${compressed%.*}
            sed -i "s/^0/-1 |/" $extracted
            sed -i "s/^1/1 |/" $extracted
            echo $extracted' \
                | rev | cut -d "." -f1 | rev)

# train in parallel

ncpu=$(nproc)
total=$(echo "$trainfiles" | wc -l)
chunksize=$(( ($total + $ncpu - 1) / $ncpu ))
njob=$(( ($total + $chunksize - 1) / $chunksize ))

cmd="vw --bit_precision 26 --loss_function logistic"
cmd="$cmd --kill_cache --cache --passes 5 --cache_file ${dataset}.train.svm.{%}.cache"
cmd="$cmd --node \$(({%} - 1)) --unique_id ${jobid} --span_server localhost --total ${njob}"

start=$(date +%s.%N)
echo "$trainfiles" | parallel -j${njob} -N${chunksize} --will-cite '
  if [ {%} -eq 1 ] ; then
      extra_args="--final_regressor '"${dataset}"'.model"
  fi
  cat '"${dataset}"'.train.svm.{} | '"$cmd"' $extra_args 2>&1
  ' > ${dataset}.train.log
end=$(date +%s.%N)

dt=$(echo "$end - $start" | bc)
echo '"train_sec":' "$dt" ","

# eval on train

find  . -maxdepth 1 -regex '\./'"${dataset}"'\.train\.svm\.[0-9]+' -type f -print0  \
    | parallel -0 --will-cite '
        echo {}
        vw --bit_precision 26 --loss_function logistic \
          --data {} --cache \
          --testonly --initial_regressor '"${dataset}"'.model \
          --predictions {}.pred 2>&1
        cut -d" " -f1 {} > {}.label
        rm {}' >> ${dataset}.train.log
cat ${dataset}.train.svm.*.pred > ${dataset}.train.pred
cat ${dataset}.train.svm.*.label > ${dataset}.train.label
rm -f ${dataset}.train.svm.* # rm cache, zst, train, preds, labels

# eval on test

find  . -maxdepth 1 -name "${dataset}.test.svm.*.zst" -type f -print0  \
    | parallel -0 --will-cite '
        zstd -f -q -d {}
        rm {}
        compressed="{}"
        extracted=${compressed%.*}
        sed -i "s/^0/-1 |/" $extracted
        sed -i "s/^1/1 |/" $extracted
        vw --bit_precision 26 --loss_function logistic \
          --data ${extracted} \
          --testonly --initial_regressor '"${dataset}"'.model \
          --predictions ${extracted}.pred 2>&1
        cut -d" " -f1 ${extracted} > ${extracted}.label
        rm ${extracted}' > ${dataset}.test.log
cat ${dataset}.test.svm.*.pred > ${dataset}.test.pred
cat ${dataset}.test.svm.*.label > ${dataset}.test.label
rm -f ${dataset}.test.svm.* # rm preds, labels, test
