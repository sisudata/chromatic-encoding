#!/bin/bash
#
# Performs coloring and compression that's field-aware, storing field dimensions.
# uses <argument> number of dimensions

set -euo pipefail

budget="$1"
nthreads=64
num_concurrent_datasets=1
total_parallelism=$(( $nthreads * $num_concurrent_datasets))
dst_s3="s3://sisu-datasets/binary-svms/"
datasets="url kdda kddb kdd12"

echo "field-aware encoding all datasets (budget = ${budget})"

for dataset in $datasets ; do
echo \
  --budget $budget --compress FieldAwareFrequencyTruncation \
  --train ./svms-data/${dataset}.train.svm \
  --valid ./svms-data/${dataset}.test.svm
echo \
  --budget $budget --compress FrequencyTruncation \
  --train ./svms-data/${dataset}.train.svm \
  --valid ./svms-data/${dataset}.test.svm
done | RAYON_NUM_THREADS=$nthreads xargs -P $num_concurrent_datasets -L 1 ./csl/target/release/csl >/dev/null

echo "binary-encoding all field-aware encoded datasets"

for dataset in $datasets ; do
for suffix in "faft" "ft" ; do
suffix=".${suffix}${budget}"

echo ./svms-data/${dataset}.train${suffix}.svm
echo ./svms-data/${dataset}.test${suffix}.svm

done
done | RAYON_NUM_THREADS=$nthreads xargs -P $num_concurrent_datasets -L 1 ./csl/target/release/svm2bins >/dev/null

dst_file="fieldaware${budget}.tar.zst"

echo "parallel taring all binary datasets into ./svms-data/${dst_file}"

for suffix in faft ft ; do for dataset in $datasets ; do 
cp ./svms-data/${dataset}.train.${suffix}${budget}.svm.field_dims.txt ./svms-data/${dataset}.test.${suffix}${budget}.svm.field_dims.txt
done; done                               
tar -I "pzstd -p $total_parallelism" -cf "./svms-data/${dst_file}" \
  ./svms-data/{url,kdda,kddb,kdd12}.{train,test}.faft${budget}.{data,indices,indptr,y}.bin \
  ./svms-data/{url,kdda,kddb,kdd12}.{train,test}.faft${budget}.svm \
  ./svms-data/{url,kdda,kddb,kdd12}.{train,test}.faft${budget}.svm.field_dims.txt \
  ./svms-data/{url,kdda,kddb,kdd12}.{train,test}.ft${budget}.{data,indices,indptr,y}.bin \
  ./svms-data/{url,kdda,kddb,kdd12}.{train,test}.ft${budget}.svm \
  ./svms-data/{url,kdda,kddb,kdd12}.{train,test}.ft${budget}.svm.field_dims.txt

echo "uploading to s3"
aws s3 cp ./svms-data/${dst_file} "${dst_s3}${dst_file}"

for i in \
  ./svms-data/{url,kdda,kddb,kdd12}.{train,test}.faft${budget}.{data,indices,indptr,y}.bin \
  ./svms-data/{url,kdda,kddb,kdd12}.{train,test}.faft${budget}.svm \
  ./svms-data/{url,kdda,kddb,kdd12}.{train,test}.faft${budget}.svm.field_dims.txt \
  ./svms-data/{url,kdda,kddb,kdd12}.{train,test}.ft${budget}.{data,indices,indptr,y}.bin \
  ./svms-data/{url,kdda,kddb,kdd12}.{train,test}.ft${budget}.svm \
  ./svms-data/{url,kdda,kddb,kdd12}.{train,test}.ft${budget}.svm.field_dims.txt ; do
    if [ -f $i ] ; then
        rm $i
    else
        echo ERROR $i DOES NOT EXIST
        exit 1
    fi
done

