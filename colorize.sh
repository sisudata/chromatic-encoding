#!/bin/bash
#
# does multiple featurizations and uploads results to a destination
# bucket
#
# computes for 2^10 up to 2^18 the text format of ft and sm
# and uploads as ${dataset}.tar.zst
#
# then for budget = 1024 does te ft and sm, all available in text
# and binary formats, uploading to binary.tar.zst

set -euo pipefail
    
bits=$(seq 10 18)
nthreads=32
num_concurrent_datasets=1
total_parallelism=$(( $nthreads * $num_concurrent_datasets))
dst_s3="s3://sisu-datasets/glauber/"
datasets="url kdda kddb kdd12"

echo "compressing $datasets across bits $(echo $bits)"

for dataset in $datasets ; do

echo "pre-processing dataset $dataset"
    
for i in $bits ; do 
    budget=$((1 << $i))
flags="--k 10 --glauber-samples 10000000 "

for compress_suffix in "FrequencyTruncation ft" "SubmodularExpansion sm"; do 
compress=$(echo $compress_suffix | cut -d" " -f1)

echo \
  --budget $budget --compress ${compress} \
  --train ./svms-data/${dataset}.train.svm \
  --valid ./svms-data/${dataset}.test.svm $flags
done
done | RAYON_NUM_THREADS=$nthreads xargs -P $num_concurrent_datasets -L 1  ./csl/target/release/csl >svms-data/log-colorize.txt

echo "parallel taring $dataset into ./svms-data/${dataset}.tar.zst"

tar -I "pzstd -p $total_parallelism" -cf ./svms-data/${dataset}.tar.zst \
   ./svms-data/${dataset}.{train,test}.svm \
   ./svms-data/${dataset}.{train,test}.*{sm,ft}.svm
  
echo "uploading to s3"
aws s3 cp ./svms-data/${dataset}.tar.zst "$dst_s3${dataset}.tar.zst"
for t in train test ; do for s in sm ft ; do
  cp ./svms-data/${dataset}.${t}.1024${s}.svm ./svms-data/saved${dataset}.${t}.1024${s}.svm
done ; done
for s in sm ft ; do 
cp ./svms-data/${dataset}.train.1024${s}.svm.field_dims.txt ./svms-data/saved${dataset}.train.1024${s}.svm.field_dims.txt
done

rm -f ./svms-data/${dataset}.{train,test}.*{sm,ft}.svm \
      ./svms-data/${dataset}.train.*{ft,sm}.svm.field_dims.txt

done # dataset

budget=1024
for dataset in $datasets ; do
for t in train test ; do for s in sm ft ; do
mv ./svms-data/saved${dataset}.${t}.1024${s}.svm ./svms-data/${dataset}.${t}.1024${s}.svm
done ; done
t=train
for s in sm ft ; do
mv ./svms-data/saved${dataset}.${t}.1024${s}.svm.field_dims.txt ./svms-data/${dataset}.${t}.1024${s}.svm.field_dims.txt
done ; done
# echo "target encoding all datasets"

# for dataset in $datasets ; do
# echo \
#   --budget 1024 --compress TargetEncode \
#   --train ./svms-data/${dataset}.train.svm \
#   --valid ./svms-data/${dataset}.test.svm
# done | RAYON_NUM_THREADS=$nthreads xargs -P $num_concurrent_datasets -L 1 ./csl/target/release/csl >/dev/null

echo "binary-encoding all datasets"

for dataset in $datasets ; do
for suffix in ft sm ; do
suffix=".${budget}${suffix}"

echo ./svms-data/${dataset}.train${suffix}.svm
echo ./svms-data/${dataset}.test${suffix}.svm

done
echo ./svms-data/${dataset}.train.svm
echo ./svms-data/${dataset}.test.svm
done  | RAYON_NUM_THREADS=$nthreads xargs -P $num_concurrent_datasets -L 1 ./csl/target/release/svm2bins >/dev/null

echo "pre-computing binary ht for all datasets"

for dataset in $datasets ; do
for t in train test ; do
python hashing_trick.py ./svms-data/${dataset}.${t}.svm $budget
done
done

echo "parallel taring all binary datasets into ./svms-data/binary.tar.zst"

tar -I "pzstd -p $total_parallelism" -cf ./svms-data/binary.tar.zst \
  ./svms-data/{url,kdda,kddb,kdd12}.{train,test}.{data,indices,indptr,y}.bin \
  ./svms-data/{url,kdda,kddb,kdd12}.{train,test}.svm \
  ./svms-data/{url,kdda,kddb,kdd12}.{train,test}.1024{sm,ft,ht}.{data,indices,indptr,y}.bin \
  ./svms-data/{url,kdda,kddb,kdd12}.{train,test}.1024{sm,ft}.svm

echo "uploading to s3"
aws s3 cp ./svms-data/binary.tar.zst "${dst_s3}binary.tar.zst"

rm \
  ./svms-data/{url,kdda,kddb,kdd12}.{train,test}.{data,indices,indptr,y}.bin \
  ./svms-data/{url,kdda,kddb,kdd12}.{train,test}.1024{sm,ft,ht}.{data,indices,indptr,y}.bin \
  ./svms-data/{url,kdda,kddb,kdd12}.{train,test}.1024{sm,ft}.svm \
  ./svms-data/{url,kdda,kddb,kdd12}.train.1024{sm,ft}.svm.field_dims.txt
