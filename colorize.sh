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
    
bits="10 12 14 16 18"
nthreads=15
num_concurrent_datasets=6
total_parallelism=$(( $nthreads * $num_concurrent_datasets))
dst_s3="s3://sisu-datasets/unbiased-svms/"
datasets="url kdda"

cutoff=Double
echo "compressing $datasets across bits $(echo $bits) cutoff $cutoff"

for dataset in $datasets ; do

for i in $bits ; do 
budget=$((1 << $i))
flags="--k 1 --cutoff-style $cutoff --glauber-samples 10000000"

for compress_suffix in "FrequencyTruncation ft" "SubmodularExpansion sm"; do 
compress=$(echo $compress_suffix | cut -d" " -f1)

echo \
  --budget $budget --compress ${compress} \
  --train ./svms-data/${dataset}.train.svm \
  --valid ./svms-data/${dataset}.test.svm $flags
done
done
done | shuf | RAYON_NUM_THREADS=$nthreads xargs -P $num_concurrent_datasets -L 1 ./csl/target/release/csl >svms-data/log-colorize-${cutoff}.txt

# echo "parallel taring $dataset into ./svms-data/${dataset}.tar.zst"

# tar -I "pzstd -p $total_parallelism" -cf ./svms-data/${dataset}.tar.zst \
#   ./svms-data/${dataset}.{train,test}.svm \
#   ./svms-data/${dataset}.{train,test}.{sm,ft}*.svm
  
# echo "uploading to s3"
# aws s3 cp ./svms-data/${dataset}.tar.zst "$dst_s3${dataset}.tar.zst"
# for t in train test ; do for s in sm ft ; do
# cp ./svms-data/${dataset}.${t}.${s}1024.svm ./svms-data/saved${dataset}.${t}.${s}1024.svm
# done ; done
# rm ./svms-data/${dataset}.{train,test}.{un,ft}*.svm \
#    ./svms-data/${dataset}.{train,test}.{ft,ht}*.svm.field_dims.txt
  
#done # dataset

exit 0

budget=1024
for dataset in $datasets ; do
for t in train test ; do for s in sm ft ; do
mv ./svms-data/saved${dataset}.${t}.${s}1024.svm ./svms-data/${dataset}.${t}.${s}1024.svm
done ; done

echo "target encoding all datasets"

for dataset in $datasets ; do
echo \
  --budget 1024 --compress TargetEncode \
  --train ./svms-data/${dataset}.train.svm \
  --valid ./svms-data/${dataset}.test.svm
done | RAYON_NUM_THREADS=$nthreads xargs -P $num_concurrent_datasets -L 1 ./csl/target/release/csl >/dev/null


echo "binary-encoding all datasets"

for dataset in $datasets ; do
for suffix in ft sm te ; do
suffix=".${suffix}${budget}"

echo ./svms-data/${dataset}.train${suffix}.svm
echo ./svms-data/${dataset}.test${suffix}.svm

done
echo ./svms-data/${dataset}.train.svm
echo ./svms-data/${dataset}.test.svm
done | RAYON_NUM_THREADS=$nthreads xargs -P $num_concurrent_datasets -L 1 ./csl/target/release/svm2bins >/dev/null

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
  ./svms-data/{url,kdda,kddb,kdd12}.{train,test}.{sm,te,ft,ht}1024.{data,indices,indptr,y}.bin \
  ./svms-data/{url,kdda,kddb,kdd12}.{train,test}.{sm,te,ft}1024.svm

echo "uploading to s3"
aws s3 cp ./svms-data/binary.tar.zst "${dst_s3}binary.tar.zst"

rm \
  ./svms-data/{url,kdda,kddb,kdd12}.{train,test}.{data,indices,indptr,y}.bin \
  ./svms-data/{url,kdda,kddb,kdd12}.{train,test}.{sm,te,ft,ht}1024.{data,indices,indptr,y}.bin \
  ./svms-data/{url,kdda,kddb,kdd12}.{train,test}.{sm,te,ft,ht}1024.svm \
  ./svms-data/{url,kdda,kddb,kdd12}.{train,test}.{sm,te,ft,ht}1024.svm.field_dims.txt
