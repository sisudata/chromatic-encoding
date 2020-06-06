#!/bin/bash vw_split_experiments.sh

set -euo pipefail

results=svms-data/split-results.jsonl
if [ -f $results ]; then
  echo "moving existing $results to /tmp"
  mv $results /tmp
fi 

src_s3="s3://sisu-datasets/binary-svms/"

if ! [ -f ./svms-data/splits.tar.zst ] ; then
  aws s3 cp "${src_s3}splits.tar.zst" ./svms-data/splits.tar.zst
fi

nthreads=64
tar -I "pzstd -p $nthreads" -xf ./svms-data/splits.tar.zst

bits=$(seq 10 18)
splits=$(seq 1 5)
dataset=url

for i in $bits ; do 
budget=$((1 << $i))

quiet="yes"
delete="no"
quadratic="no"
echo vw.sh $budget $dataset ft $quiet $delete $quadratic
for split in $splits ; do
echo vw.sh $budget $dataset sm $quiet $delete $quadratic ".split${split}"
done    
done | xargs -P $nthreads -L1 /bin/bash >> $results

rm ./svms-data/${dataset}.{train,test}.{sm,ft}*.svm
