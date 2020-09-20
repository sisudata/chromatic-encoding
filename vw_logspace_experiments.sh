#!/bin/bash vw_logspace_experiments.sh

set -euo pipefail

results=svms-data/vw-results.jsonl
if [ -f $results ]; then
  echo "moving existing $results to /tmp"
  mv $results /tmp
fi 

bits=$(seq 10 18)
src_s3="s3://sisu-datasets/glauber/"
nthreads=64

for dataset in "$@" ; do
    echo sequencing dataset $dataset
done

for dataset in "$@" ; do # url kdda kddb kdd12 ; do
    
if ! [ -f ./svms-data/${dataset}.tar.zst ] ; then
  aws s3 cp "${src_s3}${dataset}.tar.zst" ./svms-data/${dataset}.tar.zst
fi

tar -I "pzstd -p $nthreads" -xf ./svms-data/${dataset}.tar.zst

for i in $bits ; do 
budget=$((1 << $i))

quiet="yes"
delete="no"
quadratic="no"
echo vw.sh $budget $dataset ht $quiet $delete $quadratic
echo vw.sh $budget $dataset sm $quiet $delete $quadratic
echo vw.sh $budget $dataset ft $quiet $delete $quadratic
done | xargs -P $nthreads -L1 /bin/bash >> $results

rm ./svms-data/${dataset}.{train,test}.*{sm,ft}.svm
   
done
