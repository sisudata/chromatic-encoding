#!/bin/bash
#
# Like colorize.sh, but only works on the URL dataset and
# investigates different split rates
#
# computes encodings for 2^10 up to 2^18 the text format of ft and sm
# and for sm does splits 2**(-x) for x in 1, ..., 5
# and uploads as ${dataset}.tar.zst
#
# then for budget = 1024 does te ft and sm, all available in text
# and binary formats, uploading to binary.tar.zst

set -euo pipefail
    
bits=$(seq 10 18)
splits=$(seq 1 5)
nthreads=64
num_concurrent_datasets=1
total_parallelism=$(( $nthreads * $num_concurrent_datasets))
dst_s3="s3://sisu-datasets/binary-svms/"

dataset=url

for split in $splits ; do
for t in train test ; do      
  ln ./svms-data/${dataset}.${t}.svm ./svms-data/${dataset}.split${split}.${t}.svm
done
done

for i in $bits ; do 
budget=$((1 << $i))

for compress_suffix in "FrequencyTruncation ft" "SubmodularExpansion sm"; do 
compress=$(echo $compress_suffix | cut -d" " -f1)

if [ "$compress" = "FrequencyTruncation" ]; then
  continue
  echo "**************************************"
  echo " RUNNING FT @ ${budget} "
  echo "**************************************"
  prefix="./svms-data/${dataset}."
  RAYON_NUM_THREADS=$nthreads ./csl/target/release/csl \
                     --budget $budget --compress ${compress} \
                     --train ${prefix}train.svm \
                     --valid ${prefix}test.svm > /dev/null
else
  for split in $splits ; do
  echo "**************************************"
  echo " RUNNING SM @ ${budget} w/ split ${split}"
  echo "**************************************"
  prefix="./svms-data/${dataset}.split${split}."
  RAYON_NUM_THREADS=$nthreads ./csl/target/release/csl \
                     --budget $budget --compress ${compress} \
                     --train ${prefix}train.svm \
                     --split-rate ${split} \
                     --valid ${prefix}test.svm > /dev/null
  done
fi

done
done

rm ./svms-data/${dataset}.split*.{train,test}.svm

tar -I "pzstd -p $total_parallelism" -cf ./svms-data/splits.tar.zst \
  ./svms-data/${dataset}.{train,test}.svm \
  ./svms-data/${dataset}.{train,test}.ft*.svm \
  ./svms-data/${dataset}.split*.{train,test}.sm*.svm

aws s3 cp ./svms-data/splits.tar.zst "${dst_s3}splits.tar.zst"

rm \
  ./svms-data/${dataset}.{train,test}.ft*.svm \
  ./svms-data/${dataset}.split*.{train,test}.sm*.svm
