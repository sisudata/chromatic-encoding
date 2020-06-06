#!/bin/bash
#
# binary vw run with budget for either hash
# or feature expansion
#
# args: budget dataset compress quiet(yes/n) delete quadratic [split] [parsimonious]
# if neurons is 0 then uses a linear model
# if delete = yes then deletes dataset when complete
# if quadratic = yes
#
# compress should be 'ht' (hashing trick, suffix is empty string
# to get dataset file)
#
# else one of sm, te, ft, sn, ss, ns
#
# Note that for ht the budget should be a power of 2

budget="$1"
dataset="$2"
compress="$3"
quiet="$4"
delete="$5"
quadratic="$6"
split="$7"
parsimonious="$8"
tmpdir=vw-data

if [ "$compress" = "ht" ]; then
    suffix=""
    hash="--hash all"
else
    suffix=".${parsimonious}${compress}${budget}"
    hash=""
fi

power2=$(python -c "print(2 ** (($budget).bit_length() - 1) == $budget)")
if [ "$compress" = "ht" ] && [ "$power2" = "False" ]; then
    exit 0
fi

log2ceil=$(python -c "print(($budget - 1).bit_length())")

# for quadratic we use large HT width for pre-budgeted datasets
# in principle we'd not want to use VW, which hashes anyway, but then
# i'd need to implement a polynomial algorithm that does
# variable-width rebalancing of its terms.
if [ "$quadratic" = "yes" ]; then
    # these results are super noisy, you'd need to tune
    # reglarization (vw-hyperopt) to eval this
    poly="-q ::"
    log2ceil=22
    prefixpoly="quadratic"
else
    poly=""
    prefixpoly="linear"
fi

conditional_quiet() {
    if [ "$quiet" = "yes" ]; then
        "$@" > /dev/null
    else
        "$@"
    fi
}

if [ -z "$split" ] || [ "$split" = "no" ] ; then
    split_data=""
    split=""
else
    split_data="${split}"
fi

prefix="b${budget}${prefixpoly}${split}${parsimonious}"

cache=$(mktemp -p $tmpdir -t ${prefix}${dataset}${suffix}.train.cache.XXXXXXXXXXXX )
test_cache=$(mktemp -p $tmpdir -t ${prefix}${dataset}${suffix}.test.cache.XXXXXXXXXXXX )

cat svms-data/${dataset}${split_data}.train${suffix}.svm | \
  sed -e 's/^0/-1 |/' | \
  sed -e 's/^1/1 |/' | \
  /usr/bin/time -f "%e %M" \
    vw --bit_precision ${log2ceil} $hash \
      --cache_file $cache -k \
      --loss_function logistic $poly \
      --final_regressor $tmpdir/${prefix}${dataset}${suffix}.vw.model 2>&1 | \
    conditional_quiet tee $tmpdir/${prefix}${dataset}${suffix}.vw.model.log
elapsed_kb=$(tail -1 $tmpdir/${prefix}${dataset}${suffix}.vw.model.log)

loss=$(cat svms-data/${dataset}${split_data}.test${suffix}.svm | sed -e 's/^0/-1 |/' | sed -e 's/^1/1 |/' | \
  vw --binary --testonly \
  --cache_file $test_cache -k \
  -i $tmpdir/${prefix}${dataset}${suffix}.vw.model 2>&1 | grep "average loss" )

loss=$(echo $loss | cut -d"=" -f2)
loss=$(echo "1 - $loss" | bc -l | awk '{printf "%.6f\n", $0}')

train_loss=$(cat svms-data/${dataset}${split_data}.train${suffix}.svm | sed -e 's/^0/-1 |/' | sed -e 's/^1/1 |/' | \
  vw --binary --testonly --cache_file $cache \
  -i $tmpdir/${prefix}${dataset}${suffix}.vw.model 2>&1 | grep "average loss" )

train_loss=$(echo $train_loss | cut -d"=" -f2)
train_loss=$(echo "1 - $train_loss" | bc -l | awk '{printf "%.6f\n", $0}')

test_logloss=$(cat svms-data/${dataset}${split_data}.test${suffix}.svm | sed -e 's/^0/-1 |/' | sed -e 's/^1/1 |/' | \
  vw --loss_function logistic --testonly --cache_file $test_cache \
  -i $tmpdir/${prefix}${dataset}${suffix}.vw.model 2>&1 \
  | grep "average loss" )
test_logloss=$(echo $test_logloss | cut -d"=" -f2)

train_logloss=$(cat svms-data/${dataset}${split_data}.train${suffix}.svm | sed -e 's/^0/-1 |/' | sed -e 's/^1/1 |/' | \
  vw --loss_function logistic --testonly --cache_file $cache \
  -i $tmpdir/${prefix}${dataset}${suffix}.vw.model 2>&1 \
  | grep "average loss" )
train_logloss=$(echo $train_logloss | cut -d"=" -f2)

elapsed_sec=$(echo $elapsed_kb | cut -d" " -f1)

# mrss is very crude and doesn't make sense... text is longer
# than heap in some examples.
mem_kb=$(echo $elapsed_kb | cut -d" " -f2)
mem_mb=$(echo "$mem_kb / 1024" | bc -l | awk '{printf "%.6f\n", $0}')

sz=$(cat svms-data/${dataset}${split_data}.train${suffix}.svm | wc -l)
szt=$(cat svms-data/${dataset}${split_data}.test${suffix}.svm | wc -l)

cat <<EOF | tr '\n' ' ' | sed '$s/ $/\n/' # finish newline
{
"learner": "vw-${prefixpoly}",
"split": "${split}",
"budget": ${budget},
"train_examples": ${sz:-null},
"test_examples": ${szt-:null},
"dataset": "${dataset}",
"compress": "${parsimonious}${compress}",
"train_sec": ${elapsed_sec:-null},
"train_acc": ${train_loss:-null},
"test_acc": ${loss:-null},
"train_logloss": ${train_logloss/n.a./null},
"test_logloss": ${test_logloss/n.a./null}
}
EOF

rm $cache $test_cache $tmpdir/${prefix}${dataset}${suffix}.vw.model

if [ "$delete" = "yes" ] ; then
    rm svms-data/${dataset}.train${suffix}.svm svms-data/${dataset}.test${suffix}.svm
fi
