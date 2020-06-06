 #!/bin/bash
#
# binary xgb run. if bits is present then uses
# chromatic learning, else runs against sparse
#
# args: dataset quiet(yes/n) bits

dataset="$1"
quiet="$2"
bits="$3"

if [ -z "$bits" ]; then
    suffix=""
else
    suffix=".sm$((1 << $bits))"
fi

conditional_quiet() {
    if [ "$quiet" = "yes" ]; then
        "$@" > /dev/null
    else
        "$@"
    fi
}

load_str='
def load():
    return load_svmlight_file("'"svms-data/${dataset}.train${suffix}.svm"'")
'

/usr/bin/time -f "%M" python xgb.py 1 0 0 svms-data/bits${bits}_${dataset}.model "${load_str}" 2>/tmp/xgb_bits${bits}_${dataset}.memory | conditional_quiet cat

mem_kb=$(cat /tmp/xgb_bits${bits}_${dataset}.memory)
mem_mb=$(echo "$mem_kb / 1024" | bc -l | awk '{printf "%.6f\n", $0}')

conditional_quiet python xgb_eval.py svms-data/bits${bits}_${dataset}.model "${load_str/.train/.test}"
