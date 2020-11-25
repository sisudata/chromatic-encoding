#!/bin/bash
# Given a list of svmlight files representing a single training
# dataset, prints out those features which take on more than
# MAXCARD distinct values (maxcard should be >= 1)

set -euo pipefail

if ! [ "$MAXCARD" -ge 1 ]; then
    echo "MAXCARD=$MAXCARD too small" >&2
    exit 1
fi

map='{
  for (i = 2; i <= NF; ++i) {
    s = index($i, ":")
    feature = substr($i, 1, s - 1)
    if (uniques[feature] > '"$MAXCARD"') continue
    value = substr($i, s + 1) + 0.0
    if (feature in present && value in present[feature]) continue
    present[feature][value] = 1
    uniques[feature]++
  }
}
END {
  for (x in uniques) 
    if (uniques[x] > '"$MAXCARD"')
      print x
  for (x in uniques) {
    if (uniques[x] <= '"$MAXCARD"' && length(present[x]) > 0) {
      printf "%s", x
      for (v in present[x])
        printf " %s", v
      printf "\n"
    } 
  }
}
'

echo "$1" \
    | parallel --will-cite awk "'$map'" \
    | awk 'NF > 1 && !($1 in frequent) {
  for (i = 2; i <= NF; ++i) {
    value = $i + 0.0
    if (($1, value) in present) continue
    present[$1, value] = 1
    uniques[$1]++
    if (uniques[$1] > '"$MAXCARD"') {
      frequent[$1] = 1
      break
    }
  }
}
NF == 1 {
  frequent[$1] = 1
}
END {
  for (x in frequent)
    print x
}
'
