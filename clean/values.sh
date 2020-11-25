#!/bin/bash
# Reads a dictionary from stdin, words separated by newlines.
# for every argument file, in parallel, construct a list
# of the distinct values and the frequency of occurrence.
# The output is of the form
# <feature> <value> <frequency> <value> <frequency>...

set -euo pipefail

autoawk=$(mktemp -p clean/data)
trap "rm -f $autoawk" EXIT
echo "BEGIN {" > $autoawk
awk 'NF { printf "d[\"%s\"]=1;", $1}' >> $autoawk
echo '
} 
{
  for (i = 2; i <= NF; ++i) {
    s = index($i, ":")
    feature = substr($i, 1, s - 1)
    if (!(feature in d)) continue
    value = substr($i, s + 1) + 0.0
    values[feature][value]++
  }
}
END {
  for (feature in values) {
    printf "%s", feature
    for (value in values[feature])
      printf " %s %s", value, values[feature][value]
    printf "\n"
  }
}
' >> $autoawk

echo "$1" \
    | parallel -j -1 --will-cite awk -f $autoawk \
    | awk '
{
  for (i=2; i<=NF; i+=2) {
      count = i + 1
      values[$1][$i] += $count
  }
}
END {
  for (feature in values) {
    printf "%s", feature
    for (value in values[feature])
      printf " %s %s", value, values[feature][value]
    printf "\n"
  }
}'
