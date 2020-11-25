#!/bin/bash
# Reads a bin spec for positive real-valued numbers generated on each
# line by clean/bins.py, of the form
# <key> <x1> <x2>...
# where <xi> denotes the lower bound for bin i
# then for each file in $1, modifies it in place to have features digitized
# for all <key> features according to their bin index.
#
# instead of requiring the lower bound <x1>, we start the first bin at 0.0
# (so all positive values are binned)

set -euo pipefail

autoawk=$(mktemp -p clean/data)
trap "rm -f $autoawk" EXIT
echo "BEGIN {" > $autoawk
# awk that writes awk
awk 'NF > 1 { 
for(i=2;i<=NF;i++) {
  printf "d[\"%s\"][%s]=%s;", $1, i-1, i == 2 ? 0.0 : $i + 0.0
}
printf "\n"
}
' >> $autoawk
echo '
}
{
  for (i = 2; i <= NF; i++) {
    s = index($i, ":")
    feature = substr($i, 1, s - 1)
    value = substr($i, s + 1) + 0.0
    if (!(feature in d)) {
      $i = feature ":" value
      continue
    }
    bin = 0
    while (bin+1 <= length(d[feature]) && value >= d[feature][bin+1])
      bin++
    $i = feature ":" bin
  }
  print
}' >> $autoawk

echo "$1" \
    | parallel --will-cite '
awk -f '"$autoawk"' {} > {}_digitize
mv {}_digitize {}'


