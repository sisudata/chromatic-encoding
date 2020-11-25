#!/bin/bash
# Given a list of svmlight files representing a single dataset
# does a very dumb wordcount on the feature-value pairs (no value normalization
# is performed, make sure all floats are represented with the same string).

set -euo pipefail

map='{
  for (i = 2; i <= NF; ++i) {
    counts[$i]++
  }
}
END {
  for (x in counts)
    print x, counts[x]
}
'

reduce='{
  counts[$1] += $2
}
END {
  for (x in counts)
    print x, counts[x]
}
'

echo "$1" \
    | parallel -j -1 --will-cite awk "'$map'" \
    | awk "$reduce"
