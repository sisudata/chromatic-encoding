#!/bin/bash
# Validates whether the given file is in svmlight format (technically,
# a slightly stricter version with only 0/1 labels, and either no values
# (all values are implicitly 1.0 for every feature) or values everywhere.
#
# For explicit and binary formats, normalizes floats in-place.

set -euo pipefail

fn="$1"
features_explicitness="$2"

awk '
!($1 == "0" || $1 == "1") { 
  print "'"$fn"' invalid label line", NR, $0 > "/dev/stderr"
  exit 1
}' $fn

if [ "$features_explicitness" = "explicit" ] ; then
awk '{
  for (i = 2; i <= NF; ++i) {
    s = index($i, ":")
    feature = substr($i, 1, s - 1)
    value = substr($i, s + 1) + 0.0
    if (!s || !(feature ~ /^[1-9][0-9]*$/) || value == 0.0) {
      print "'"$fn"' invalid feature", $i, "line", NR, $0 > "/dev/stderr"
      exit 1
    }
    $i = feature ":" value
  }
  print
}
' $fn > ${fn}_valid_svm
mv ${fn}_valid_svm $fn
elif [ "$features_explicitness" = "implicit" ] ; then
awk '{
  for (i = 2; i <= NF; ++i) {
    if (index($i, ":") || !($i ~ /^[1-9][0-9]*$/)) {
      print "'"$fn"' invalid feature", $i, "line", NR, $0 > "/dev/stderr"
      exit 1
    }
  }
}
' $fn
elif [ "$features_explicitness" = "binary" ] ; then
awk '{
  for (i = 2; i <= NF; ++i) {
    s = index($i, ":")
    feature = substr($i, 1, s - 1)
    value = substr($i, s + 1) + 0.0
    if (!s || !(feature ~ /^[1-9][0-9]*$/) || value != 1.0) {
      print "'"$fn"' invalid feature", $i, "line", NR, $0 > "/dev/stderr"
      exit 1
    }
    $i = feature ":" 1
  }
  print
}
' $fn > ${fn}_valid_svm
mv ${fn}_valid_svm $fn
else
    echo invalid setting for features explicitness "$features_explicitness" >&2
    exit 1
fi
