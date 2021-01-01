#!/usr/bin/env awk -f
BEGIN {
    m = 0
}
{
    for (i=1; i<=NF; i++) {
        feature = $i + 0
        if (feature > m) m = feature
    }
}
END {
    print m
}
