#!/usr/bin/env awk -f
# assumes the first MAXVAL + 1 lines are the bucket mapping.
function ceil(x){return int(x)+(x>int(x))}
function assert(condition, string)
{
    if (! condition) {
        printf("assertion failed: %s\n", string) > "/dev/stderr"
        _assert_exit = 1
        exit 1
    }
}
BEGIN {
    TRUNCATE = TRUNCATE + 0
    MAXVAL = MAXVAL + 0
}
NR <= MAXVAL + 1 {
    feature = $1
    bucket = $2
    idx = $3
    assert(NF == 3, "NF != 3: " $0)
    feature2bucket[feature] = bucket
    feature2idx[feature] = idx
}
NR > MAXVAL + 1 {
    printf "%s", $1
    delete line
    for (i=2; i<=NF; i++) {
        feature = $i + 0
        assert(feature in feature2bucket, feature " not in feature2bucket, max " MAXVAL)
        assert(feature in feature2idx, feature " not in feature2idx, max " MAXVAL)
        assert(feature2bucket[feature] < TRUNCATE, feature2bucket[feature] " not less than TRUNCATE " TRUNCATE)
        assert(feature2idx[feature] <= ceil(MAXVAL / TRUNCATE), feature2idx[feature] " greater than ceil(MAXVAL/TRUNCATE) " ceil(MAXVAL / TRUNCATE))
        line[feature2bucket[feature]] = feature2idx[feature]
    }
    n = asorti(line, sorted, "@ind_num_asc")
    for (i=1; i<=n; i++) {
        bucket = sorted[i]
        idx = line[bucket]
        printf " %s:%s", bucket, idx
    }
    printf "\n"
}
END {
    if (_assert_exit)
        exit 1
}
