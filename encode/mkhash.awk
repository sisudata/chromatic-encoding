#!/usr/bin/env awk -f
# assumint an input of shuf -i 0-MAXVAL, generates
# feature bucket idx output
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
}
{
    assert(NF == 1, "NF != 1: " $0);
    feature = i++
    r = $1 + 0
    bucket = r % TRUNCATE
    idx = ++bucketsize[bucket]
    print feature " " bucket " " idx
}
END {
    if (_assert_exit)
        exit 1
}
