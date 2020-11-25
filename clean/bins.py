#!/usr/bin/env python
# Given stdin of the form
# <key> <value> <count> <value> <count>...
# computes sorted lower bounds of equidepth bins returned as
# <key> <value> <value>...
# such that for some number x, the 1-index of the last value smaller than x
# is the index of the bin for x
# this means that being "below range" makes you 0-indexed.

import sys
import numpy as np
import fileinput

strs = next(fileinput.input()).split(" ")
values = np.asarray(strs[1::2], dtype=np.float64)
counts = np.asarray(strs[2::2], dtype=np.uint64)
sortix = np.argsort(values)
values = values[sortix]
counts = counts[sortix]

n = counts.sum()
counts = np.cumsum(counts)

import os
nbins = int(os.environ["BIN"]) - 1 # first bin's not free

value_ixs = [0]
ctr = 0
for bin_ix in range(1, nbins + 1):
    bintop = bin_ix * n // nbins
    while counts[ctr] < bintop:
        ctr += 1
    value_ixs.append(ctr)

# anything which takes up over a bin gets a bin to itself
ixs = []
overfull = None
for ix in value_ixs:
    if ixs and ixs[-1] == ix:
        overfull = ix
        continue

    if overfull is not None:
        ixs.append(overfull + 1)

    if overfull != ix:
        ixs.append(ix)

    overfull = None

print(strs[0], end=" ")
np.savetxt(sys.stdout, values[ixs], newline=" ")
print()
