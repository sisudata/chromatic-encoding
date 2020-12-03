#!/bin/bash
# Expects dataset tar files to be present in clean/data already.
# For each dataset/encoding pair, generates a CSR matrix
# ${dataset}.${encoding}.{train,test}.{data,indices,indptr,y}
# where the parameters above are all equal-length native
# arrays of type u32.
#
# Also creates newline-delimited json records for each pair
# ${dataset}.${encoding}.jsonl.
#
# Expects S3ROOT, DATASETS, ENCODINGS to be set.

set -euo pipefail

echo encoding $DATASETS with $ENCODINGS
echo cache root "$S3ROOT"

# ENCODINGS = ft greedy

# ft: just filter ft indices, recode to :1 with awk

# GREEDy -> convert to separate cases, eg freq ordering or degree ordering
