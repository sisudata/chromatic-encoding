#!/bin/bash

set -euo pipefail

if ! git diff-index --quiet HEAD -- encode/ nn/ pycrank/ setup.py requirements.txt ; then
    echo 'current branch has changes in encode/ or nn/'
    exit 1
fi 

sha=$(git rev-parse HEAD)

ray exec distributed/ce-cluster.yaml "mkdir -p ~/$sha/{nn,encode}/data" 1>/dev/null 2>/dev/null

for i in $(git ls-files common.sh encode/ nn/ pycrank/ setup.py requirements.txt) ; do
    ray rsync_up distributed/ce-cluster.yaml "$i" "~/$sha/$i" 1>/dev/null 2>/dev/null 
done

AWS_ACCESS_KEY_ID=$(python -c 'import boto3;s = boto3.Session(); print(s.get_credentials().access_key)')
AWS_DEFAULT_REGION=$(python -c 'import boto3;s = boto3.Session(); print(s.region_name or "us-west-2")')
AWS_SECRET_ACCESS_KEY=$(python -c 'import boto3;s = boto3.Session(); print(s.get_credentials().secret_key)')

function remote_exec() {
    remotecmd="$1"
    shift

    awscreds="AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID AWS_DEFAULT_REGION=$AWS_DEFAULT_REGION AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY"
    ray exec distributed/ce-cluster.yaml "cd ~/$sha && $awscreds $remotecmd" "$@"
}

remote_exec 'S3ROOT="'"$S3ROOT"'" DATASETS="'"$DATASETS"'" ENCODINGS="'"$ENCODINGS"'" TRUNCATES="'"$TRUNCATES"'" bash encode/run.sh'
remote_exec 'pip install --force-reinstall .'
remote_exec 'RAY_ADDRESS="auto" S3ROOT="'"$S3ROOT"'" DATASETS="'"$DATASETS"'" ENCODINGS="'"$ENCODINGS"'" TRUNCATES="'"$TRUNCATES"'" MODELNAMES="wd" bash nn/run.sh --force' "$@"
