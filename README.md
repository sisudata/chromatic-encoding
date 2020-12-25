# Chromatic Encoding

Dimensionality reduction for discrete data encodings.

## Environment Setup

[Install Rust](https://www.rust-lang.org/tools/install).

Next, for any Python at least version `3.6`.

```
pip install numpy jupyter scikit-learn scipy tqdm torch ray[tune]
pip install --force-reinstall git+https://github.com/vlad17/pytorch-fm/
```

This environment does NOT assume GPUs (and will train NNs with CPUs). GPU training is performed through the ray cluster launcher on AWS.

We will also need various tools for parsing/etc:

```
sudo apt install -y vowpal-wabbit datamash zstd moreutils gawk
sudo apt install -y nodejs npm && npm install -g relaxed-json # TODO: do i still need this?
# TODO gnu parallel -> rust parallel
# https://github.com/mmstick/parallel
```

## Usage

All scripts are intended to be run from the repository root (the directory containing this `README.md` file). Environment variables set parameters for all scripts (though some scripts ignore some parameters which don't apply to them).

Everything is S3-backed so if you need disk space you can remove each script's adjacent `data/*` files and
re-running will fetch from S3 on-demand. Some of the scripts support `--force` which will move old data
files to `/tmp` locally and some `old/` folder in S3 before continuing.

* All scripts read and write to S3 to cache intermediate data if not available on the local FD, using prefix `S3ROOT` which must be set as an environment variable.
* `DATASETS` defines which datasets to operate on, and should be a space-delimited string containing words `urltoy url kdda kddb kdd12`.
* `ENCODINGS` specifies the different encodings to try, `ft` (feature truncation) or `ce` (chromatic encoding). The table below summarizes what each script does. TODO: ht encoding
* `TRUNCATES` specifies the feature truncation limiting input width

```
export S3ROOT="s3://sisu-datasets/ce-build"
export DATASETS="url kdda kddb kdd12"
export ENCODINGS="ft ce"
export TRUNCATES="1000 10000"
```

| Script | Description |
| --- | --- |
| `bash raw/run.sh` | download raw datasets |
| `bash clean/run.sh` | clean datasets |
| `bash graph/run.sh` | generate co-occurrence graphs |
| `bash encode/run.sh` | encode datasets up to prescribed dimension |
| `bash nn/run.sh` | neural net train/test on encoded datasets |
| `bash wabbit/run.sh` | vowpal wabbit train/test on clean datasets |
| `bash distributed/run.sh` | run neural net experiments in parallel |

Looks like this re-implements Metaflow, basically. Whoops.

### Additional Parameters

For most GPUs, I would not recommend setting any `TRUNCATES` over `100000`.

Default AWS credentials are assumed available for S3 and EC2 access.

#### Neural Networks

For hyperparameter optimization and execution, `nn/run.sh` looks for `RAY_ADDRESS` to be set (can be set to `auto` for ray to automatically connect to ray if it's running on the same node). If unset, `nn/run.sh` runs in a debug/toy mode with light settings.

Invoking `nn/run.sh` will only run hyperparameter optimization (HPO) in parallel, but will run different encoding/dataset/truncation triplets serially

#### Vowpal Wabbit 

`wabbit/run.sh` allows datasets of the form `${encoding}_${truncate}_${dataset}`, e.g., `ce_1000_url` for debug purposes.

#### Distributed Execution

Use `distributed/run.sh` to after starting a cluster to launch all network jobs for
dataset/encoding/truncate triplets in parallel on the same ray cluster (with nested HPO parallelism).
A lot of the convenience is owed to `ray` here.

The head is cpu-only but has lots of disk, which means:
  * you should pre-run the `encode/run.sh` step on a machine with lots of CPUs to save preprocessed data to S3.
  * before execution datasets will be downloaded (or generated on-demand if missing) on head node.
  * the cluster will wind down to just this node after inactivity, so you only pay for GPUs during use.

Convenient commands:
```
# start the cluster
#
# I've found I had to pin worker counts to a constant
# to get things going then drop off to min worker set to 0, tune doesn't "kick"
# the autoscaler into activity on its own.
ray up distributed/ce-cluster.yaml --yes --min-workers=8 --max-workers=8

# launch a dataset synchronously in the local shell
# you can modify the script to add a --tmux for async
# I launch one of these per dataset in parallel
ENCODINGS="ft" TRUNCATES="1024" DATASETS="url" bash distributed/run.sh

# drop min workers back down for wind-down after launch
ray up distributed/ce-cluster.yaml --yes --no-restart --min-workers=0

# port forward a health dashboard
ray dashboard distributed/ce-cluster.yaml

# wind down cluster
ray down distributed/ce-cluster.yaml
```
