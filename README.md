# Chromatic Encoding

Dimensionality reduction for discrete data encodings.

## Environment Setup

[Install Rust](https://www.rust-lang.org/tools/install).

Next, for any Python at least version `3.6`.

```
pip install numpy jupyter scikit-learn scipy tqdm torch
pip uninstall torchfm
pip install --force-reinstall git+https://github.com/vlad17/pytorch-fm/
```

This environment does NOT assume GPUs (and will train NNs with CPUs). GPU training is performed through the ray cluster launcher on AWS.

We will also need various tools for parsing/etc:

```
sudo apt install -y vowpal-wabbit datamash zstd nodejs npm moreutils gawk
npm install -g relaxed-json
# TODO gnu parallel -> rust parallel
# https://github.com/mmstick/parallel
```

## Usage

All scripts are intended to be run from the repository root (the directory containing this `README.md` file).

All scripts read and write to S3 to cache intermediate data if not available on the local FD, using prefix `S3ROOT` which must be set as an environment variable. `DATASETS` defines which datasets to operate on, and should be a space-delimited string containing words `urltoy url kdda kddb kdd12`. `ENCODINGS` specifies the different encodings to try, `ft` (feature truncation) or `weight` (chromatic encoding). The table below summarizes what each script does. TODO: weight -> ce, add ht encoding.

```
export S3ROOT="s3://sisu-datasets/ce-build"
export DATASETS="urltoy url kdda kddb kdd12"
export ENCODINGS="ft weight"
```

| Script | Description |
| --- | --- |
| `bash raw/run.sh` | download raw datasets |
| `bash clean/run.sh` | clean datasets |
| `bash encode/run.sh` | encode datasets with specified encodings under (TODO) default parameters |
| `bash nn/run.sh` | neural net training on encoded datasets |
