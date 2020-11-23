# Chromatic Encoding

Dimensionality reduction for discrete data encodings.

## Environment Setup

[Install Rust](https://www.rust-lang.org/tools/install).

Next, install [Anaconda 3](https://www.anaconda.com/distribution/) and add it to your path.

**To be done when you clone or move this repo**:
```
conda env create -f environment.yaml
```

**Should be done once per session:**
```
source activate env2020
```

**To save new dependencies**:
```
conda env export --no-builds | grep -v "prefix: " > environment.yaml
```

**To update out-of-date local environment with new dependencies in the `environment.yaml` file**:
```
conda env update -f environment.yaml --prune
```

```
# all new envs need my patch here
pip uninstall torchfm
pip install --force-reinstall git+https://github.com/vlad17/pytorch-fm/
```

For GPU machines, you'll need to install a gpu version of pytorch over this env.

We will also need various tools for parsing/etc:

```
sudo apt install -y vowpal-wabbit datamash zstd nodejs npm 
npm install -g relaxed-json
```

## Usage

All scripts are intended to be run from the repository root (the directory containing this `README.md` file).

All scripts read and write to S3 to cache intermediate data if not available on the local FD, using prefix `S3ROOT` which must be set as an environment variable. `DATASETS` defines which datasets to operate on, and should be a space-delimited string containing words `urltoy url kdda kddb kdd12`. The table below summarizes what each script does.

```
export S3ROOT="s3://sisu-datasets/ce-build"
export DATASETS="urltoy url kdda kddb kdd12"
```

| Script | Description |
| --- | --- |
| `bash raw/run.sh` | download raw datasets |
