# Chromatic Learning Materials

Python notebooks require some environment set up, specified below.

## Data download

I re-used my old downloading code, so you'll need to run the entire `urls.ipynb` notebook and then the
`urls-eval.ipynb` notebook before being able to use the main evalutaion notebook, `svms.ipynb`,
which relies on data downloaded and sorted/split from those two notebooks.

## Python Environment Setup for Notebooks

First, install [Anaconda 3](https://www.anaconda.com/distribution/) and add it to your path.


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
