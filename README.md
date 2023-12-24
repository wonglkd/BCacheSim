This repository contains:

1. BCacheSim, a Python simulator with a focus on flash
caching for bulk storage systems.
2. EpisodicAnalysis, a Python module implementing the episodes model
for flash caching and the training of ML admission and ML prefetching models
based on episodes.

Supported admission policies:
- Baleen
- RejectX
- CoinFlip

Supported eviction policies
- LRU
- FIFO

## Directory structure

- cachesim/
    - simulate_ap.py: command line wrapper for for simulator
    - sim_cache.py, admission_policies.py, prefetchers.py: key simulator code
    - testbed/: utilities to benchmark machines for Service Time and launch CacheBench runs
    - stats: C++ utilities that ingest the entire trace and produce stats (to be released)
- episodic_analysis: 
- scripts/: scripts for processing traces

## Usage
```
```
TODO

## Installation

### Conda (recommended)

```
conda env create -f install/env_cachelib-py-3.11.yaml
conda env create -f install/env_cachelib-pypy-3.8.yaml
```

Alternatively:

```
micromamba create -c conda-forge -n cachelib-py-3.11 python=3.11      numpy pandas psutil scipy matplotlib seaborn tqdm lightgbm scikit-learn redis-py jsonargparse retry jupyterlab ipywidgets jupyter_nbextensions_configurator commentjson
micromamba create -c conda-forge -n cachelib-pypy-3.8 python=3.8 pypy numpy pandas psutil scipy matplotlib seaborn tqdm lightgbm scikit-learn redis-py jsonargparse retry jupyterlab ipywidgets jupyter_nbextensions_configurator commentjson
```

Note: scikit-learn only works with PyPy 3.8, not 3.9 yet. LightGBM requires sklearn.
    `Error: No module named 'sklearn.__check_build._check_build'`

### Pip
```
# For simulator
pip install lightgbm numpy pandas scikit-learn
pip install spookyhash jsonargparse compress_json compress_pickle retry
# Optional
pip install psutil
# optional: pympler.tracker
# Scripts
pip install tqdm
# For episodic_analysis
pip install scipy
pip install redis
# For cache-analysis
pip install matplotlib seaborn
```
