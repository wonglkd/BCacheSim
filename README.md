# BCacheSim: Cache Simulator specialized for flash caching in bulk storage

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

## Installation

You may install packages via Conda or Pip, which will be sufficient to run the simulator.

For further research and development, you may wish to use a cluster manager to run many experiments in parallel (you can write an adaptor to your preferred one by modifying episodic_analysis/local_cluster.py). I use brooce with my experiment filesystem mounted on NFS -- you can clone my [fork](https://github.com/wonglkd/brooce) if you wish. This is not necessary to run basic experiments.

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
pip install -r install/requirements.txt
```

which is equivalent to

```
# For simulator
pip install lightgbm numpy pandas scikit-learn
pip install spookyhash jsonargparse compress_json compress_pickle retry commentjson
# Optional
pip install psutil ipywidgets
# optional: pympler.tracker
# Scripts
pip install tqdm
# For episodic_analysis
pip install scipy
pip install redis
# For cache-analysis
pip install matplotlib seaborn
# Advanced policies
pip install pqdict
```

## Datasets

Traces are available at https://ftp.pdl.cmu.edu/pub/datasets/Baleen24/. We ask that academic works using any code or traces to cite Baleen[^1] and, if appropriate, CacheLib [^2] and Tectonic [^3].

For the Baleen-FAST24 repository (meant for those trying to reproduce results in the Baleen paper), please see https://github.com/wonglkd/Baleen-FAST24.

## Contact

For further questions, please contact [Daniel Lin-Kit Wong](https://wonglkd.fi-de.net/).

## References

[^1]: **Baleen: ML Admission & Prefetching for Flash Caches** <br>
      Daniel Lin-Kit Wong, Hao Wu, Carson Molder, Sathya Gunasekar, Jimmy Lu, Snehal Khandkar, Abhinav Sharma, Daniel S. Berger, Nathan Beckmann, Gregory R. Ganger <br>
      *USENIX FAST 2024*

[^2]: **The CacheLib Caching Engine: Design and Experiences at Scale** <br>
      Benjamin Berg, Daniel S. Berger, Sara McAllister, Isaac Grosof, Sathya Gunasekar, Jimmy Lu, Michael Uhlar, Jim Carrig, Nathan Beckmann, Mor Harchol-Balter, and Gregory R. Ganger <br>
      *USENIX OSDI 2020*

[^3]: **Facebook's Tectonic Filesystem: Efficiency from Exascale** <br>
      Satadru Pan, Theano Stavrinos, Yunqiao Zhang, Atul Sikaria, Pavel Zakharov, Abhinav Sharma, Mike Shuey, Richard Wareing, Monika Gangapuram, Guanglei Cao, Christian Preseau, Pratap Singh, Kestutis Patiejunas, and JR Tipton, Ethan Katz-Bassett, and Wyatt Lloyd <br>
      *USENIX FAST 2021*
