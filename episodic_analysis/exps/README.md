# Structure

# Public

- factory.py: creates the respective Experiments based on params

# Internal

## Experiment managers

Simulator based:
- opt.py: OPT AP, OPT+
- static.py: RejectX, CoinFlip
- ml.py: Baleen, OldML
- base.py: ExpSizeWR (others extend this)

Analytical model:
- analysis.py: episodic analysis using Little's Law


# Others

- helpers.py: fitting
