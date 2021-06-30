# Flyhash Bloom Filter Classifier

Code for the Fly Bloom Filter Classifier and the scripts for running the experiments presented in the paper


## Setting up environment

This section details the setup of the compute environment for executing the provided scripts.

### Prerequisites

- `python3.8`
- `pip`
- `virtualenv`

### Installing requirements

```
$ mkdir fbfc
$ virtualenv -p /usr/bin/python3.8 fbfc
$ source fbfc/bin/activate
(fbfc) $ export PYTHONPATH=`pwd`
(fbfc) $ pip install --upgrade pip
(fbfc) $ pip install -r requirements.txt
```

## Algorithm


### Binary FBFC learning


- Inputs:
  - Training data `S` of `n` pairs `(x, y)` of features `x` with label `y`
  - FBFC hyper-parameters `m`, `s`, `rho`
- Initialization:
  - For each class `l \in L`


### Non-binary FBFC learning



### FBFC inference




## Citation

Please use the following citation for the paper:
```
TODO
```