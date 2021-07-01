# Flyhash Bloom Filter Classifier

Code for the Fly Bloom Filter Classifier (FBFC) and the scripts for running the experiments presented in the paper **_Fruit-fly Inspired Neighborbood Encoding for Classification_** at SIGKDD conference on KDD, 2021.


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

### Flyhash

The Flyhash operation is defined as:


### Binary FBFC learning



### Non-binary FBFC learning



### FBFC inference




## Citation

Please use the following citation for the paper:
```
@inproceedings{sinha2021fruitfly,
  title={Fruit-fly Inspired Neighborbood Encoding for Classification},
  author={Sinha, Kaushik and Ram, Parikshit},
  booktitle={To appear in the Proceedings of the 27th ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining},
  year={2021}
}
```
