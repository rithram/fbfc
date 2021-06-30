# Flyhash Bloom Filter Classifier

Code for the Fly Bloom Filter Classifier (FBFC) and the scripts for running the experiments presented in the paper


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

![img](http://www.sciweavers.org/tex2img.php?eq=h%3A%20%5Cmathbb%7BR%7D%5Ed%20%5Cto%20%5C%7B0%2C%201%5C%7D%5Em%2C%20h%28x%29%20%3D%20%5CGamma_%5Crho%28M_m%5Es%20x%29&bc=White&fc=Black&im=png&fs=18&ff=ccfonts,eulervm&edit=0)

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
