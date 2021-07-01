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

The Flyhash operation ![h](https://render.githubusercontent.com/render/math?math=h%3A%20%5Cmathbb%7BR%7D%5Ed%20%5Cto%20%5C%7B%200%2C%201%20%5C%7D%5Em) for any ![xinR](https://render.githubusercontent.com/render/math?math=x%20%5Cin%20%5Cmathbb%7BR%7D%5Ed) is defined as:

![image](https://render.githubusercontent.com/render/math?math=h(x)%20%3D%20%5CGamma_%5Crho(M_m%5Es%20x)),

where ![mproj](https://render.githubusercontent.com/render/math?math=M_m%5Es%20%5Cin%20%5C%7B0%2C%201%5C%7D%5E%7Bm%20%5Ctimes%20d%7D) is the _sparse_ binary projection matrix with ![sd](https://render.githubusercontent.com/render/math?math=s%20%5Cll%20d) nonzero entries in each row of the matrix, and ![wta](https://render.githubusercontent.com/render/math?math=%5CGamma_%7B%5Crho%7D%3A%20%5Cmathbb%7BR%7D%5Em%20%5Cto%20%5C%7B0%2C%201%5C%7D%5Em) is the _winner-take-all_ operation that sets the top-![rho](https://render.githubusercontent.com/render/math?math=%5Crho) entries in a vector to 1 and the rest of the entries in the vector to 0, where ![rhollm](https://render.githubusercontent.com/render/math?math=%5Crho%20%5Cll%20m).

### Binary FBFC learning

Given a data set ![dset](https://render.githubusercontent.com/render/math?math=S%20%3D%20%5C%7B(x_i%2C%20y_i)%5C%7D_%7Bi%3D1%7D%5En), with ![labs](https://render.githubusercontent.com/render/math?math=y_i%20%5Cin%20%5BL%5D), the per-class **_binary_** Flyhash Bloom Filters (FBFs) ![wl](https://render.githubusercontent.com/render/math?math=w_l%2C%20l%20%5Cin%20%5BL%5D), can be generated as follows:

![wconc](https://render.githubusercontent.com/render/math?math=w_l%20%3D%20%5Cbigwedge_%7B(x%2Cy)%20%5Cin%20S%3A%20y%20%3D%20l%7D%20%5Coverline%7B(h(x))%7D%20%3D%20%5Coverline%7B(%5Cbigvee_%7B(x%2Cy)%20%5Cin%20S%3A%20y%20%3D%20l%7D%20h(x))%7D). This above process can be visualized in a simple 2-class toy classification problem in the following figure:

![FBFC1](https://user-images.githubusercontent.com/5964219/124139537-b6ddda00-da55-11eb-910d-f160d4560328.png)


### Non-binary FBFC learning

The above binary FBFs are not robust to labeling noise, we can use a **_non-binary_** FBF for each class defined as:

![nbwconc](https://render.githubusercontent.com/render/math?math=w_l%20%3D%20(1%20-%20c)%5E%7B%5Codot%20z_l%7D%2C%20z_l%20%3D%20%5Csum_%7B(x%2Cy)%20%5Cin%20S%3A%20y%20%3D%20l%7D%20h(x)),

where ![cin](https://render.githubusercontent.com/render/math?math=c%20%5Cin%20(0%2C%201)%20%5Ccup%20%5C%7B1%5C%7D) is the _FBF decay rate_ that controls how much impact one example has, with ![c1](https://render.githubusercontent.com/render/math?math=c%20%3D%201) corresponding to the binary FBF. 

### FBFC inference

For a test point ![xr](https://render.githubusercontent.com/render/math?math=x%20%5Cin%20%5Cmathbb%7BR%7D%5Ed), the predicted label ![yL](https://render.githubusercontent.com/render/math?math=%5Chat%7By%7D%20%5Cin%20%5BL%5D) is done as follows:

![ymin](https://render.githubusercontent.com/render/math?math=%5Chat%7By%7D%20%5Cgets%20%5Carg%20%5Cmin_%7Bl%20%5Cin%20%5BL%5D%7D%20w_l%5E%5Ctop%20h(x)).

The inference can be visualized in the above toy classification problem as follows:

![FBFC2](https://user-images.githubusercontent.com/5964219/124139831-f60c2b00-da55-11eb-9a41-61933fb6c589.png)


## Experiments

The experiments for the paper are detailed in [expts.md](expts.md).

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
