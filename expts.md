# Code for submission

Table of contents:
- [Setting up environment](#setting-up-environment)
  - [Prerequisites](#prerequisites)
  - [Installing requirements](#installing-requirements)
- [Running experiments](#running-experiments)
  - [Running experiments on hyper-parameter dependence](#running-experiments-on-hyper-parameter-dependence)
  - [Running comparison to baselines with synthetic data](#running-comparison-to-baselines-with-synthetic-data)
  - [Running comparison to baselines with OpenML data](#running-comparison-to-baselines-with-openml-data)
  - [Generating class similarities](#generating-class-similarities)

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

## Running experiments

This section provides the precise commandline arguments for the various scripts to generate results for the different experiments conducted.

### Running experiments on hyper-parameter dependence

#### Evaluation script options
```
(fbfc) $ python test/eval_hpdep_kfold_small_data.py --help
usage: eval_hpdep_kfold_small_data.py [-h] [-t N_PARALLEL] [-F N_FOLDS] [-e EXP_FACTOR_LB]
                                      [-E EXP_FACTOR_UB] [-s CONN_SPAR_LB]
                                      [-S CONN_SPAR_UB] [-w WTA_NNZ_LB] [-W WTA_NNZ_UB]
                                      [-c NB_C_LB] [-C NB_C_UB] [-H {ef,cs,wn,c}]
                                      [-n NVALS_FOR_HP]

optional arguments:
  -h, --help            show this help message and exit
  -t N_PARALLEL, --n_parallel N_PARALLEL
                        Number of parallel workers
  -F N_FOLDS, --n_folds N_FOLDS
                        Number of folds
  -e EXP_FACTOR_LB, --exp_factor_lb EXP_FACTOR_LB
                        Lower bound on the expansion factor HP
  -E EXP_FACTOR_UB, --exp_factor_ub EXP_FACTOR_UB
                        Upper bound on the expansion factor HP
  -s CONN_SPAR_LB, --conn_spar_lb CONN_SPAR_LB
                        Lower bound on the connection sparsity HP
  -S CONN_SPAR_UB, --conn_spar_ub CONN_SPAR_UB
                        Upper bound on the connection sparsity HP
  -w WTA_NNZ_LB, --wta_nnz_lb WTA_NNZ_LB
                        Lower bound on the winner-take-all NNZ HP
  -W WTA_NNZ_UB, --wta_nnz_ub WTA_NNZ_UB
                        Upper bound on the winner-take-all NNZ HP
  -c NB_C_LB, --nb_c_lb NB_C_LB
                        Lower bound for 'c' in the non-binary bloom filter
  -C NB_C_UB, --nb_c_ub NB_C_UB
                        Upper bound for 'c' in the non-binary bloom filter
  -H {ef,cs,wn,c}, --hp2tune {ef,cs,wn,c}
                        The hyper-parameter to tune while fixing the rest
  -n NVALS_FOR_HP, --nvals_for_hp NVALS_FOR_HP
                        The number of values to try for the hyper-parameter
```

#### Exact commandline arguments

Set `<NTHREADS>` as per the resources (number of threads, memory) available.

##### FlyHash dimension `m`
```
(fbfc) $ python test/eval_hpdep_kfold_small_data.py -t <NTHREADS> -F 10 -e 4 -E 2048 -s 0.1 -S 0.3 -w 8 -W 32 -c 0.5 -C 1.0 -H ef -n 10
```

##### Projection density `s`
```
(fbfc) $ python test/eval_hpdep_kfold_small_data.py -t <NTHREADS> -F 10 -e 256 -E 1024 -s 0.1 -S 0.8 -w 8 -W 32 -c 0.5 -C 1.0 -H cs -n 10
```

##### FlyHash NNZ `\rho`
```
(fbfc) $ python test/eval_hpdep_kfold_small_data.py -t <NTHREADS> -F 10 -e 256 -E 1024 -s 0.1 -S 0.3 -w 4 -W 256 -c 0.5 -C 1.0 -H wn -n 10
```

##### FBFC decay rate `c`
```
(fbfc) $ python test/eval_hpdep_kfold_small_data.py -t <NTHREADS> -F 10 -e 256 -E 1024 -s 0.1 -S 0.3 -w 8 -W 32 -c 0.2 -C 0.9 -H c -n 10
```

### Running comparison to baselines with synthetic data

Set the `<NTHREADS>` based on the compute resources available.

#### Synthetic binary data
We will be trying the following configurations:
- `<NDIMS> = 100, <NNZ_PER_ROW> = 10, <NSAMPLES> = 1000`
- `<NDIMS> = 100, <NNZ_PER_ROW> = 20, <NSAMPLES> = 1000`
- `<NDIMS> = 100, <NNZ_PER_ROW> = 30, <NSAMPLES> = 1000`
- `<NDIMS> = 100, <NNZ_PER_ROW> = 40, <NSAMPLES> = 1000`

##### Evaluation script options
```
(fbfc) $ python test/eval_synthetic_binary_data.py --help
usage: eval_synthetic_binary_data.py [-h] [-t N_PARALLEL] [-F N_FOLDS] [-c MAX_CALLS]
                                     [-r N_RANDOM_STARTS] [-f EXP_FACTOR_UB]
                                     [-s CONN_SPAR_UB] [-w WTA_NNZ_UB] [-S N_ROWS]
                                     [-d N_COLS] [-L N_CLASSES] [-C N_CLUSTERS_PER_CLASS]
                                     [-W NNZ_PER_ROW] [-R N_REPS]

optional arguments:
  -h, --help            show this help message and exit
  -t N_PARALLEL, --n_parallel N_PARALLEL
                        Number of parallel workers
  -F N_FOLDS, --n_folds N_FOLDS
                        Number of folds
  -c MAX_CALLS, --max_calls MAX_CALLS
                        Maximum number of calls for GP
  -r N_RANDOM_STARTS, --n_random_starts N_RANDOM_STARTS
                        Number of random start points in GP
  -f EXP_FACTOR_UB, --exp_factor_ub EXP_FACTOR_UB
                        Upper bound on the expansion factor HP
  -s CONN_SPAR_UB, --conn_spar_ub CONN_SPAR_UB
                        Upper bound on the connection sparsity HP
  -w WTA_NNZ_UB, --wta_nnz_ub WTA_NNZ_UB
                        Upper bound on the winner-take-all ratio HP
  -S N_ROWS, --n_rows N_ROWS
                        Number of rows in the data set
  -d N_COLS, --n_cols N_COLS
                        Number of columns in the data set
  -L N_CLASSES, --n_classes N_CLASSES
                        Number of classes in the data set
  -C N_CLUSTERS_PER_CLASS, --n_clusters_per_class N_CLUSTERS_PER_CLASS
                        Number of clusters per class
  -W NNZ_PER_ROW, --nnz_per_row NNZ_PER_ROW
                        Number of NNZ per row in the binary data set
  -R N_REPS, --n_reps N_REPS
                        Number of repetitions
```
##### Exact commandline arguments
```
(fbfc) $ python test/eval_synthetic_binary_data.py -t <NTHREADS> -F 10 -c 60 -r 15 -f 2048.0 -s 0.5 -w 256 -S <NSAMPLES> -d <NDIMS> -L 5 -C 3 -W <NNZ_PER_ROW> -R 30
```

#### Synthetic continuous data
We will be trying the following configurations:
- `<NDIMS> = 100, <NSAMPLES> = 1000`

##### Evaluation script options
```
(fbfc) $ python test/eval_synthetic_data.py --help
usage: eval_synthetic_data.py [-h] [-t N_PARALLEL] [-F N_FOLDS] [-c MAX_CALLS]
                              [-r N_RANDOM_STARTS] [-f EXP_FACTOR_UB] [-s CONN_SPAR_UB]
                              [-w WTA_NNZ_UB] [-S N_ROWS] [-d N_COLS] [-L N_CLASSES]
                              [-C N_CLUSTERS_PER_CLASS] [-R N_REPS]

optional arguments:
  -h, --help            show this help message and exit
  -t N_PARALLEL, --n_parallel N_PARALLEL
                        Number of parallel workers
  -F N_FOLDS, --n_folds N_FOLDS
                        Number of folds
  -c MAX_CALLS, --max_calls MAX_CALLS
                        Maximum number of calls for GP
  -r N_RANDOM_STARTS, --n_random_starts N_RANDOM_STARTS
                        Number of random start points in GP
  -f EXP_FACTOR_UB, --exp_factor_ub EXP_FACTOR_UB
                        Upper bound on the expansion factor HP
  -s CONN_SPAR_UB, --conn_spar_ub CONN_SPAR_UB
                        Upper bound on the connection sparsity HP
  -w WTA_NNZ_UB, --wta_nnz_ub WTA_NNZ_UB
                        Upper bound on the winner-take-all ratio HP
  -S N_ROWS, --n_rows N_ROWS
                        Number of rows in the data set
  -d N_COLS, --n_cols N_COLS
                        Number of columns in the data set
  -L N_CLASSES, --n_classes N_CLASSES
                        Number of classes in the data set
  -C N_CLUSTERS_PER_CLASS, --n_clusters_per_class N_CLUSTERS_PER_CLASS
                        Number of clusters per class
  -R N_REPS, --n_reps N_REPS
                        Number of repetitions
```
##### Exact commandline arguments
```
(fbfc) $ python test/eval_synthetic_data.py -t <NTHREADS> -F 10 -c 60 -r 15 -f 2048.0 -s 0.5 -w 256 -S <NSAMPLES> -d <NDIMS> -L 5 -C 3 -R 30
```

#### Robustness of label noise with synthetic continuous data
We will be trying the following configurations:
- `<NDIMS> = 100, <NSAMPLES> = 10000, <NTEST> = 1000`

##### Evaluation script options
```
(fbfc) $ python test/eval_label_noise_synthetic.py --help
usage: eval_label_noise_synthetic.py [-h] [-t N_PARALLEL] [-f EXP_FACTOR] [-s CONN_SPAR]
                                     [-w WTA_NNZ] [-S N_TRAIN] [-N N_TEST] [-d N_COLS]
                                     [-L N_CLASSES] [-C N_CLUSTERS_PER_CLASS] [-R N_REPS]

optional arguments:
  -h, --help            show this help message and exit
  -t N_PARALLEL, --n_parallel N_PARALLEL
                        Number of parallel workers
  -f EXP_FACTOR, --exp_factor EXP_FACTOR
                        the expansion factor HP
  -s CONN_SPAR, --conn_spar CONN_SPAR
                        the connection sparsity HP
  -w WTA_NNZ, --wta_nnz WTA_NNZ
                        the winner-take-all ratio HP
  -S N_TRAIN, --n_train N_TRAIN
                        Number of rows in the data set
  -N N_TEST, --n_test N_TEST
                        Number of rows in the data set
  -d N_COLS, --n_cols N_COLS
                        Number of columns in the data set
  -L N_CLASSES, --n_classes N_CLASSES
                        Number of classes in the data set
  -C N_CLUSTERS_PER_CLASS, --n_clusters_per_class N_CLUSTERS_PER_CLASS
                        Number of clusters per class
  -R N_REPS, --n_reps N_REPS
                        Number of repetitions
```
##### Exact commandline arguments
```
(fbfc) $ python test/eval_label_noise_synthetic.py -t <NTHTREADS> -f 1024 -s 0.05 -w 256 -S <NSAMPLES> -N <NTEST> -d <NDIMS> -L 5 -C 3 -R 10
```


### Running comparison to baselines with OpenML data

Two sets of OpenML data sets:
- `<MIN_DATA_DIM> = 10, <MAX_DATA_DIM> = 100, <MAX_DATA_SAMPLES> = 50000`
- `<MIN_DATA_DIM> = 101, <MAX_DATA_DIM> = 1024, <MAX_DATA_SAMPLES> = 10000`

Based on resources available:
- Set `--n_parallel/-t <NTHREADS>` based on number of threads `<NTHREADS>` available to process that many data sets in parallel.

#### Running `kNNC` baseline
##### Evaluation script options
```
(fbfc) $ python test/knn_baseline.py --help
usage: knn_baseline.py [-h] [-t N_PARALLEL] [-F N_FOLDS] [-n MIN_DATA_DIM]
                       [-x MAX_DATA_DIM] [-S MAX_DATA_SAMPLES] [-K MAX_K]

optional arguments:
  -h, --help            show this help message and exit
  -t N_PARALLEL, --n_parallel N_PARALLEL
                        Number of parallel workers
  -F N_FOLDS, --n_folds N_FOLDS
                        Number of folds
  -n MIN_DATA_DIM, --min_data_dim MIN_DATA_DIM
                        Minimum data dimensionality on OpenML
  -x MAX_DATA_DIM, --max_data_dim MAX_DATA_DIM
                        Maximum data dimensionality on OpenML
  -S MAX_DATA_SAMPLES, --max_data_samples MAX_DATA_SAMPLES
                        Maximum number of samples in data on OpenML
  -K MAX_K, --max_k MAX_K
                        Maximum k for kNNC
```
##### Exact commandline arguments
```
(fbfc) $ python test/knn_baseline.py -t <NTHREADS> -F 10 -n <MIN_DATA_DIM> -x <MAX_DATA_DIM> -S <MAX_DATA_SAMPLES>
```

#### Running `CC1/CC` baseline
##### Evaluation script options
```
(fbfc) $ python test/cc_baseline.py --help
usage: cc_baseline.py [-h] [-t N_PARALLEL] [-F N_FOLDS] [-n MIN_DATA_DIM] [-x MAX_DATA_DIM]
                      [-S MAX_DATA_SAMPLES] [-M {CC1,CC}]

optional arguments:
  -h, --help            show this help message and exit
  -t N_PARALLEL, --n_parallel N_PARALLEL
                        Number of parallel workers
  -F N_FOLDS, --n_folds N_FOLDS
                        Number of folds
  -n MIN_DATA_DIM, --min_data_dim MIN_DATA_DIM
                        Minimum data dimensionality on OpenML
  -x MAX_DATA_DIM, --max_data_dim MAX_DATA_DIM
                        Maximum data dimensionality on OpenML
  -S MAX_DATA_SAMPLES, --max_data_samples MAX_DATA_SAMPLES
                        Maximum number of samples in data on OpenML
  -M {CC1,CC}, --method {CC1,CC}
                        Whether to use 'CC1' or 'CC' baseline
```
##### Exact commandline arguments for `CC1`
```
(fbfc) $ python test/cc_baseline.py -t <NTHREADS> -F 10 -n <MIN_DATA_DIM> -x <MAX_DATA_DIM> -S <MAX_DATA_SAMPLES> -M CC1
```
##### Exact commandline arguments for `CC`
```
(fbfc) $ python test/cc_baseline.py -t <NTHREADS> -F 10 -n <MIN_DATA_DIM> -x <MAX_DATA_DIM> -S <MAX_DATA_SAMPLES> -M CC
```

#### Running `SBFC` baseline
##### Evaluation script options
```
(fbfc) $ python test/sbf_baseline.py --help
usage: sbf_baseline.py [-h] [-t N_PARALLEL] [-F N_FOLDS] [-n MIN_DATA_DIM]
                       [-x MAX_DATA_DIM] [-S MAX_DATA_SAMPLES]
                       [-E EXPANSION_FACTOR_UB]

optional arguments:
  -h, --help            show this help message and exit
  -t N_PARALLEL, --n_parallel N_PARALLEL
                        Number of parallel workers
  -F N_FOLDS, --n_folds N_FOLDS
                        Number of folds
  -n MIN_DATA_DIM, --min_data_dim MIN_DATA_DIM
                        Minimum data dimensionality on OpenML
  -x MAX_DATA_DIM, --max_data_dim MAX_DATA_DIM
                        Maximum data dimensionality on OpenML
  -S MAX_DATA_SAMPLES, --max_data_samples MAX_DATA_SAMPLES
                        Maximum number of samples in data on OpenML
  -E EXPANSION_FACTOR_UB, --expansion_factor_ub EXPANSION_FACTOR_UB
                        Upper bound on the factor with which to project up
```
##### Exact commandline arguments
```
(fbfc) $ python test/sbf_baseline.py -t <NTHREADS> -F 10 -n <MIN_DATA_DIM> -x <MAX_DATA_DIM> -S <MAX_DATA_SAMPLES> -E 2048.0
```

#### Running `LR` baseline
##### Evaluation script options
```
(fbfc) $ python test/lr_baseline.py --help
usage: lr_baseline.py [-h] [-t N_PARALLEL] [-F N_FOLDS] [-n MIN_DATA_DIM] [-x MAX_DATA_DIM]
                      [-S MAX_DATA_SAMPLES] [-C NUM_VALS_FOR_C]

optional arguments:
  -h, --help            show this help message and exit
  -t N_PARALLEL, --n_parallel N_PARALLEL
                        Number of parallel workers
  -F N_FOLDS, --n_folds N_FOLDS
                        Number of folds
  -n MIN_DATA_DIM, --min_data_dim MIN_DATA_DIM
                        Minimum data dimensionality on OpenML
  -x MAX_DATA_DIM, --max_data_dim MAX_DATA_DIM
                        Maximum data dimensionality on OpenML
  -S MAX_DATA_SAMPLES, --max_data_samples MAX_DATA_SAMPLES
                        Maximum number of samples in data on OpenML
  -C NUM_VALS_FOR_C, --num_vals_for_C NUM_VALS_FOR_C
                        Number of values to try for the regularization parameter
```
##### Exact commandline arguments
```
(fbfc) $ python test/lr_baseline.py -t <NTHREADS> -F 10 -n <MIN_DATA_DIM> -x <MAX_DATA_DIM> -S <MAX_DATA_SAMPLES> -C 20
```

#### Running `MLPC` baseline
##### Evaluation script options
```
(fbfc) $ python test/mlpc_baseline.py --help
usage: mlpc_baseline.py [-h] [-t N_PARALLEL] [-F N_FOLDS] [-n MIN_DATA_DIM] [-x MAX_DATA_DIM]
                        [-S MAX_DATA_SAMPLES] [-A NUM_VALS_FOR_ALPHA] [-B NUM_VALS_FOR_BATCH_SIZE]

optional arguments:
  -h, --help            show this help message and exit
  -t N_PARALLEL, --n_parallel N_PARALLEL
                        Number of parallel workers
  -F N_FOLDS, --n_folds N_FOLDS
                        Number of folds
  -n MIN_DATA_DIM, --min_data_dim MIN_DATA_DIM
                        Minimum data dimensionality on OpenML
  -x MAX_DATA_DIM, --max_data_dim MAX_DATA_DIM
                        Maximum data dimensionality on OpenML
  -S MAX_DATA_SAMPLES, --max_data_samples MAX_DATA_SAMPLES
                        Maximum number of samples in data on OpenML
  -A NUM_VALS_FOR_ALPHA, --num_vals_for_alpha NUM_VALS_FOR_ALPHA
                        Number of values to try for the regularization parameter
  -B NUM_VALS_FOR_BATCH_SIZE, --num_vals_for_batch_size NUM_VALS_FOR_BATCH_SIZE
                        Number of values to try for batch size
```
##### Exact commandline arguments
```
(fbfc) $ python test/mlpc_baseline.py -t <NTHREADS> -F 10 -n <MIN_DATA_DIM> -x <MAX_DATA_DIM> -S <MAX_DATA_SAMPLES> -A 5  -B 4
```

#### Running `FBFC`
##### Evaluation script options
```
(fbfc) $ python test/fbfc_hpo.py --help
usage: fbfc_hpo.py [-h] [-t N_PARALLEL] [-F N_FOLDS] [-c MAX_CALLS] [-r N_RANDOM_STARTS]
                   [-f EXP_FACTOR_UB] [-s CONN_SPAR_UB] [-w WTA_NNZ_UB] [-D DONE_SET_RE]
                   [-N NON_BINARY] [-n MIN_DATA_DIM] [-x MAX_DATA_DIM]
                   [-S MAX_DATA_SAMPLES] [-B MAX_BATCH_SIZE]

optional arguments:
  -h, --help            show this help message and exit
  -t N_PARALLEL, --n_parallel N_PARALLEL
                        Number of parallel workers
  -F N_FOLDS, --n_folds N_FOLDS
                        Number of folds
  -c MAX_CALLS, --max_calls MAX_CALLS
                        Maximum number of calls for GP
  -r N_RANDOM_STARTS, --n_random_starts N_RANDOM_STARTS
                        Number of random start points in GP
  -f EXP_FACTOR_UB, --exp_factor_ub EXP_FACTOR_UB
                        Upper bound on the expansion factor HP
  -s CONN_SPAR_UB, --conn_spar_ub CONN_SPAR_UB
                        Upper bound on the projection density HP
  -w WTA_NNZ_UB, --wta_nnz_ub WTA_NNZ_UB
                        Upper bound on the winner-take-all NNZ HP
  -D DONE_SET_RE, --done_set_re DONE_SET_RE
                        Regex for the files corresponding to datasets already processed
  -N NON_BINARY, --non_binary NON_BINARY
                        Whether to use non-binary FBFs
  -n MIN_DATA_DIM, --min_data_dim MIN_DATA_DIM
                        Minimum data dimensionality on OpenML
  -x MAX_DATA_DIM, --max_data_dim MAX_DATA_DIM
                        Maximum data dimensionality on OpenML
  -S MAX_DATA_SAMPLES, --max_data_samples MAX_DATA_SAMPLES
                        Maximum number of samples in data on OpenML
  -B MAX_BATCH_SIZE, --max_batch_size MAX_BATCH_SIZE
                        Maximum batch size for FBFC training
```
- Set `--max_batch_size/-B <MAX_BATCH_SIZE>` based on amount of memory available; large values lead to larger memory overheads but faster execution times.
##### Exact commandline arguments for `FBFC`
```
(fbfc) $ python test/fbfc_hpo.py -t <NTHREADS> -F 10 -c 60 -r 15 -f 2048.0 -s 0.5 -w 256 -n <MIN_DATA_DIM> -x <MAX_DATA_DIM> -S <MAX_DATA_SAMPLES>
```
##### Exact commandline arguments for `FBFC*`
- Set `--non_binary/-N` to `True` to evaluate (non-binary) `FBFC*`; by default, the script evaluates (binary) `FBFC` respectively. Specifically
```
(fbfc) $ python test/fbfc_hpo.py -t <NTHREADS> -F 10 -c 60 -r 15 -f 2048.0 -s 0.5 -w 256 -n <MIN_DATA_DIM> -x <MAX_DATA_DIM> -S <MAX_DATA_SAMPLES> -N True
```

### Generating class similarities
#### Evaluation script options
```
(fbfc) $ python test/fbfc_label_sim.py --help
usage: fbfc_label_sim.py [-h] [-t N_PARALLEL] [-f EXP_FACTOR] [-s CONN_SPAR] [-w WTA_NNZ] [-c DECAY_RATE]
                         [-B MAX_BATCH_SIZE] [-R NREPS]
                         [-d {digits,letter,mnist,fashion_mnist,cifar10,cifar100}]

optional arguments:
  -h, --help            show this help message and exit
  -t N_PARALLEL, --n_parallel N_PARALLEL
                        Number of parallel workers
  -f EXP_FACTOR, --exp_factor EXP_FACTOR
                        The expansion factor HP
  -s CONN_SPAR, --conn_spar CONN_SPAR
                        The projection density HP
  -w WTA_NNZ, --wta_nnz WTA_NNZ
                        The winner-take-all NNZ HP
  -c DECAY_RATE, --decay_rate DECAY_RATE
                        The FBF decay rate HP
  -B MAX_BATCH_SIZE, --max_batch_size MAX_BATCH_SIZE
                        Maximum batch size
  -R NREPS, --nreps NREPS
                        Number of repetitions for each setting
  -d {digits,letter,mnist,fashion_mnist,cifar10,cifar100}, --dname {digits,letter,mnist,fashion_mnist,cifar10,cifar100}
                        The data set to perform analysis on
```
#### Exact commandline arguments
- Set `--n_parallel/-t <NTHREADS>` based on number of threads `<NTHREADS>` available to use that many threads for each `FBFC*` training.
- Set `--max_batch_size/-B <MAX_BATCH_SIZE>` based on amount of memory available; large values lead to larger memory overheads but faster execution times.
```
(fbfc) $ python test/fbfc_label_sim.py -t <NTHREADS> -f 217 -s 0.026 -w 26 -c 0.49 -B <MAX_BATCH_SIZE> -R 30 -d mnist
(fbfc) $ python test/fbfc_label_sim.py -t <NTHREADS> -f 1512 -s 0.5 -w 256 -c 0.9 -B <MAX_BATCH_SIZE> -R 30 -d letter
```
