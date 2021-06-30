#!/usr/bin/env python

import argparse
from datetime import datetime
from glob import glob
import logging
logger = logging.getLogger('FBFC_LABEL_SIM')
import os
os.environ.update(
    OMP_NUM_THREADS = '1',
    OPENBLAS_NUM_THREADS = '1',
    NUMEXPR_NUM_THREADS = '1',
    MKL_NUM_THREADS = '1',
)

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from timeit import default_timer

from utils.evaluations import get_datasets, eval_est_hv
from ebf.bf import FHBloomFilter as FBFC


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-t', '--n_parallel', help='Number of parallel workers', type=int,
        default=1,
    )
    parser.add_argument(
        '-f', '--exp_factor', help='The expansion factor HP',
        type=float, default=20.,
    )
    parser.add_argument(
        '-s', '--conn_spar', help='The projection density HP',
        type=float, default=0.025,
    )
    parser.add_argument(
        '-w', '--wta_nnz', help='The winner-take-all NNZ HP', type=int,
        default=32,
    )
    parser.add_argument(
        '-c', '--decay_rate', help='The FBF decay rate HP', type=float,
        default=None,
    )
    parser.add_argument(
        '-B', '--max_batch_size', help='Maximum batch size', type=int,
        default=4096,
    )
    parser.add_argument(
        '-R', '--nreps', help='Number of repetitions for each setting',
        type=int, default=1,
    )
    parser.add_argument(
        '-d', '--dname', help='The data set to perform analysis on',
        choices=['digits', 'letter', 'mnist', 'fashion_mnist', 'cifar10', 'cifar100'],
    )
    args = parser.parse_args()
    assert (
        args.decay_rate is None or
        (args.decay_rate > 0. and args.decay_rate < 1.0)
    )
    today = datetime.today()
    timestamp = (
        str(today.year) + str(today.month) + str(today.day) + '.' +
        str(today.hour) + str(today.minute) + str(today.second)
    )
    print('-'*30)
    print('Experiment timestamp: %s' % timestamp)
    print('-'*30)
    res_dir = 'results/fbfc+lsim/' + timestamp
    print('Saving results/logs in \'%s\' ...' % res_dir)
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    fname = '{}/fbfc_label_sim.d_{}.f{:.1f}.s{:.3f}.w{}.c{}.results.{}.csv'.format(
        res_dir, args.dname, args.exp_factor, args.conn_spar, args.wta_nnz,
        args.decay_rate, timestamp
    )
    results_dict = {
        'Label1': [],
        'Label2': [],
        'CS':[],
    }
    data_head = 'PROCESSING ' + args.dname
    hline = '=' * len(data_head)
    print(hline)
    print(data_head)
    print(hline)
    # get dataset
    X, y, _, _ = get_datasets(args.dname)
    X = normalize(X)
    nrows, ndims = X.shape
    nlabels = len(np.unique(y))
    print('[{}] Shape: {} x {}, #labels: {}'.format(args.dname, nrows, ndims, nlabels))
    # obtain hyper-parameters
    wta_ratio = min(0.5, (float(args.wta_nnz) / (args.exp_factor * ndims)))
    fbfc_args = {
        'expansion_factor': args.exp_factor,
        'connection_sparsity': args.conn_spar,
        'wta_ratio': wta_ratio,
        'wta_nnz': args.wta_nnz,
        'batch_size': args.max_batch_size,
        'nthreads': args.n_parallel,
    }
    if args.decay_rate is not None:
        fbfc_args['binary'] = False
        fbfc_args['c'] = args.decay_rate
    print('[{}] Processing FBFC({})'.format(args.dname, str(fbfc_args)))
    for i in range(args.nreps):
        rep_seed = np.random.randint(5489)
        print('Rep seed:', rep_seed)
        fbfc_args['random_state'] = rep_seed
        print('*' * 10)
        fbfc = FBFC(**fbfc_args)
        print('[%s] Starting training ...' % args.dname)
        start_time = default_timer()
        fbfc.fit(X, y)
        stop_time = default_timer()
        training_time = stop_time - start_time
        print(
            '[%s]    ... completed in %g seconds with %i threads'
            % (args.dname, training_time, args.n_parallel)
        )
        per_class_fbfs = fbfc.bloom_filters.astype(np.float32)
        print(per_class_fbfs.shape)
        class_ip_matrix = per_class_fbfs @ np.transpose(per_class_fbfs)
        print(class_ip_matrix.shape)
        for i in range(nlabels):
            for j in range(i):
                ip = class_ip_matrix[i, j]
                l2 = class_ip_matrix[i, i] + class_ip_matrix[j, j] - (2 * ip)
                cs = ip / np.sqrt(class_ip_matrix[i, i] * class_ip_matrix[j, j])
                results_dict['Label1'].append(i)
                results_dict['Label2'].append(j)
                results_dict['IP'].append(ip)
                results_dict['L2'].append(l2)
                results_dict['CS'].append(cs)
        results_df = pd.DataFrame.from_dict(results_dict)
        print('Checkpoint/save results in %s' % fname)
        results_df.to_csv(fname, index=False)
    print(hline)
    print('Results', results_df.shape)
    print(results_df.sort_values(by='IP', axis='rows', ascending=False).head())
    print(results_df.sort_values(by='CS', axis='rows', ascending=False).head())
    print(results_df.sort_values(by='L2', axis='rows', ascending=True).head())
