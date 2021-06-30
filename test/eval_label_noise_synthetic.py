#!/usr/bin/env python

import argparse
from datetime import datetime
import logging
logger = logging.getLogger('LABELNOISE')
import os
os.environ.update(
    OMP_NUM_THREADS = '1',
    OPENBLAS_NUM_THREADS = '1',
    NUMEXPR_NUM_THREADS = '1',
    MKL_NUM_THREADS = '1',
)

from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from timeit import default_timer

from ebf.bf import FHBloomFilter as FBFC
from utils.evaluations import eval_est_hv
from eval_synthetic_data import create_dataset


def eval_fbfc(
        dset, fbf_decay_rate, wta_nnz, conn_density, exp_factor, batch_size,
        nthreads, nreps
):
    print('=' * 35)
    print('{} Starting FBFC(ef={:.3f},cd={:3f},rho={},c={:.1f}) eval ...'.format(
        ' '*7, exp_factor, conn_density, wta_nnz, fbf_decay_rate
    ))
    X, y, vX, vy = dset
    nrows, ndims = X.shape
    wta_ratio = float(wta_nnz) / (exp_factor * ndims)
    fbfc_args = {
        'expansion_factor': exp_factor,
        'connection_sparsity': conn_density,
        'wta_ratio': wta_ratio,
        'wta_nnz': wta_nnz,
        'batch_size': batch_size,
        'nthreads': nthreads,
        'binary': not (fbf_decay_rate < 1.0),
        'c': fbf_decay_rate,
    }
    eval_time = 0.
    acc_list = []
    f1mac_list = []
    f1mic_list = []
    for i in range(nreps):
        rep_seed = np.random.randint(5489)
        print(
            '{} Rep {:d} with seed {:d} with train set ({}x{}) and '
            'test set ({}x{})'.format(
                (' '*15), i + 1, rep_seed, X.shape[0], X.shape[1], vX.shape[0], vX.shape[1]
            )
        )
        # Initializing FBFC
        fbfc = FBFC(**{**fbfc_args, **{'random_state': rep_seed}})

        # evaluating FBFC
        start_time = default_timer()
        (acc, f1a, f1i), w = eval_est_hv(fbfc, dset)
        if acc is None:
            assert f1a is None and f1i is None
            print('{} Configuration appears to have failed ...'.format(' '*7))
            return None, None, None
        stop_time = default_timer()
        eval_time += (stop_time - start_time)
        acc_list.append(acc)
        f1mac_list.append(f1a)
        f1mic_list.append(f1i)
    # Average over repetitions
    eval_time /= float(nreps)
    print('{}    ... completed eval in {:.2f} secs per {} evals'.format(
        ' '*7, eval_time, nreps
    ))
    print('{}... with accuracy: {:.5f} ({})'.format(
        ' '*8, np.mean(acc_list), np.array(acc_list)
    ))
    print('-' * 35)
    return np.mean(acc_list), np.mean(f1mac_list), np.mean(f1mic_list)


def eval_fbfc_on_dset(
        dset, hp_props, non_binary=False
):
    c = 0.9 if non_binary else 1.0
    acc, _, _ = eval_fbfc(
        dset, c, hp_props['wta_nnz'], hp_props['conn_spar'],
        hp_props['exp_factor'], 1024, 1, 10
    )
    return acc


def get_all_methods(cmd_args):
    hp_properties = {
        'exp_factor': cmd_args.exp_factor,
        'conn_spar': cmd_args.conn_spar,
        'wta_nnz': cmd_args.wta_nnz,
    }
    # methods
    return [
        (
            'FBFC',
            lambda d: eval_fbfc_on_dset(
                d, hp_props=hp_properties,
            )
        ),
        (
            'FBFC*',
            lambda d: eval_fbfc_on_dset(
                d, hp_props=hp_properties, non_binary=True
            )
        ),
    ]


def corrupt_labels(labels, nclasses, flip_y):
    assert 0. <= flip_y <= 1.
    assert np.min(labels) == 0 and np.max(labels) == nclasses - 1
    nswitches = 0
    new_labels = [l for l in labels]
    for i in range(len(labels)):
        if np.random.uniform(0, 1) > flip_y:
            continue
        o = new_labels[i]
        n = (o + np.random.randint(1, nclasses-1)) % nclasses
        new_labels[i] = n
        nswitches += 1
    logger.debug(f'{nswitches}/{len(labels)} labels switched')
    return np.array(new_labels)    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-t', '--n_parallel', help='Number of parallel workers', type=int,
        default=1
    )
    parser.add_argument(
        '-f', '--exp_factor', help='the expansion factor HP',
        type=float,
    )
    parser.add_argument(
        '-s', '--conn_spar',
        help='the connection sparsity HP', type=float,
    )
    parser.add_argument(
        '-w', '--wta_nnz',
        help='the winner-take-all ratio HP', type=int,
    )
    parser.add_argument(
        '-S', '--n_train', help='Number of rows in the data set', type=int,
        default=1000
    )
    parser.add_argument(
        '-N', '--n_test', help='Number of rows in the data set', type=int,
        default=300
    )
    parser.add_argument(
        '-d', '--n_cols', help='Number of columns in the data set', type=int,
        default=30
    )
    parser.add_argument(
        '-L', '--n_classes', help='Number of classes in the data set', type=int,
        default=5
    )
    parser.add_argument(
        '-C', '--n_clusters_per_class', help='Number of clusters per class',
        type=int, default=8
    )
    parser.add_argument(
        '-R', '--n_reps', help='Number of repetitions', type=int, default=5
    )
    args = parser.parse_args()
    today = datetime.today()
    timestamp = (
        str(today.year) + str(today.month) + str(today.day) + '.' +
        str(today.hour) + str(today.minute) + str(today.second)
    )
    print('-'*30)
    print('Experiment timestamp: %s' % timestamp)
    print('-'*30)
    res_dir = 'results/label-noise/' + timestamp
    print('Saving results/logs in \'%s\' ...' % res_dir)
    out_filename = (
        f'{res_dir}/noise'
        f'.f{args.exp_factor}.s{args.conn_spar}.w{args.wta_nnz}'
        f'.S{args.n_train}.N{args.n_test}.d{args.n_cols}.L{args.n_classes}'
        f'.C{args.n_clusters_per_class}.R{args.n_reps}.{timestamp}.csv'
    )
    print(f'Saving relative performance in \'{out_filename}\'')
    methods = get_all_methods(args)
    noises = [0.0, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25]
    dsets = [
        create_dataset(
            args.n_train + args.n_test, args.n_cols,
            args.n_classes, args.n_clusters_per_class,
        ) for i in range(args.n_reps)
    ]
    def process_rep(noise, rep_idx, dset, nclasses):
        X, y = dset
        X1, X2, y1, y2 = train_test_split(X, y, test_size=args.n_test, stratify=y)
        n_y1 = corrupt_labels(y1, nclasses, noise)
        ret = [noise]
        dset = (X1, n_y1, X2, y2)
        for m, efunc in methods:
            print(f'[R{rep_idx}] Processing \'{m}\' with noise {noise} ...')
            acc = efunc(dset)
            ret += [m, acc]
            print(f'[R{rep_idx}] Processing \'{m}\' with noise {noise}: {acc}')
        return ret
    # --
    allres = Parallel(n_jobs=args.n_parallel)(
        delayed(process_rep)(n, i, dsets[i], args.n_classes)
        for n in noises for i in range(args.n_reps)
    )
    for l in allres: print(l)
    for l in allres:
        assert l[1] == 'FBFC' and l[3] == 'FBFC*'
    df_res = pd.DataFrame.from_dict({
        'noise': [l[0] for l in allres],
        'improvement': [((l[4]/l[2]) - 1.)*100. for l in allres],
        'fbfc': [l[2] for l in allres],
        'fbfc*': [l[4] for l in allres],
    })
    print(df_res.head(10))
    
    print(f'Saving relative performance in \'{out_filename}\'')
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    df_res.to_csv(out_filename, index=False)
