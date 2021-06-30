#!/usr/bin/env python

import argparse
from datetime import datetime
import itertools
import logging
logger = logging.getLogger('MLPC')
import os
os.environ.update(
    OMP_NUM_THREADS = '1',
    OPENBLAS_NUM_THREADS = '1',
    NUMEXPR_NUM_THREADS = '1',
    MKL_NUM_THREADS = '1',
)
import warnings

from joblib import Parallel, delayed
import numpy as np
from openml.datasets import get_dataset
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import StratifiedKFold as SKFold
from sklearn.neural_network import MLPClassifier as MLPC
from sklearn.preprocessing import normalize
from tqdm import tqdm

from utils.evaluations import eval_est_kfold_cv, get_openml_data_list


def get_mlpc_ests(nvals_for_alpha=5, nvals_for_batch_size=4):
    nlayers = [1, 2]
    nnodes = [16, 64, 128]
    hls = [
        {'hidden_layer_sizes': tuple([nn] * nl)}
        for nl, nn in itertools.product(nlayers, nnodes)
    ]
    acts = [{'activation': a} for a in ['tanh', 'relu']]
    alphas = [{'alpha': a} for a in np.geomspace(
        2**-10, 2**10, num=nvals_for_alpha, endpoint=True
    ).tolist()]
    bsizes = [{'batch_size': b} for b in np.geomspace(
        2, 2**8, num=nvals_for_batch_size, endpoint=True, dtype=int
    ).tolist()]
    lr_inits = [{'learning_rate_init': l} for l in [1e-5, 1e-3, 1e-1]]
    non_opt_combs = []
    for h, ac, al, bs, lri in itertools.product(
            hls, acts, alphas, bsizes, lr_inits
    ):
        non_opt_combs.append({**h, **ac, **al, **bs, **lri})
    logger.debug('#HP non-opt: {}'.format(len(non_opt_combs)))
    # solver options
    # sgd_lr_opts = [{'learning_rate': 'constant'}]
    # sgd_lr_is_opts =  [{'learning_rate': 'invscaling', 'power_t': p}
    #                    for p in [0.5]
    # ]
    # sgd_lr_opts.extend(sgd_lr_is_opts)
    # sgd_mom_opts = [{'momentum': 0.9, 'nesterovs_momentum': n}
    #                 for n in [True, False]
    # ]
    # sgd_opts = [{**{'solver': 'sgd'}, **slo, **smo}
    #             for slo, smo in itertools.product(sgd_lr_opts, sgd_mom_opts)
    # ]
    # adam_eps = [{'epsilon': e} for e in [1e-8]]
    # adam_betas = [{'beta_1': b1, 'beta_2': b2}
    #               for b1, b2 in zip([0.9], [0.999])
    # ]
    # adam_opts = [{**{'solver': 'adam'}, **ae, **ab}
    #              for ae, ab in itertools.product(adam_eps, adam_betas)
    # ]
    #
    # solver_opts = sgd_opts + adam_opts
    solver_opts = [{'solver': 'adam'}]
    hp_list = []
    est_list = []
    for noo, oo in itertools.product(non_opt_combs, solver_opts):
        hp = {**noo, **oo}
        hp['max_iter'] = 1
        hp_list.append(hp)
        est_list.append(MLPC(**hp))
    logger.debug('#HP: {}'.format(len(est_list)))
    return zip(hp_list, est_list)


def eval_mlpc_on_dataset_kfold_cv(
        dset, nfolds, nvals_for_alpha, nvals_for_batch_size
):
    # get CV splitter
    skf = SKFold(n_splits=nfolds, shuffle=True, random_state=5489)
    accuracy = []
    f1macro = []
    f1micro = []
    warns = []
    for hp, est in get_mlpc_ests(
            nvals_for_alpha=nvals_for_alpha,
            nvals_for_batch_size=nvals_for_batch_size
    ):
        logging.debug('Processing MLPC(%s)' % str(hp))
        assert est.get_params()['max_iter'] == 1
        (acc, f1a, f1i), w = eval_est_kfold_cv(est, dset, skf)
        warns.extend(w)
        if acc is None: continue
        accuracy.append((acc, str(hp)))
        f1macro.append((f1a, str(hp)))
        f1micro.append((f1i, str(hp)))
    accuracy.sort(reverse=True)
    f1macro.sort(reverse=True)
    f1micro.sort(reverse=True)
    return (accuracy[0], f1macro[0], f1micro[0]), warns


def eval_openml_did(did, nfolds, nvals_for_alpha, nvals_for_batch_size):
    ret = None
    try:
        d = get_dataset(did)
        dname = d.name + '.' + str(did)
        X, y, c, a = d.get_data(
            target=d.default_target_attribute, dataset_format='array'
        )
        res, warns = eval_mlpc_on_dataset_kfold_cv(
            (normalize(X), y), nfolds, nvals_for_alpha, nvals_for_batch_size
        )
        return (dname, res, warns)
    except Exception as e:
        return (None, None, None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-t', '--n_parallel', help='Number of parallel workers',
        type=int, default=1
    )
    parser.add_argument(
        '-F', '--n_folds', help='Number of folds', type=int, default=10
    )
    parser.add_argument(
        '-n', '--min_data_dim', help='Minimum data dimensionality on OpenML',
        type=int,
    )
    parser.add_argument(
        '-x', '--max_data_dim', help='Maximum data dimensionality on OpenML',
        type=int,
    )
    parser.add_argument(
        '-S', '--max_data_samples',
        help='Maximum number of samples in data on OpenML', type=int,
    )
    parser.add_argument(
        '-A', '--num_vals_for_alpha',
        help='Number of values to try for the regularization parameter',
        type=int, default=5
    )
    parser.add_argument(
        '-B', '--num_vals_for_batch_size',
        help='Number of values to try for batch size',
        type=int, default=4
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
    res_dir = 'results/mlpc/' + timestamp
    print('Saving results/logs in \'%s\' ...' % res_dir)
    if not os.path.exists(res_dir) :
        os.makedirs(res_dir)
    warnings_file = res_dir + '/warnings.log'
    results_dict = {
        'dataset': [],
        'accuracy': [],
        'acc_hp': [],
        'f1macro': [],
        'f1a_hp': [],
        'f1micro': [],
        'f1i_hp': [],
    }
    X, y = load_digits(return_X_y=True)
    if (
            X.shape[1] >= args.min_data_dim
            and X.shape[1] <= args.max_data_dim
            and X.shape[0] <= args.max_data_samples
    ):
        print('Processing \'digits\' ...')
        res, warns = eval_mlpc_on_dataset_kfold_cv(
            (normalize(X), y), args.n_folds,
            args.num_vals_for_alpha, args.num_vals_for_batch_size
        )
        acc, f1a, f1i = res
        results_dict['dataset'].append('digits')
        results_dict['accuracy'].append(acc[0])
        results_dict['acc_hp'].append(acc[1])
        results_dict['f1macro'].append(f1a[0])
        results_dict['f1a_hp'].append(f1a[1])
        results_dict['f1micro'].append(f1i[0])
        results_dict['f1i_hp'].append(f1i[1])
        with open(warnings_file, 'w') as f :
            for i, w in enumerate(warns) :
                f.write(
                    '[digits %i] [%s]\t %s\n'
                    % (i + 1, w.category.__name__, w.message)
                )
    # Get OpenML dataset list
    val_dsets = get_openml_data_list(
        args.min_data_dim, args.max_data_dim, args.max_data_samples
    )
    dids = val_dsets['did'].values.astype(int).tolist()
    edfunc = lambda did : eval_openml_did(
        did, nfolds=args.n_folds,
        nvals_for_alpha=args.num_vals_for_alpha,
        nvals_for_batch_size=args.num_vals_for_batch_size,
    )
    print(
        'Processing {} sets with {} parallel workers'.format(
            len(dids), args.n_parallel)
    )
    results = Parallel(n_jobs=args.n_parallel)(
        delayed(edfunc)(did) for did in tqdm(
            dids, desc='Dataset', total=len(dids)
        )
    )
    assert len(results) == len(dids)
    for dname, res, warns in results :
        if dname is None :
            assert res is None
            continue
        acc, f1a, f1i = res
        results_dict['dataset'  ].append(dname)
        results_dict['accuracy'].append(acc[0])
        results_dict['acc_hp'].append(acc[1])
        results_dict['f1macro'].append(f1a[0])
        results_dict['f1a_hp'].append(f1a[1])
        results_dict['f1micro'].append(f1i[0])
        results_dict['f1i_hp'].append(f1i[1])
        with open(warnings_file, 'a') as f :
            for i, w in enumerate(warns) :
                f.write(
                    '[%s %i] [%s]\t %s\n'
                    % (dname, i + 1, w.category.__name__, w.message)
                )
    results_df = pd.DataFrame.from_dict(results_dict)
    print(results_df.head())
    fname = res_dir + '/mlpc.results.' + timestamp + '.csv'
    print('Saving results in %s' % fname)
    results_df.to_csv(fname, index=False)
