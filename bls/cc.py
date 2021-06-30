import logging
logger = logging.getLogger('CC')
from math import floor

import numpy as np
from sklearn.cluster import KMeans

class ClassClusters:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.classes_ = None
        self.n_classes_ = None
        self.nclusters_per_class = kwargs['nclusters_per_class']
        self.kmcs = None

    def set_params(self, **kwargs):
        for k in kwargs:
            self.kwargs[k] = kwargs[k]
        self.nclusters_per_class = (
            kwargs['nclusters_per_class'] if 'nclusters_per_class' in kwargs
            else 1
        )
        self.classes_ = None
        self.n_classes_ = None
        self.kmcs = None

    def fit(self, X, y):
        # Get number of classes
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        l2i = {l: i for i, l in enumerate(self.classes_)}
        # initialize k-means per class
        self.kmcs = [
            KMeans(
                n_clusters=self.nclusters_per_class,
                n_init=5, max_iter=30, tol=0.001
            )
            for i in range(self.n_classes_)
        ]
        # run k-means per class
        for i, l in enumerate(self.classes_):
            logger.debug('Label %i: %s' % (l, str(X[y == l, :].shape)))
            self.kmcs[i].fit(X[y == l, :])
        return self

    def partial_fit(X, y):
        pass

    def _km_scores(self, X):
        fX = np.transpose(
            np.array([ np.min(km.transform(X), axis=1) for km in self.kmcs ])
        )
        assert fX.shape[0] == X.shape[0]
        assert fX.shape[1] == self.n_classes_
        return fX

    def predict(self, X):
        fX = self._km_scores(X)
        # Return the class with minimum value
        # Break ties randomly, currently it choses minimum index with min value
        min_km_scores = np.min(fX, axis=1)
        logger.debug('min scores: %s' % str(min_km_scores.shape))
        y = []
        nties = 0
        for min_km_score, fx in zip(min_km_scores, fX):
            y_set = self.classes_[fx == min_km_score]
            l = None
            if len(y_set) > 1:
                nties += 1
                l = y_set[np.random.randint(0, len(y_set))]
            else:
                l = y_set[0]
            y.append(l)
        logger.debug('%i / %i points had ties' % (nties, X.shape[0]))
        return np.array(y)

    def predict_proba(self, X):
        fX = self._km_scores(X)
        exp_neg_fX = np.exp(-fX)
        probs = exp_neg_fX / np.sum(exp_neg_fX, axis=1)[:, None]
        return probs

    def get_params(self, deep=False):
        return self.kwargs


if __name__ == '__main__':
    logger.setLevel(logging.DEBUG)
    from sklearn.datasets import load_iris
    X, y = load_iris(return_X_y=True)
    batch = 4
    kwargs = {'nclusters_per_class': 2}
    cc = ClassClusters(**kwargs)
    cc.fit(X, y)
    print('Cluster center:', cc.kmcs[0].cluster_centers_)
    print('Preds:', cc.predict(X[:batch, :]))
    print('Probs:', cc.predict_proba(X[:batch, :]))
