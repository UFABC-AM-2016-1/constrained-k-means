import numpy as np

from sklearn.metrics.pairwise import pairwise_distances
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin

class ConstrainedKMeans(BaseEstimator, ClusterMixin, TransformerMixin):

    def __init__(self, n_clusters=8):
        self.n_clusters = n_clusters

    def fit(self, X, y=None, must_link=set(), cannot_link=set()):
        assert type(X).__name__ == 'ndarray', "X must be a ndarray"
        assert X.shape[0] >= self.n_clusters, "X can't have less objects than n_clusters"

        # Initialize cluster centers
        n_samples = X.shape[0]
        samples_idx = np.random.choice(range(n_samples), self.n_clusters, replace=False)
        centroids = map(lambda i: X[i], samples_idx)

        # ...

    # TODO: Implement _violate_constrains
    def _violate_constraints(instance, cluster, must_link, cannot_link):
        return False


if __name__ == '__main__':
    c = ConstrainedKMeans(n_clusters=3)
    X = np.array([
        [0],
        [1],
        [2],
        [3],
        [4],
        [5],
        [6],
        [7],
    ])
    c.fit(X)