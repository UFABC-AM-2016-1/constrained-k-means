import numpy as np

from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin

from sklearn.datasets import load_digits, load_iris, load_diabetes
from sklearn import datasets


class ConstrainedKMeans(BaseEstimator, ClusterMixin, TransformerMixin):

    def __init__(self, n_clusters=8, n_iter=300):
        self.n_clusters = n_clusters
        self.n_iter = n_iter

    def fit(self, X, y=None, must_link=[], cannot_link=[]):
        assert type(X).__name__ == 'ndarray', "X must be a ndarray"
        assert X.shape[0] >= self.n_clusters, "X can't have less objects than n_clusters"

        # Initialize cluster centers
        n_samples = X.shape[0]
        samples_idx = np.random.choice(range(n_samples), self.n_clusters, replace=False)
        centroids = np.array(map(lambda i: X[i], samples_idx))
        clusters = [[] for _ in range(self.n_clusters)]

        labels = [0 for _ in X]
        inertia = 0

        for _ in range(self.n_iter):
            for i in range(self.n_clusters):
                clusters[i] = []
                inertia = 0

            for i, d in enumerate(X):
                rank = _rank_centroids(d, centroids)
                violate_constraint = True
                for idx in rank:
                    other_clusters = [clusters[j] for j in range(self.n_clusters) if j != idx]
                    cluster = clusters[idx]
                    violate_constraint = _violate_constraints(d, cluster, other_clusters, must_link, cannot_link)
                    if not violate_constraint:
                        clusters[idx].append(d)
                        labels[i] = idx

                        inertia += _dist(centroids[idx], d)
                        break
                if violate_constraint:
                    raise IOError("Unable to cluster")

            for idx in range(self.n_clusters):
                centroids[idx] = np.array(clusters[idx]).mean(axis=0)


        self.cluster_centers_ = centroids
        self.labels_ = labels
        self.inertia_ = inertia

        return self

    def predict(self, X):
        def predict_single(d):
            return _rank_centroids(d, self.cluster_centers_)[0]

        return map(predict_single, X)

    def transform(self, X):
        def transform_single(d):
            return np.array([ _dist(d, c) for c in self.cluster_centers_])

        return map(transform_single, X)


def _dist(a, b=0):
    d = a - b
    return np.sqrt(d.dot(d))


def _rank_centroids(instance, centroids):
    """
    Return a ascendant list of nearest clusters of certain instance
    :param instance: Instance to compare
    :param centroids: Centroids
    :return: the clusters array sorted by distance of instance
    """
    deltas = centroids - instance
    rank = np.asarray([_dist(d) for d in deltas]).argsort()

    return rank


def _violate_constraints(instance, cluster, other_clusters, must_link, cannot_link):
    def get_other_instance(instance, link):
        if (link[0] == instance).all():
            return link[1]
        elif (link[1] == instance).all():
            return link[0]

    for link in must_link:
        other_instance = get_other_instance(instance, link)
        if other_instance is not None:
            for c in other_clusters:
                if _contains(other_instance, c):
                    return True

    for link in cannot_link:
        other_instance = get_other_instance(instance, link)
        if other_instance is not None and _contains(other_instance, cluster):
            return True

    return False


def _contains(instance, _list):
    """Returns True if the ndarray list contains the specified instance"""
    if not _list:
        return False
    return bool([y for y in (instance == _list) if y.all()])
