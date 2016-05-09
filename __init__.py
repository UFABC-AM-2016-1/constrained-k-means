import numpy as np

from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin

from sklearn.datasets import load_digits, load_iris, load_diabetes

class Cluster:
        def __init__(self, position, _id):
            self.instances = []
            self.position = position
            self._id = _id

        def recalculate(self):
            self.position = np.mean(self.instances, axis=0)

        def add(self, instance):
            self.instances.append(instance)

        def clear(self):
            self.instances = []

        def __str__(self):
            return str({"id": self._id, "position": self.position})

class ConstrainedKMeans(BaseEstimator, ClusterMixin, TransformerMixin):

    def debug(self, msg):
        if self._debug:
            print "["+self.__class__.__name__+"]", msg

    def __init__(self, n_clusters=8, debug=False):
        self._debug = debug
        self.n_clusters = n_clusters

    def fit(self, X, y=None, must_link=[], cannot_link=[]):
        assert type(X).__name__ == 'ndarray', "X must be a ndarray"
        assert X.shape[0] >= self.n_clusters, "X can't have less objects than n_clusters"

        # Initialize cluster centers
        n_samples = X.shape[0]
        samples_idx = np.random.choice(range(n_samples), self.n_clusters, replace=False)
        centroids = map(lambda i: X[i], samples_idx)
        clusters = []
        _id = 0
        for centroid in centroids:
            cluster = Cluster(centroid,_id)
            clusters.append(cluster)
            _id += 1

        self.debug("Initial Centroids: ")
        for cluster in clusters:
            self.debug(cluster)

        for _ in range(100):
            for cluster in clusters:
                cluster.clear()
            for d in X:
                ranked_clusters = _rank_clusters(d, clusters)
                for cluster in ranked_clusters:
                    violate_constraint = _violate_constraints(d, cluster, must_link, cannot_link)
                    if violate_constraint == False:
                        cluster.add(d)
                        break
                if violate_constraint:
                    raise IOError("Unable to cluster")

            for cluster in clusters:
                cluster.recalculate()

        self.debug(50*"-")
        for cluster in clusters:
            self.debug("cluster: "+str(cluster))
            self.debug("cluster size: "+str(len(cluster.instances)))
            self.debug(50*"-")
        # ...


def _rank_clusters(instance, clusters):
    """
    Return a ascendant list of nearest clusters of certain instance
    :param instance: Instance to compare
    :param clusters: Clusters
    :return: the clusters array sorted by distance of instance
    """
    clusters_ranks = np.asarray([np.linalg.norm(p.position - instance) for p in clusters]).argsort()
    sorted_clusters = np.asarray(clusters)[clusters_ranks]

    return sorted_clusters

def _violate_constraints(instance, cluster, must_link, cannot_link):
    if len(cluster.instances) == 0:
        return False

    for link in filter(lambda l: _contains(instance, l), must_link):
        other_instance = link[1] if (link[0] - instance).all() else link[0]
        if not _contains(other_instance, cluster.instances):
            return True

    for link in filter(lambda l: _contains(instance, l), cannot_link):
        other_instance = link[1] if (link[0] - instance).all() else link[0]
        if _contains(other_instance, cluster.instances):
            return True

    return False


def _contains(instance, _list):
    """Returns True if the ndarray list contains the specified instance"""
    return bool([y for y in (instance == _list) if y.all()])


def generate_must_cannot_links(dataset, size=2):
    must_link = []
    cannot_link = []

    samples = np.random.choice(len(dataset.data), size)

    for sample in samples:
        value = dataset.target[sample]




if __name__ == '__main__':

    digits = load_digits()

    datasets = [
        ("iris", load_iris()),
        ("diabetes", load_diabetes()),
        ("digits", load_digits())
        ]

    generate_must_cannot_links(digits)

    #c1 = ConstrainedKMeans(n_clusters=10, debug=True)
    #c1.fit(digits.data)
    #c2 = KMeans(n_clusters=10)
    #c2.fit(digits.data)