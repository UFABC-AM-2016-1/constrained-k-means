print(__doc__)

# Author: Phil Roth <mr.phil.roth@gmail.com>
# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
	
from sklearn.datasets import load_digits, load_iris, load_diabetes
from sklearn.metrics import silhouette_samples, silhouette_score, adjusted_rand_score
from sklearn import datasets
import __init__ as ck

random_state = 170

iris = datasets.load_iris()
X=iris.data
y=iris.target


y_pred = KMeans(n_clusters=3, random_state=random_state).fit_predict(X)
rand_avg = adjusted_rand_score( y  , y_pred )
print rand_avg


iris = datasets.load_iris()

clf = ck.ConstrainedKMeans(n_clusters=3, debug=False)

links = {
    'must_link': [
        [
            iris.data[0],
            iris.data[50]
        ]
    ],
    'cannot_link': [
        [
            iris.data[0],
            iris.data[2]
        ]
    ]
}

# np.save('/tmp/123', links)
# links = np.load('/tmp/123.npy').item()

clf.fit(iris.data, iris.target, **links)
#print clf.labels_

#print clf.labels_[0], clf.labels_[50]

rand_avg = adjusted_rand_score( y  , clf.labels_ )
print rand_avg

quit()
