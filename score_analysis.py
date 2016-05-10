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
import generate_constraints_link as generate_constraints_link

random_state = 170

#print generate_constraints_link.datasets

for dataset_label, dataset in generate_constraints_link.datasets:
	print dataset_label
	X=dataset.data
	y=dataset.target

	clusters=[5,10,15,20,25,30,35,40,45]
	serie_score=[]
	for n_cluster in clusters:
		y_pred = KMeans(n_clusters=n_cluster, random_state=random_state).fit_predict(X)
		rand_avg = adjusted_rand_score( y  , y_pred )
		print "KMeans Classic:",rand_avg
		serie_score.append(rand_avg)
	plt.plot(clusters, serie_score, label='KMeans', linewidth=2.0)

		
	
	for link_size in [5,10,15,20]:
		serie_score=[]
		generate_constraints_link.generate(link_array_size=link_size)
		links = np.load(dataset_label+'.npy').item()
		for n_cluster in clusters:
			clf = ck.ConstrainedKMeans(n_clusters=n_cluster)

			clf.fit(X, y, **links)

			rand_avg = adjusted_rand_score( y  , clf.labels_ )
			serie_score.append(rand_avg)
			print "Link Size ",link_size,": ",rand_avg
		plt.plot(clusters, serie_score, label="LinkSize "+str(link_size))
	plt.xlabel("Clusters")
	plt.ylabel("Rand Ajustado")
	plt.legend(loc="upper right")		
	plt.show()



quit()
