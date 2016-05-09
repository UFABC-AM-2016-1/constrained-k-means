import numpy as np
import json

from sklearn.datasets import load_digits, load_iris, load_diabetes

LINK_ARRAY_SIZE = 10

datasets =[
    ("iris", load_iris()),
    ("digits", load_digits()),
    ("diabetes", load_diabetes())
]

for name, data_set in datasets:
    samples = np.random.choice(len(data_set.data), LINK_ARRAY_SIZE)
    must_links = []
    cannot_links = []
    for sample in samples:
        value = data_set.target[sample]
        for selected in range(len(data_set.data)):
            if value == data_set.target[selected]:
                if sample == selected:
                    continue
                must_link = [
                    np.ndarray.tolist(data_set.data[sample]),
                    np.ndarray.tolist(data_set.data[selected])
                ]
                must_links.append(must_link)
                break
            else:
                continue

    for sample in samples:
        value = data_set.target[sample]
        for selected in range(len(data_set.data)):
            if value != data_set.target[selected]:
                must_link = [
                    np.ndarray.tolist(data_set.data[sample]),
                    np.ndarray.tolist(data_set.data[selected])
                ]
                must_links.append(must_link)
                break
            else:
                continue

    with open(name+".mustlink", "w") as outfile:
        json.dump(must_links, outfile, indent=4)

    with open(name+".cannotlink", "w") as outfile:
        json.dump(must_links, outfile, indent=4)
