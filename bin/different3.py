import numpy as np
from scanorama import *
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
import sys
from time import time

from process import load_names

NAMESPACE = 'different3'

data_names = [
    'data/brain/neuron_9k',
    'data/hsc/hsc_mars',
    'data/macrophage/uninfected',
]

if __name__ == '__main__':
    datasets, genes_list, n_cells = load_names(data_names)

    datasets_dimred, datasets, genes = correct(
        datasets, genes_list, ds_names=data_names,
        return_dimred=True
    )

    labels = []
    names = []
    curr_label = 0
    for i, a in enumerate(datasets):
        labels += list(np.zeros(a.shape[0]) + curr_label)
        names.append(data_names[i])
        curr_label += 1
    labels = np.array(labels, dtype=int)
    
    visualize(datasets_dimred, labels, NAMESPACE, data_names,
              perplexity=100)
