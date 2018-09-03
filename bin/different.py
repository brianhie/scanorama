from scanorama import assemble, connect, plot_clusters
from process import load_names, merge_datasets

import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
import sys
from time import time

NAMESPACE = 'different'

data_names = [
    'data/293t_jurkat/293t',
    'data/brain/neuron_9k',
    'data/hsc/hsc_mars',
    'data/macrophage/uninfected',
    'data/pancreas/pancreas_human',
    'data/pbmc/10x/68k_pbmc',
]

if __name__ == '__main__':
    datasets, genes_list, n_cells = load_names(data_names)
    datasets, genes = correct(datasets, genes_list)
    datasets = [ normalize(ds, axis=1) for ds in datasets ]
    datasets_dimred = dimensionality_reduce(datasets)

    labels = []
    names = []
    curr_label = 0
    for i, a in enumerate(datasets):
        labels += list(np.zeros(a.shape[0]) + curr_label)
        names.append(data_names[i])
        curr_label += 1
    labels = np.array(labels, dtype=int)
    
    visualize(datasets_dimred, labels, NAMESPACE, data_names)
