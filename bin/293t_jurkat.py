import numpy as np
from scanorama import *
from sklearn.preprocessing import normalize, LabelEncoder
import sys

from process import load_names, merge_datasets, save_datasets

NAMESPACE = '293t_jurkat'

data_names = [
    'data/293t_jurkat/293t',
    'data/293t_jurkat/jurkat',
    'data/293t_jurkat/jurkat_293t_50_50',
]

if __name__ == '__main__':
    datasets, genes_list, n_cells = load_names(data_names)

    datasets_dimred, datasets, genes = correct(
        datasets, genes_list, ds_names=data_names,
        return_dimred=True
    )

    save_datasets(datasets, genes, data_names)

    labels = []
    names = []
    curr_label = 0
    for i, a in enumerate(datasets):
        labels += list(np.zeros(a.shape[0]) + curr_label)
        names.append(data_names[i])
        curr_label += 1
    labels = np.array(labels, dtype=int)
    
    embedding = visualize(datasets_dimred,
                          labels, NAMESPACE + '_ds', names,
                          perplexity=600, n_iter=400, size=100)
    
    cell_labels = (
        open('data/cell_labels/293t_jurkat_cluster.txt')
        .read().rstrip().split()
    )
    le = LabelEncoder().fit(cell_labels)
    labels = le.transform(cell_labels)
    cell_types = le.classes_
    
    visualize(None,
              labels, NAMESPACE + '_type', cell_types,
              perplexity=600, n_iter=400, size=100,
              embedding=embedding) 

    # Uncorrected.
    datasets, genes_list, n_cells = load_names(data_names)
    datasets, genes = merge_datasets(datasets, genes_list)
    datasets = [ normalize(ds, axis=1) for ds in datasets ]
    datasets_dimred = dimensionality_reduce(datasets)
    
    visualize(datasets_dimred, labels,
              NAMESPACE + '_type_uncorrected', cell_types,
              perplexity=600, n_iter=400, size=100)
