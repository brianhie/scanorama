import numpy as np
from scanorama import *
from scipy.sparse import vstack
from sklearn.preprocessing import normalize, LabelEncoder
import sys

from process import load_names

NAMESPACE = 'hsc'

data_names = [
    'data/hsc/hsc_mars',
    'data/hsc/hsc_ss2',
]

if __name__ == '__main__':
    datasets, genes_list, n_cells = load_names(data_names)

    datasets, genes = correct(datasets, genes_list)
    datasets_dimred = dimensionality_reduce(datasets, dimred=2)

    labels = []
    names = []
    curr_label = 0
    for i, a in enumerate(datasets):
        labels += list(np.zeros(a.shape[0]) + curr_label)
        names.append(data_names[i])
        curr_label += 1
    labels = np.array(labels, dtype=int)

    hsc_genes = [
        'GATA2', 'APOE', 'SPHK1', 'CTSE', 'FOS'
    ]

    # Visualize with PCA.
    visualize(None, labels, NAMESPACE + '_ds', names,
              gene_names=hsc_genes, genes=genes,
              gene_expr=vstack(datasets),
              embedding=np.concatenate(datasets_dimred),
              size=4)

    cell_labels = (
        open('data/cell_labels/hsc_cluster.txt')
        .read().rstrip().split()
    )
    le = LabelEncoder().fit(cell_labels)
    cell_labels = le.transform(cell_labels)
    cell_types = le.classes_

    visualize(None, cell_labels, NAMESPACE + '_type', cell_types,
              embedding=np.concatenate(datasets_dimred), size=4)

    # Uncorrected.
    datasets, genes_list, n_cells = load_names(data_names)
    datasets, genes = merge_datasets(datasets, genes_list)
    datasets = [ normalize(ds, axis=1) for ds in datasets ]
    datasets_dimred = dimensionality_reduce(datasets, dimred=2)
    
    visualize(None, labels, NAMESPACE + '_ds_uncorrected', names,
              embedding=np.concatenate(datasets_dimred), size=4)
    visualize(None, cell_labels, NAMESPACE + '_type_uncorrected', cell_types,
              embedding=np.concatenate(datasets_dimred), size=4)
