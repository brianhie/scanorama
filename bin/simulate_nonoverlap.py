import numpy as np
from scanorama import *
from scipy.sparse import vstack, csr_matrix
from sklearn.preprocessing import normalize, LabelEncoder
import sys

from benchmark import write_table
from process import load_names

NAMESPACE = 'simulate_nonoverlap'

data_names = [
    'data/simulation/simulate_nonoverlap/simulate_nonoverlap_A',
    'data/simulation/simulate_nonoverlap/simulate_nonoverlap_B',
    'data/simulation/simulate_nonoverlap/simulate_nonoverlap_C',
]

if __name__ == '__main__':
    datasets, genes_list, n_cells = load_names(data_names)
    n_genes = len(genes_list[0])
    datasets = [ csr_matrix(ds + (np.absolute(np.random.randn(1, n_genes)) + 10))
                 for ds in datasets ]
    
    #for i in range(len(datasets)):
    #    print('Writing {}'.format(data_names[i]))
    #    write_table(datasets[i].toarray(), genes_list[i], data_names[i])

    datasets_dimred, _, genes = correct(
        datasets[:], genes_list, ds_names=data_names,
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

    embedding = visualize(datasets_dimred,
                          labels, NAMESPACE + '_ds', names,
                          perplexity=100, n_iter=400, size=10)

    cell_labels = (
        open('data/cell_labels/{}_cluster.txt'.format(NAMESPACE))
        .read().rstrip().split()
    )
    le = LabelEncoder().fit(cell_labels)
    cell_labels = le.transform(cell_labels)
    cell_types = le.classes_

    visualize(datasets_dimred,
              cell_labels, NAMESPACE + '_type', cell_types,
              embedding=embedding, size=10)
    
    # Uncorrected.
    datasets, genes = merge_datasets(datasets, genes_list)
    datasets = [ normalize(ds, axis=1) for ds in datasets ]
    datasets_dimred = dimensionality_reduce(datasets)
    
    embedding = visualize(datasets_dimred, labels,
                          NAMESPACE + '_ds_uncorrected', names,
                          perplexity=100, n_iter=400, size=10)

    visualize(datasets_dimred,
              cell_labels, NAMESPACE + '_type_uncorrected', cell_types,
              perplexity=100, n_iter=400, embedding=embedding, size=10)
