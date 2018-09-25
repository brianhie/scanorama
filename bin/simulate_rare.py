import numpy as np
from scanorama import *
from scipy.sparse import vstack, csr_matrix
from sklearn.preprocessing import normalize, LabelEncoder
import sys

from benchmark import write_table
from process import load_names

NAMESPACE = 'simulate_rare'

data_names = [
    'data/simulation/simulate_rare/simulate_rare_0',
    'data/simulation/simulate_rare/simulate_rare_1',
    'data/simulation/simulate_rare/simulate_rare_2',
    'data/simulation/simulate_rare/simulate_rare_3',
    'data/simulation/simulate_rare/simulate_rare_4',
]

if __name__ == '__main__':
    datasets, genes_list, n_cells = load_names(data_names)
    n_genes = len(genes_list[0])
    
    #for i in range(len(datasets)):
    #    print('Writing {}'.format(data_names[i]))
    #    write_table(datasets[i].toarray(), genes_list[i], data_names[i])
    #exit()

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
                          perplexity=50, n_iter=400, size=10)


    # Uncorrected.
    datasets, genes = merge_datasets(datasets, genes_list)
    datasets = [ normalize(ds, axis=1) for ds in datasets ]
    datasets_dimred = dimensionality_reduce(datasets)
    
    embedding = visualize(datasets_dimred, labels,
                          NAMESPACE + '_ds_uncorrected', names,
                          perplexity=50, n_iter=400, size=10)
