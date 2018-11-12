from process import load_names, merge_datasets
from scanorama import *

from time import time
import numpy as np

NAMESPACE = 'panorama_union'

if __name__ == '__main__':
    from config import data_names

    datasets, genes_list, n_cells = load_names(data_names)

    t0 = time()
    datasets_dimred, genes = integrate(
        datasets, genes_list, ds_names=data_names,
        union=True, hvg=2000
    )
    if VERBOSE:
        print('Integrated panoramas in {:.3f}s'
              .format(time() - t0))

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
                          multicore_tsne=True)
