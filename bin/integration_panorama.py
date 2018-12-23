from process import load_names
from scanorama import *

from time import time
import numpy as np

NAMESPACE = 'panorama_integrated'

if __name__ == '__main__':
    from config import data_names

    datasets, genes_list, n_cells = load_names(data_names)
    datasets, genes = merge_datasets(datasets, genes_list)

    labels = []
    names = []
    curr_label = 0
    for i, a in enumerate(datasets):
        labels += list(np.zeros(a.shape[0]) + curr_label)
        names.append(data_names[i])
        curr_label += 1
    labels = np.array(labels, dtype=int)
    
    ########################
    ## Scanorama Assembly ##
    ########################

    datasets_dimred, genes = process_data(datasets, genes)
    
    # Put each of the datasets into a panorama.
    t0 = time()
    datasets_dimred = assemble(
        datasets_dimred, ds_names=data_names, sigma=150,
    )
    if VERBOSE:
        print('Integrated panoramas in {:.3f}s'.format(time() - t0))

    embedding = visualize(
        datasets_dimred, labels, NAMESPACE, data_names,
        multicore_tsne=False
    )

    np.savetxt('data/{}_embedding.txt'.format(NAMESPACE),
               embedding, delimiter='\t')
