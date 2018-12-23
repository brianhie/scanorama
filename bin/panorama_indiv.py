from config import data_names

from scanorama import *
from sklearn.metrics import silhouette_samples as sil
from process import load_names, merge_datasets

from time import time
import numpy as np

NAMESPACE = 'panorama1'

# Find the panoramas in the data.
def panorama(datasets_full, genes_list):
    if VERBOSE:
        print('Processing and reducing dimensionality...')
    datasets, genes = merge_datasets(datasets_full, genes_list)
    datasets_dimred, genes = process_data(datasets, genes)

    if VERBOSE:
        print('Finding panoramas...')
    panoramas = connect(datasets_dimred)

    if VERBOSE:
        print(panoramas)

    return panoramas

if __name__ == '__main__':
    # Load raw data from files.
    datasets, genes_list, n_cells = load_names(data_names)

    ##########################
    ## Panorama correction. ##
    ##########################

    # Put each of the datasets into a panorama.
    t0 = time()
    panoramas = panorama(datasets[:], genes_list)
    if VERBOSE:
        print('Found panoramas in {:.3f}s'.format(time() - t0))

    # Assemble and correct each panorama individually.
    t0 = time()
    for p_idx, pano in enumerate(panoramas):
        if VERBOSE:
            print('Building panorama {}'.format(p_idx))

        # Consider just the datasets in the panorama.
        pan_datasets = [ datasets[p] for p in pano ]
        pan_genes_list = [ genes_list[p] for p in pano ]

        # Do batch correction on those datasets.
        pan_datasets, genes = correct(pan_datasets, pan_genes_list, sigma=150)
        if VERBOSE:
            print('Found {} genes in panorama'.format(len(genes)))

        # Store batch corrected result.
        for i, p in enumerate(pano):
            datasets[p] = pan_datasets[i]
            genes_list[p] = genes

    if VERBOSE:
        print('Batch corrected panoramas in {:.3f}s'.format(time() - t0))

    ##########################
    ## Final visualization. ##
    ##########################

    # Put all the corrected data together for visualization purposes.
    datasets, genes = merge_datasets(datasets, genes_list)
    datasets_dimred, genes = process_data(datasets, genes)

    # Label each cell with its dataset of origin.
    labels = np.zeros(n_cells, dtype=int)
    base = 0
    for i, ds in enumerate(datasets):
        labels[base:(base + ds.shape[0])] = i
        base += ds.shape[0]

    embedding = visualize(
        datasets_dimred, labels, NAMESPACE, data_names,
        n_iter=450, multicore_tsne=False
    )
    np.savetxt('data/{}_embedding.txt'.format(NAMESPACE),
               embedding, delimiter='\t')
