from process import load_names, merge_datasets
from scanorama import *

from time import time
import numpy as np

NAMESPACE = 'panorama'

if __name__ == '__main__':
    from config import data_names

    datasets, genes_list, n_cells = load_names(data_names, verbose=0)
    _, genes = merge_datasets(datasets[:], genes_list)
    intersection = set(genes)

    hvg = 2000
    for i, ds in enumerate(datasets):
        _, genes_hvg = process_data([ ds ], genes_list[i], hvg=hvg)
        assert(len(genes_hvg) == hvg)
        print('{} has {} HVGs in intersection'.
              format(data_names[i], len(intersection & set(genes_hvg))))

