import numpy as np
from sklearn.preprocessing import normalize
import sys

from process import load_names, merge_datasets
from scanorama import correct, visualize, process_data
from scanorama import dimensionality_reduce
from scanorama import interpret_alignments

NAMESPACE = 'macrophage'

data_names = [
    'data/macrophage/infected',
    'data/macrophage/uninfected',
]

if __name__ == '__main__':
    datasets_full, genes_list, n_cells = load_names(data_names)
    datasets, genes = merge_datasets(datasets_full, genes_list)
    datasets_dimred, genes = process_data(datasets, genes)
    
    interpret_alignments(
        datasets_dimred, datasets, genes, sigma=100, approx=True,
        n_permutations=100000
    )
