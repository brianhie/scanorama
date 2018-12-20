import numpy as np

from process import load_names, merge_datasets
from scanorama import process_data, find_alignments_table
from time_align import time_align_correlate, time_align_visualize, time_dist

NAMESPACE = 'dendritic'

data_names = [
    'data/dendritic/unstimulated',
    'data/dendritic/unstimulated_repl',
    'data/dendritic/lps_1h',
    'data/dendritic/lps_2h',
    'data/dendritic/lps_4h',
    'data/dendritic/lps_4h_repl',
    'data/dendritic/lps_6h',
]

if __name__ == '__main__':
    datasets, genes_list, n_cells = load_names(data_names)
    datasets, genes = merge_datasets(datasets, genes_list)
    datasets_dimred, genes = process_data(datasets, genes)

    _, A, _ = find_alignments_table(datasets_dimred)
    
    time = np.array([ 0, 0, 1, 2, 4, 4, 6 ]).reshape(-1, 1)
    
    time_align_correlate(A, time)
    
    time_dist(datasets_dimred, time)

    x = np.array([ 0, 0, 1, 2, 3, 3, 4 ]).reshape(-1, 1)
    y = [ -.1, .1, 0, 0, -.1, .1, 0 ]
    time_align_visualize(A, x, y, namespace=NAMESPACE)

