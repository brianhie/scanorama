import numpy as np

from process import load_names, merge_datasets
from scanorama import process_data, find_alignments_table
from time_align import time_align_correlate, time_align_visualize, time_dist

NAMESPACE = 'fly_brain'

data_names = [
    'data/fly_brain/DGRP-551_0d_rep1',
    'data/fly_brain/DGRP-551_0d_rep2',
    'data/fly_brain/DGRP-551_1d_rep1',
    'data/fly_brain/DGRP-551_3d_rep1',
    'data/fly_brain/DGRP-551_6d_rep1',
    'data/fly_brain/DGRP-551_6d_rep2',
    'data/fly_brain/DGRP-551_9d_rep1',
    'data/fly_brain/DGRP-551_15d_rep1',
    'data/fly_brain/DGRP-551_30d_rep1',
    'data/fly_brain/DGRP-551_30d_rep2',
    'data/fly_brain/DGRP-551_50d_rep1',
]

if __name__ == '__main__':
    datasets, genes_list, n_cells = load_names(data_names)
    datasets, genes = merge_datasets(datasets, genes_list)
    datasets_dimred, genes = process_data(datasets, genes)

    _, A, _ = find_alignments_table(datasets_dimred)
    
    time = np.array([ 0, 0, 1, 3, 6, 6, 9, 15, 30, 30, 50 ]).reshape(-1, 1)
    time_align_correlate(A, time)
    
    time_dist(datasets_dimred, time)
    
    x = np.array([ 0, 0, 1, 2, 3, 3, 4, 5, 6, 6, 7 ]).reshape(-1, 1)
    y = [ -.1, .1, 0, 0, -.1, .1, 0, 0, -.1, .1, 0 ]
    time_align_visualize(A, x, y, namespace=NAMESPACE)

