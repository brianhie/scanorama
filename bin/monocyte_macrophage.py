import numpy as np

from process import load_names, merge_datasets
from scanorama import process_data, find_alignments_table
from time_align import time_align_correlate, time_align_visualize

NAMESPACE = 'mono_macro'

data_names = [
    'data/macrophage/monocytes_1',
    'data/macrophage/monocytes_2',
    'data/macrophage/monocytes_3',
    'data/macrophage/monocytes_4',
    'data/pbmc/10x/cd14_monocytes',
    'data/macrophage/mcsf_day3_1',
    'data/macrophage/mcsf_day3_2',
    'data/macrophage/mcsf_day6_1',
    'data/macrophage/mcsf_day6_2',
]

if __name__ == '__main__':
    datasets, genes_list, n_cells = load_names(data_names)

    monocytes, mono_genes = datasets[:4], genes_list[:4]
    monocytes, mono_genes = merge_datasets(monocytes, mono_genes)
    datasets = [ monocytes[0] ] + datasets[4:]
    genes_list = [ mono_genes ] + genes_list[4:]
    data_names = [ 'data/macrophage/monocytes_seqwell' ] + data_names[4:]
    
    datasets, genes = merge_datasets(datasets, genes_list)
    datasets_dimred, genes = process_data(datasets, genes)

    _, A = find_alignments_table(datasets_dimred)
    
    time = np.array([ 0, 0, 3, 3, 6, 6 ]).reshape(-1, 1)
    time_align_correlate(A, time)

    x = np.array([ 0, 0, 1, 1, 2, 2 ]).reshape(-1, 1)
    y = [ -.1, .1, -.1, .1, -.1, .1 ]
    time_align_visualize(A, x, y, namespace=NAMESPACE)
