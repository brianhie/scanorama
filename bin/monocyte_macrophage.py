import numpy as np
from scanorama import *
from scipy.sparse import vstack

from benchmark import write_table
from process import load_names, merge_datasets, process
from time_align import time_align_correlate, time_align_visualize

NAMESPACE = 'mono_macro'

data_names = [
    'data/macrophage/monocytes',
    'data/pbmc/10x/cd14_monocytes',
    'data/macrophage/mcsf_day3_1',
    'data/macrophage/mcsf_day3_2',
    'data/macrophage/mcsf_day6_1',
    'data/macrophage/mcsf_day6_2',
]

if __name__ == '__main__':
    datasets, genes_list, n_cells = load_names(data_names, norm=False)
    
    datasets, genes = merge_datasets(datasets, genes_list)
    datasets_dimred, genes = process_data(datasets, genes)

    _, A, _ = find_alignments_table(datasets_dimred)
    
    time = np.array([ 0, 0, 3, 3, 6, 6 ]).reshape(-1, 1)
    time_align_correlate(A, time)
    
    x = np.array([ 0, 0, 1, 1, 2, 2 ]).reshape(-1, 1)
    y = [ -.1, .1, -.1, .1, -.1, .1 ]
    time_align_visualize(A, x, y, namespace=NAMESPACE)
    
    X = vstack(datasets).toarray()
    write_table(X, genes, 'data/macrophage/' + NAMESPACE)
    
    labels = []
    days = []
    curr_label = 0
    curr_day = 0
    for i, a in enumerate(datasets):
        labels += [ data_names[i] ] * int(a.shape[0])
        curr_label += 1
        if data_names[i] == 'data/macrophage/mcsf_day3_1':
            curr_day += 3
        if data_names[i] == 'data/macrophage/mcsf_day6_1':
            curr_day += 3            
        days += list(np.zeros(a.shape[0]) + curr_day)
    
    with open('data/macrophage/mono_macro_meta.txt', 'w') as of:
        of.write('Days\tBatch\n')
        for idx, day in enumerate(days):
            of.write('mono_macro{}\t{}\t{}'
                     .format(idx, int(day), labels[idx].split('/')[-1]) + '\n')
    
    with open('data/macrophage/' + NAMESPACE + '_genes.txt', 'w') as f:
        f.write('gene_short_name\n')
        for gene in genes:
            f.write('{0}\t{0}\n'.format(gene))
    
    assemble(datasets_dimred, expr_datasets=datasets)
    X = vstack(datasets).toarray()
    write_table(X, genes, 'data/macrophage/' + NAMESPACE + '_corrected')
