import numpy as np
from scanorama import *

from benchmark import write_table
from process import load_names, merge_datasets, process
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

def diff_expr(A, B, genes):
    from scipy.stats import mannwhitneyu
    from statsmodels.stats.multitest import multipletests

    p_vals = []
    for idx, gene in enumerate(genes):
        if sum(A[:, idx]) == 0 and sum(B[:, idx]) == 0:
            p_vals.append(1.)
            continue
        u, p = mannwhitneyu(A[:, idx], B[:, idx])
        p_vals.append(p)

    reject, q_vals, _, _ = multipletests(p_vals, method='bonferroni')
        
    for idx, gene in enumerate(genes):
        if reject[idx]:
            print('{}\t{}'.format(gene, q_vals[idx]))

    exit()

if __name__ == '__main__':
    datasets, genes_list, n_cells = load_names(data_names)

    monocytes, mono_genes = datasets[:4], genes_list[:4]
    monocytes, mono_genes = merge_datasets(monocytes, mono_genes)
    datasets = [ vstack(monocytes) ] + datasets[4:]
    genes_list = [ mono_genes ] + genes_list[4:]
    data_names = [ 'data/macrophage/monocytes_seqwell' ] + data_names[4:]

    datasets, genes = merge_datasets(datasets, genes_list)
    datasets_dimred, genes = process_data(datasets, genes)

    for i in range(len(datasets)):
        print('Writing {}'.format(data_names[i]))
        write_table(datasets[i].toarray(), genes, data_names[i])
    exit()

    #diff_expr(datasets[0].toarray(), datasets[-1].toarray(), genes)

    #_, A, _ = find_alignments_table(datasets_dimred)
    
    #time = np.array([ 0, 0, 3, 3, 6, 6 ]).reshape(-1, 1)
    #time_align_correlate(A, time)
    #
    #x = np.array([ 0, 0, 1, 1, 2, 2 ]).reshape(-1, 1)
    #y = [ -.1, .1, -.1, .1, -.1, .1 ]
    #time_align_visualize(A, x, y, namespace=NAMESPACE)

    from scipy.sparse import vstack
    
    X = vstack(datasets).toarray()
    write_table(X, genes, 'data/macrophage/' + NAMESPACE)

    datasets_dimred = assemble(
        datasets_dimred, # Assemble in low dimensional space.
        expr_datasets=datasets, # Modified in place.
    )

    X = vstack(datasets).toarray()
    write_table(X, genes, 'data/macrophage/' + NAMESPACE + '_corrected')
    
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

    with open('data/macrophage/mono_macro_hours.txt', 'w') as of:
        of.write('Days\tBatch\n')
        for idx, day in enumerate(days):
            of.write('mono_macro{}\t{}\t{}'
                     .format(idx, int(day), labels[idx].split('/')[-1]) + '\n')
