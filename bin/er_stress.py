import numpy as np
from scipy.sparse import vstack
from scipy.stats import ttest_ind
from sklearn.preprocessing import normalize, LabelEncoder
import sys

from process import load_names
from scanorama import correct, visualize, process_data, plt
from scanorama import dimensionality_reduce, merge_datasets

NAMESPACE = 'er_stress'

data_names = [
    'data/pancreas/pancreas_inDrop',
    'data/pancreas/pancreas_multi_celseq2_expression_matrix',
    'data/pancreas/pancreas_multi_celseq_expression_matrix',
    'data/pancreas/pancreas_multi_fluidigmc1_expression_matrix',
    'data/pancreas/pancreas_multi_smartseq2_expression_matrix',
]

if __name__ == '__main__':
    datasets, genes_list, n_cells = load_names(data_names)
    #datasets, genes = merge_datasets(datasets, genes_list)
    #datasets_dimred, genes = process_data(datasets, genes, hvg=hvg)
    datasets, genes = correct(datasets, genes_list)
    X = vstack(datasets).toarray()
    X[X < 0] = 0

    cell_labels = (
        open('data/cell_labels/pancreas_cluster.txt')
        .read().rstrip().split()
    )
    er_idx = [ i for i, cl in enumerate(cell_labels)
               if cl == 'beta_er' ]
    beta_idx = [ i for i, cl in enumerate(cell_labels)
                 if cl == 'beta' ]

    gadd_idx = list(genes).index('GADD45A')
    herp_idx = list(genes).index('HERPUD1')

    plt.figure()
    plt.boxplot([ X[er_idx, gadd_idx], X[beta_idx, gadd_idx] ],
                showmeans=True)
    plt.title('GADD45A (p = {})'.format(ttest_ind(
        X[er_idx, gadd_idx], X[beta_idx, gadd_idx], equal_var=False
    )[1]))
    plt.xticks([1, 2], ['beta_er', 'beta'])
    plt.ylabel('Scaled gene expression')
    plt.savefig('er_stress_GADD45A.svg')

    plt.figure()
    plt.boxplot([ X[er_idx, herp_idx], X[beta_idx, herp_idx] ],
                showmeans=True)
    plt.title('HERPUD1 (p = {})'.format(ttest_ind(
        X[er_idx, herp_idx], X[beta_idx, herp_idx], equal_var=False
    )[1]))
    plt.xticks([1, 2], ['beta_er', 'beta'])
    plt.ylabel('Scaled gene expression')
    plt.savefig('er_stress_HERPUD1.svg')
