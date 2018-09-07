import numpy as np
from scanorama import correct, visualize, process_data
from scanorama import dimensionality_reduce, merge_datasets
from scipy.sparse import vstack
from sklearn.preprocessing import normalize, LabelEncoder
import sys

from process import load_names

NAMESPACE = 'pancreas'

data_names = [
    'data/pancreas/pancreas_inDrop',
    'data/pancreas/pancreas_multi_celseq2_expression_matrix',
    'data/pancreas/pancreas_multi_celseq_expression_matrix',
    'data/pancreas/pancreas_multi_fluidigmc1_expression_matrix',
    'data/pancreas/pancreas_multi_smartseq2_expression_matrix',
]

if __name__ == '__main__':
    datasets, genes_list, n_cells = load_names(data_names)
    datasets, genes = correct(datasets, genes_list)
    datasets = [ normalize(ds, axis=1) for ds in datasets ]
    datasets_dimred = dimensionality_reduce(datasets)

    labels = []
    names = []
    curr_label = 0
    for i, a in enumerate(datasets):
        labels += list(np.zeros(a.shape[0]) + curr_label)
        names.append(data_names[i])
        curr_label += 1
    labels = np.array(labels, dtype=int)

    pancreas_genes = [
        'HADH', 'G6PC2', 'PAPSS2', 'PCSK1', 'NKX6.1', 'GC',
        'TTR', 'GCG', 'GPX3', 'VGF', 'CST3', 'KRT7', 'LCN2', 'SDC4',
        'ZFP36L1', 'KRT19', 'CDC42EP1', 'LAD1', 'FLNA', 'AHNAK',
        'ANXA2', 'RBP4', 'SST', 'PECAM1', 'FLT1', 'PLVAP', 'ENG',
        'S1PR1', 'EGFL7', 'ADGRL4', 'CD93', 'ESM1', 'KDR', 'PPY',
        'BTG2', 'HERPUD1', 'GADD45A'
    ]

    embedding = visualize(datasets_dimred,
                          labels, NAMESPACE + '_ds', names,
                          gene_names=pancreas_genes, genes=genes,
                          gene_expr=vstack(datasets),
                          perplexity=100, n_iter=400)
    cell_labels = (
        open('data/cell_labels/pancreas_cluster.txt')
        .read().rstrip().split()
    )
    le = LabelEncoder().fit(cell_labels)
    cell_labels = le.transform(cell_labels)
    cell_types = le.classes_

    visualize(datasets_dimred,
              cell_labels, NAMESPACE + '_type', cell_types,
              embedding=embedding)
    
    # Uncorrected.
    datasets, genes_list, n_cells = load_names(data_names)
    datasets, genes = merge_datasets(datasets, genes_list)
    datasets = [ normalize(ds, axis=1) for ds in datasets ]
    datasets_dimred = dimensionality_reduce(datasets)
    
    embedding = visualize(datasets_dimred, labels,
                          NAMESPACE + '_ds_uncorrected', names,
                          perplexity=100, n_iter=400)
    visualize(None, cell_labels,
              NAMESPACE + '_type_uncorrected', cell_types,
              embedding=embedding,
              perplexity=100, n_iter=400)
