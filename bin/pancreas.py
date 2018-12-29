import numpy as np
from scanorama import *
from sklearn.metrics import silhouette_samples as sil
from scipy.sparse import vstack
from sklearn.preprocessing import normalize, LabelEncoder
import sys

from pancreas_tests import *
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
    
    labels = []
    names = []
    curr_label = 0
    for i, a in enumerate(datasets):
        labels += list(np.zeros(a.shape[0]) + curr_label)
        names.append(data_names[i])
        curr_label += 1
    labels = np.array(labels, dtype=int)

    datasets_dimred, datasets, genes = correct(
        datasets, genes_list, ds_names=data_names,
        return_dimred=True
    )
    
    cell_labels = (
        open('data/cell_labels/pancreas_cluster.txt')
        .read().rstrip().split()
    )
    le = LabelEncoder().fit(cell_labels)
    cell_labels = le.transform(cell_labels)
    cell_types = le.classes_
    
    print_oneway(vstack(datasets).toarray(), genes, labels)

    X_pan = vstack(datasets).toarray()
    var_pan = {
        genes[i]: val
        for i, val in enumerate(np.var(X_pan, axis=0))
    }

    pancreas_genes = [
        'HADH', 'G6PC2', 'PAPSS2', 'PCSK1', 'GC',
        'TTR', 'GCG', 'GPX3', 'VGF', 'CST3', 'KRT7',
        'ZFP36L1', 'KRT19',  'LAD1', 'FLNA', 'AHNAK',
        'ANXA2', 'RBP4', 'SST', 'FLT1', 'PLVAP', 'ENG',
        'S1PR1', 'EGFL7', 'CD93', 'ESM1', 'KDR', 'PPY',
        'BTG2', 'HERPUD1', 'GADD45A', 'LCN2', 'SDC4',
    ]

    embedding = visualize(datasets_dimred,
                          labels, NAMESPACE + '_ds', names,
                          gene_names=pancreas_genes, genes=genes,
                          gene_expr=vstack(datasets),
                          perplexity=100, n_iter=400)
    
    visualize(datasets_dimred,
              cell_labels, NAMESPACE + '_type', cell_types,
              embedding=embedding)

    # scran MNN.
    datasets, genes_list, n_cells = load_names(['data/mnn_corrected_pancreas'])
    
    datasets_dimred, datasets, genes = correct(
        datasets, genes_list, ds_names=data_names,
        return_dimred=True
    )
    genes = [ g.decode('utf-8') for g in genes ]
    
    print_oneway(vstack(datasets).toarray(), genes, labels)
    
    var_mnn = {
        genes[i]: val
        for i, val in enumerate(np.var(vstack(datasets).toarray(), axis=0))
    }
    
    # Uncorrected.
    datasets, genes_list, n_cells = load_names(data_names)
    datasets, genes = merge_datasets(datasets, genes_list)
    datasets = [ normalize(ds, axis=1) for ds in datasets ]
    datasets_dimred = dimensionality_reduce(datasets)

    print_oneway(vstack(datasets).toarray(), genes, labels)

    X_non = vstack(datasets).toarray()
    var_non = {
        genes[i]: val
        for i, val in enumerate(np.var(X_non, axis=0))
    }
    
    embedding = visualize(datasets_dimred, labels,
                          NAMESPACE + '_ds_uncorrected', names,
                          perplexity=100, n_iter=400)
    visualize(None, cell_labels,
              NAMESPACE + '_type_uncorrected', cell_types,
              embedding=embedding,
              perplexity=100, n_iter=400)

    from scipy.stats import pearsonr

    rhos = []
    assert(X_pan.shape == X_non.shape)
    for i in range(X_pan.shape[0]):
        rho, p = pearsonr(X_pan[i], X_non[i])
        rhos.append(rho)

    print(max(rhos))
    print(min(rhos))
    print(np.mean(rhos))
    print(np.median(rhos))
    print(np.var(rhos))
    
    in_common = sorted(set(var_pan.keys()) & set(var_non.keys()))
    print(pearsonr([ var_pan[g] for g in in_common ],
                    [ var_non[g] for g in in_common ]))
    in_common = sorted(set(var_mnn.keys()) & set(var_non.keys()))
    print(pearsonr([ var_mnn[g] for g in in_common ],
                    [ var_non[g] for g in in_common ]))

