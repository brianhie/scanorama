import numpy as np
from scanorama import *
from scipy.sparse import vstack
from sklearn.preprocessing import normalize, LabelEncoder
import sys

from process import load_names

NAMESPACE = 'pbmc'

data_names = [
    'data/pbmc/10x/68k_pbmc',
    'data/pbmc/10x/b_cells',
    'data/pbmc/10x/cd14_monocytes',
    'data/pbmc/10x/cd4_t_helper',
    'data/pbmc/10x/cd56_nk',
    'data/pbmc/10x/cytotoxic_t',
    'data/pbmc/10x/memory_t',
    'data/pbmc/10x/regulatory_t',
    'data/pbmc/pbmc_kang',
    'data/pbmc/pbmc_10X',
]

if __name__ == '__main__':
    datasets, genes_list, n_cells = load_names(data_names)

    datasets_dimred, datasets, genes = correct(
        datasets, genes_list, ds_names=data_names,
        return_dimred=True
    )
    
    labels = []
    names = []
    curr_label = 0
    for i, a in enumerate(datasets):
        labels += list(np.zeros(a.shape[0]) + curr_label)
        names.append(data_names[i])
        curr_label += 1
    labels = np.array(labels, dtype=int)

    pbmc_genes = [
        'CD14', 'PTPRC', 'FCGR3A', 'ITGAX', 'ITGAM', 'CD19', 'HLA-DRB1',
        'FCGR2B', 'FCGR2A', 'CD3E', 'CD4', 'CD8A','CD8B', 'CD28', 'CD8',
        'TBX21', 'IKAROS', 'IL2RA', 'CD44', 'SELL', 'CCR7', 'MS4A1',
        'CD68', 'CD163', 'IL5RA', 'SIGLEC8', 'KLRD1', 'NCR1', 'CD22',
        'IL3RA', 'CCR6', 'IL7R', 'CD27', 'FOXP3', 'PTCRA', 'ID3', 'PF4',
        'CCR10', 'SIGLEC7', 'NKG7', 'S100A8', 'CXCR3', 'CCR5', 'CCR3',
        'CCR4', 'PTGDR2', 'RORC'
    ]
    
    embedding = visualize(datasets_dimred,
                          labels, NAMESPACE + '_ds', names,
                          #gene_names=pbmc_genes,
                          #gene_expr=vstack(datasets),
                          genes=genes, perplexity=500, n_iter=400)

    cell_labels = (
        open('data/cell_labels/pbmc_cluster.txt')
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
