from utils import SCRNA, plt
from scanorama import assemble, connect, plot_clusters
from process import load_names, merge_datasets
from t_sne_approx import TSNEApprox

import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
import sys
from time import time

NAMESPACE = 'different'
DIMRED = 100
HVG = 10000

N_ITER = 500
PERPLEXITY = 1200
SIGMA = 150
KNN = 20
APPROX = True
VERBOSE = 2

data_names = [
    'data/293t_jurkat/293t',
    'data/brain/neuron_9k',
    'data/hsc/hsc_mars',
    'data/macrophage/uninfected',
    'data/pancreas/pancreas_human',
    'data/pbmc/10x/68k_pbmc',
]
s = SCRNA('', '', '')

# Find the panoramas in the data.
def panorama(datasets_full, genes_list):
    datasets, genes = merge_datasets(datasets_full, genes_list)
    datasets_dimred, genes = process_data(datasets, genes)

    # Connected components form panoramas.
    panoramas = connect(datasets_dimred, knn=KNN,
                        approx=APPROX, verbose=VERBOSE)
    if VERBOSE:
        print(panoramas)

    return panoramas

# Do batch correction on the data.
def correct(datasets_full, genes_list, hvg=HVG):
    datasets, genes = merge_datasets(datasets_full, genes_list)
    datasets_dimred, genes = process_data(datasets, genes, hvg=hvg)
    
    datasets_dimred = assemble(
        datasets_dimred, # Assemble in low dimensional space.
        expr_datasets=datasets, # Modified in place.
        verbose=VERBOSE, knn=KNN, sigma=SIGMA, approx=APPROX
    )

    return datasets, genes

# Normalize and reduce dimensionality.
def process_data(datasets, genes, hvg=HVG, dimred=DIMRED):
    # Only keep highly variable genes
    if hvg > 0:
        X = np.concatenate(datasets)
        disp = s.dispersion(X)
        top_genes = set(genes[
            list(reversed(np.argsort(disp)))[:HVG]
        ])
        for i in range(len(datasets)):
            gene_idx = [ idx for idx, g_i in enumerate(genes)
                         if g_i in top_genes ]
            datasets[i] = datasets[i][:, gene_idx]
        genes = np.array(sorted(top_genes))

    # Normalize.
    datasets = [ normalize(ds, axis=1) for ds in datasets ]
    
    # Compute compressed embedding.
    if dimred > 0:
        datasets_dimred = dimensionality_reduce(datasets)
        return datasets_dimred, genes

def dimensionality_reduce(datasets, dimred=DIMRED):
    s.dim_red_k = dimred
    X = np.concatenate(datasets)
    X = s.reduce_dimensionality(X)
    datasets_dimred = []
    base = 0
    for ds in datasets:
        datasets_dimred.append(X[base:(base + ds.shape[0]), :])
        base += ds.shape[0]
    return datasets_dimred

# Plot t-SNE visualization.
def visualize(assembled, labels, namespace, data_names,
              gene_names=None, gene_expr=None, genes=None,
              n_iter=N_ITER, perplexity=PERPLEXITY, verbose=VERBOSE,
              learn_rate=200., early_exag=12., embedding=None, size=1):
    # Fit t-SNE.
    if embedding is None:
        tsne = TSNEApprox(n_iter=n_iter, perplexity=perplexity,
                          verbose=verbose, random_state=69,
                          learning_rate=learn_rate,
                          early_exaggeration=early_exag)
        tsne.fit(np.concatenate(assembled))
        embedding = tsne.embedding_

    # Plot clusters together.
    plot_clusters(embedding, labels, s=size)
    plt.title(('Panorama ({} iter, perplexity: {}, sigma: {}, ' +
               'knn: {}, hvg: {}, dimred: {}, approx: {})')
              .format(n_iter, perplexity, SIGMA, KNN, HVG,
                      DIMRED, APPROX))
    s.viz_prefix = namespace
    plt.savefig(s.viz_prefix + '.svg', dpi=500)

    # Plot clusters individually.
    for i in range(len(data_names)):
        s.visualize_cluster(embedding, i, labels,
                            cluster_name=data_names[i], size=size)

    # Plot gene expression levels.
    if (not gene_names is None) and \
       (not gene_expr is None) and \
       (not genes is None):
        for gene_name in gene_names:
            s.visualize_expr(gene_expr, embedding,
                             genes, gene_name, size=size)
    
    return embedding

if __name__ == '__main__':
    # Load raw data from files.
    datasets, genes_list, n_cells = load_names(data_names)
    
    ##########################
    ## Panorama correction. ##
    ##########################

    # Put each of the datasets into a panorama.
    t0 = time()
    panoramas = panorama(datasets, genes_list)
    if VERBOSE:
        print('Found panoramas in {:.3f}s'.format(time() - t0))

    # Assemble and correct each panorama individually.
    t0 = time()
    for p_idx, pano in enumerate(panoramas):
        if VERBOSE:
            print('Building panorama {}'.format(p_idx))

        # Consider just the datasets in the panorama.
        pan_datasets = [ datasets[p] for p in pano ]
        pan_genes_list = [ genes_list[p] for p in pano ]

        # Do batch correction on those datasets.
        pan_datasets, genes = correct(pan_datasets, pan_genes_list)
        if VERBOSE:
            print('Found {} genes in panorama'.format(len(genes)))
            
        # Store batch corrected result.
        for i, p in enumerate(pano):
            datasets[p] = pan_datasets[i]
            genes_list[p] = genes
            
    if VERBOSE:
        print('Batch corrected panoramas in {:.3f}s'.format(time() - t0))

    ##########################
    ## Final visualization. ##
    ##########################
    
    # Put all the corrected data together for visualization purposes.
    datasets, genes = merge_datasets(datasets, genes_list)
    datasets_dimred, genes = process_data(datasets, genes)
    
    # Label each cell with its dataset of origin.
    labels = np.zeros(n_cells, dtype=int)
    base = 0
    for i, ds in enumerate(datasets):
        labels[base:(base + ds.shape[0])] = i
        base += ds.shape[0]

    visualize(datasets_dimred, labels, NAMESPACE, data_names)
            
