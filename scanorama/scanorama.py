from annoy import AnnoyIndex
from intervaltree import IntervalTree
from itertools import cycle, islice
import numpy as np
import operator
import random
import scipy
from scipy.sparse import csc_matrix, csr_matrix, vstack
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import rbf_kernel, euclidean_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
import sys
import warnings

from .t_sne_approx import TSNEApprox
from .integration import harmony_integrate
from .utils import plt, dispersion, reduce_dimensionality
from .utils import visualize_cluster, visualize_expr, visualize_dropout

np.random.seed(0)
random.seed(0)

# Default parameters.
ALPHA = 0.10
APPROX = True
BATCH_SIZE = 5000
DIMRED = 100
HVG = None
KNN = 20
N_ITER = 500
PERPLEXITY = 1200
REALIGN = True
SIGMA = 150
VERBOSE = 2

# Do batch correction on a list of data sets.
def correct(datasets_full, genes_list, return_dimred=False,
            batch_size=BATCH_SIZE, verbose=VERBOSE, ds_names=None,
            dimred=DIMRED, approx=APPROX, sigma=SIGMA, alpha=ALPHA, knn=KNN,
            return_dense=False, hvg=None, union=False, realign=REALIGN,
            geosketch=False, geosketch_max=20000):
    """Integrate and batch correct a list of data sets.

    Parameters
    ----------
    datasets_full : `list` of `scipy.sparse.csr_matrix` or of `numpy.ndarray`
        Data sets to integrate and correct.
    genes_list: `list` of `list` of `string`
        List of genes for each data set.
    return_dimred: `bool`, optional (default: `False`)
        In addition to returning batch corrected matrices, also returns
        integrated low-dimesional embeddings.
    batch_size: `int`, optional (default: `5000`)
        The batch size used in the alignment vector computation. Useful when
        correcting very large (>100k samples) data sets. Set to large value
        that runs within available memory.
    verbose: `bool` or `int`, optional (default: 2)
        When `True` or not equal to 0, prints logging output.
    ds_names: `list` of `string`, optional
        When `verbose=True`, reports data set names in logging output.
    dimred: `int`, optional (default: 100)
        Dimensionality of integrated embedding.
    approx: `bool`, optional (default: `True`)
        Use approximate nearest neighbors, greatly speeds up matching runtime.
    sigma: `float`, optional (default: 15)
        Correction smoothing parameter on Gaussian kernel.
    alpha: `float`, optional (default: 0.10)
        Alignment score minimum cutoff.
    knn: `int`, optional (default: 20)
        Number of nearest neighbors to use for matching.
    return_dense: `bool`, optional (default: `False`)
        Return `numpy.ndarray` matrices instead of `scipy.sparse.csr_matrix`.
    hvg: `int`, optional (default: None)
        Use this number of top highly variable genes based on dispersion.

    Returns
    -------
    corrected, genes
        By default (`return_dimred=False`), returns a two-tuple containing a
        list of `scipy.sparse.csr_matrix` each with batch corrected values,
        and a single list of genes containing the intersection of inputted
        genes.

    integrated, corrected, genes
        When `return_dimred=False`, returns a three-tuple containing a list
        of `numpy.ndarray` with integrated low dimensional embeddings, a list
        of `scipy.sparse.csr_matrix` each with batch corrected values, and a
        a single list of genes containing the intersection of inputted genes.
    """
    datasets_full = check_datasets(datasets_full)
    
    datasets, genes = merge_datasets(datasets_full, genes_list,
                                     ds_names=ds_names, union=union)
    datasets_dimred, genes = process_data(datasets, genes, hvg=hvg,
                                          dimred=dimred)
    
    datasets_dimred = assemble(
        datasets_dimred, # Assemble in low dimensional space.
        expr_datasets=datasets, # Modified in place.
        verbose=verbose, knn=knn, sigma=sigma, approx=approx,
        alpha=alpha, ds_names=ds_names, batch_size=batch_size,
        realign=realign, geosketch=geosketch, geosketch_max=geosketch_max,
    )

    if return_dense:
        datasets = [ ds.toarray() for ds in datasets ]

    if return_dimred:
        return datasets_dimred, datasets, genes

    return datasets, genes

# Integrate a list of data sets.
def integrate(datasets_full, genes_list, batch_size=BATCH_SIZE,
              verbose=VERBOSE, ds_names=None, dimred=DIMRED, approx=APPROX,
              sigma=SIGMA, alpha=ALPHA, knn=KNN, harmony=False, geosketch=False,
              geosketch_max=20000, n_iter=1, union=False, hvg=None):
    """Integrate a list of data sets.

    Parameters
    ----------
    datasets_full : `list` of `scipy.sparse.csr_matrix` or of `numpy.ndarray`
        Data sets to integrate and correct.
    genes_list: `list` of `list` of `string`
        List of genes for each data set.
    batch_size: `int`, optional (default: `5000`)
        The batch size used in the alignment vector computation. Useful when
        correcting very large (>100k samples) data sets. Set to large value
        that runs within available memory.
    verbose: `bool` or `int`, optional (default: 2)
        When `True` or not equal to 0, prints logging output.
    ds_names: `list` of `string`, optional
        When `verbose=True`, reports data set names in logging output.
    dimred: `int`, optional (default: 100)
        Dimensionality of integrated embedding.
    approx: `bool`, optional (default: `True`)
        Use approximate nearest neighbors, greatly speeds up matching runtime.
    sigma: `float`, optional (default: 15)
        Correction smoothing parameter on Gaussian kernel.
    alpha: `float`, optional (default: 0.10)
        Alignment score minimum cutoff.
    knn: `int`, optional (default: 20)
        Number of nearest neighbors to use for matching.
    hvg: `int`, optional (default: None)
        Use this number of top highly variable genes based on dispersion.

    Returns
    -------
    integrated, genes
        Returns a two-tuple containing a list of `numpy.ndarray` with
        integrated low dimensional embeddings and a single list of genes
        containing the intersection of inputted genes.
    """
    datasets_full = check_datasets(datasets_full)

    datasets, genes = merge_datasets(datasets_full, genes_list,
                                     ds_names=ds_names, union=union)
    datasets_dimred, genes = process_data(datasets, genes, hvg=hvg,
                                          dimred=dimred)

    for _ in range(n_iter):
        datasets_dimred = assemble(
            datasets_dimred, # Assemble in low dimensional space.
            verbose=verbose, knn=knn, sigma=sigma, approx=approx,
            alpha=alpha, ds_names=ds_names, batch_size=batch_size,
            harmony=harmony, geosketch=geosketch, geosketch_max=geosketch_max,
        )

    return datasets_dimred, genes

# Batch correction with scanpy's AnnData object.
def correct_scanpy(adatas, **kwargs):
    """Batch correct a list of `scanpy.api.AnnData`.

    Parameters
    ----------
    adatas : `list` of `scanpy.api.AnnData`
        Data sets to integrate and/or correct.
    kwargs : `dict`
        See documentation for the `correct()` method for a full list of
        parameters to use for batch correction.

    Returns
    -------
    corrected
        By default (`return_dimred=False`), returns a list of
        `scanpy.api.AnnData` with batch corrected values in the `.X` field.

    corrected, integrated
        When `return_dimred=False`, returns a two-tuple containing a list of
        `np.ndarray` with integrated low-dimensional embeddings and a list
        of `scanpy.api.AnnData` with batch corrected values in the `.X`
        field.
    """
    if 'return_dimred' in kwargs and kwargs['return_dimred']:
        datasets_dimred, datasets, genes = correct(
            [adata.X for adata in adatas],
            [adata.var_names.values for adata in adatas],
            **kwargs
        )
    else:
        datasets, genes = correct(
            [adata.X for adata in adatas],
            [adata.var_names.values for adata in adatas],
            **kwargs
        )

    new_adatas = []
    for i, adata in enumerate(adatas):
        adata.X = datasets[i]
        new_adatas.append(adata)

    if 'return_dimred' in kwargs and kwargs['return_dimred']:
        return datasets_dimred, new_adatas
    else:
        return new_adatas
    
# Integration with scanpy's AnnData object.
def integrate_scanpy(adatas, **kwargs):
    """Integrate a list of `scanpy.api.AnnData`.

    Parameters
    ----------
    adatas : `list` of `scanpy.api.AnnData`
        Data sets to integrate.
    kwargs : `dict`
        See documentation for the `integrate()` method for a full list of
        parameters to use for batch correction.

    Returns
    -------
    integrated
        Returns a list of `np.ndarray` with integrated low-dimensional
        embeddings.
    """
    datasets_dimred, genes = integrate(
        [adata.X for adata in adatas],
        [adata.var_names.values for adata in adatas],
        **kwargs
    )

    return datasets_dimred

# Visualize a scatter plot with cluster labels in the
# `cluster' variable.
def plot_clusters(coords, clusters, s=1):
    if coords.shape[0] != clusters.shape[0]:
        sys.stderr.write(
            'Error: mismatch, {} cells, {} labels\n'
            .format(coords.shape[0], clusters.shape[0])
        )
        exit(1)

    colors = np.array(
        list(islice(cycle([
            '#377eb8', '#ff7f00', '#4daf4a',
            '#f781bf', '#a65628', '#984ea3',
            '#999999', '#e41a1c', '#dede00',
            '#ffe119', '#e6194b', '#ffbea3',
            '#911eb4', '#46f0f0', '#f032e6',
            '#d2f53c', '#008080', '#e6beff',
            '#aa6e28', '#800000', '#aaffc3',
            '#808000', '#ffd8b1', '#000080',
            '#808080', '#fabebe', '#a3f4ff'
        ]), int(max(clusters) + 1)))
    )

    plt.figure()
    plt.scatter(coords[:, 0], coords[:, 1],
                c=colors[clusters], s=s)

# Put datasets into a single matrix with the intersection of all genes.
def merge_datasets(datasets, genes, ds_names=None, verbose=True,
                   union=False):
    if union:
        sys.stderr.write(
            'WARNING: Integrating based on the union of genes is '
            'highly discouraged, consider taking the intersection '
            'or requantifying gene expression.\n'
        )
    
    # Find genes in common.
    keep_genes = set()
    for idx, gene_list in enumerate(genes):
        if len(keep_genes) == 0:
            keep_genes = set(gene_list)
        elif union:
            keep_genes |= set(gene_list)
        else:
            keep_genes &= set(gene_list)
        if not union and not ds_names is None and verbose:
            print('After {}: {} genes'.format(ds_names[idx], len(keep_genes)))
        if len(keep_genes) == 0:
            print('Error: No genes found in all datasets, exiting...')
            exit(1)
    if verbose:
        print('Found {} genes among all datasets'
              .format(len(keep_genes)))

    if union:
        union_genes = sorted(keep_genes)
        for i in range(len(datasets)):
            if verbose:
                print('Processing data set {}'.format(i))
            X_new = np.zeros((datasets[i].shape[0], len(union_genes)))
            X_old = csc_matrix(datasets[i])
            gene_to_idx = { gene: idx for idx, gene in enumerate(genes[i]) }
            for j, gene in enumerate(union_genes):
                if gene in gene_to_idx:
                    X_new[:, j] = X_old[:, gene_to_idx[gene]].toarray().flatten()
            datasets[i] = csr_matrix(X_new)
        ret_genes = np.array(union_genes)
    else:
        # Only keep genes in common.
        ret_genes = np.array(sorted(keep_genes))
        for i in range(len(datasets)):
            # Remove duplicate genes.
            uniq_genes, uniq_idx = np.unique(genes[i], return_index=True)
            datasets[i] = datasets[i][:, uniq_idx]
    
            # Do gene filtering.
            gene_sort_idx = np.argsort(uniq_genes)
            gene_idx = [ idx for idx in gene_sort_idx
                         if uniq_genes[idx] in keep_genes ]
            datasets[i] = datasets[i][:, gene_idx]
            assert(np.array_equal(uniq_genes[gene_idx], ret_genes))
                
    return datasets, ret_genes

def check_datasets(datasets_full):
    datasets_new = []
    for i, ds in enumerate(datasets_full):
        if issubclass(type(ds), np.ndarray):
            datasets_new.append(csr_matrix(ds))
        elif issubclass(type(ds), scipy.sparse.csr.csr_matrix):
            datasets_new.append(ds)
        else:
            sys.stderr.write('ERROR: Data sets must be numpy array or '
                             'scipy.sparse.csr_matrix, received type '
                             '{}.\n'.format(type(ds)))
            exit(1)
    return datasets_new

# Randomized SVD.
def dimensionality_reduce(datasets, dimred=DIMRED):
    X = vstack(datasets)
    X = reduce_dimensionality(X, dim_red_k=dimred)
    datasets_dimred = []
    base = 0
    for ds in datasets:
        datasets_dimred.append(X[base:(base + ds.shape[0]), :])
        base += ds.shape[0]
    return datasets_dimred

# Normalize and reduce dimensionality.
def process_data(datasets, genes, hvg=HVG, dimred=DIMRED, verbose=False):
    # Only keep highly variable genes
    if not hvg is None and hvg > 0 and hvg < len(genes):
        if verbose:
            print('Highly variable filter...')
        X = vstack(datasets)
        disp = dispersion(X)
        highest_disp_idx = np.argsort(disp[0])[::-1]
        top_genes = set(genes[highest_disp_idx[range(hvg)]])
        for i in range(len(datasets)):
            gene_idx = [ idx for idx, g_i in enumerate(genes)
                         if g_i in top_genes ]
            datasets[i] = datasets[i][:, gene_idx]
        genes = np.array(sorted(top_genes))

    # Normalize.
    if verbose:
        print('Normalizing...')
    for i, ds in enumerate(datasets):
        datasets[i] = normalize(ds, axis=1)
    
    # Compute compressed embedding.
    if dimred > 0:
        if verbose:
            print('Reducing dimension...')
        datasets_dimred = dimensionality_reduce(datasets, dimred=dimred)
        if verbose:
            print('Done processing.')
        return datasets_dimred, genes

    if verbose:
        print('Done processing.')

    return datasets, genes

# Plot t-SNE visualization.
def visualize(assembled, labels, namespace, data_names,
              gene_names=None, gene_expr=None, genes=None,
              n_iter=N_ITER, perplexity=PERPLEXITY, verbose=VERBOSE,
              learn_rate=200., early_exag=12., embedding=None,
              shuffle_ds=False, size=1, multicore_tsne=True,
              image_suffix='.svg', viz_cluster=False):
    # Fit t-SNE.
    if embedding is None:
        try:
            from MulticoreTSNE import MulticoreTSNE
            tsne = MulticoreTSNE(
                n_iter=n_iter, perplexity=perplexity,
                verbose=verbose, random_state=69,
                learning_rate=learn_rate,
                early_exaggeration=early_exag,
                n_jobs=40
            )
        except ImportError:
            multicore_tsne = False

        if not multicore_tsne:
            tsne = TSNEApprox(
                n_iter=n_iter, perplexity=perplexity,
                verbose=verbose, random_state=69,
                learning_rate=learn_rate,
                early_exaggeration=early_exag
            )

        tsne.fit(np.concatenate(assembled))
        embedding = tsne.embedding_

    if shuffle_ds:
        rand_idx = range(embedding.shape[0])
        random.shuffle(list(rand_idx))
        embedding = embedding[rand_idx, :]
        labels = labels[rand_idx]

    # Plot clusters together.
    plot_clusters(embedding, labels, s=size)
    plt.title(('Panorama ({} iter, perplexity: {}, sigma: {}, ' +
               'knn: {}, hvg: {}, dimred: {}, approx: {})')
              .format(n_iter, perplexity, SIGMA, KNN, HVG,
                      DIMRED, APPROX))
    plt.savefig(namespace + image_suffix, dpi=500)

    # Plot clusters individually.
    if viz_cluster and not shuffle_ds:
        for i in range(len(data_names)):
            visualize_cluster(embedding, i, labels,
                              cluster_name=data_names[i], size=size,
                              viz_prefix=namespace,
                              image_suffix=image_suffix)

    # Plot gene expression levels.
    if (not gene_names is None) and \
       (not gene_expr is None) and \
       (not genes is None):
        if shuffle_ds:
            gene_expr = gene_expr[rand_idx, :]
        for gene_name in gene_names:
            visualize_expr(gene_expr, embedding,
                           genes, gene_name, size=size,
                           viz_prefix=namespace,
                           image_suffix=image_suffix)

    return embedding

# Exact nearest neighbors search.
def nn(ds1, ds2, knn=KNN, metric_p=2):
    # Find nearest neighbors of first dataset.
    nn_ = NearestNeighbors(knn, p=metric_p)
    nn_.fit(ds2)
    ind = nn_.kneighbors(ds1, return_distance=False)

    match = set()
    for a, b in zip(range(ds1.shape[0]), ind):
        for b_i in b:
            match.add((a, b_i))

    return match

# Approximate nearest neighbors using locality sensitive hashing.
def nn_approx(ds1, ds2, knn=KNN, metric='manhattan', n_trees=10):
    # Build index.
    a = AnnoyIndex(ds2.shape[1], metric=metric)
    for i in range(ds2.shape[0]):
        a.add_item(i, ds2[i, :])
    a.build(n_trees)

    # Search index.
    ind = []
    for i in range(ds1.shape[0]):
        ind.append(a.get_nns_by_vector(ds1[i, :], knn, search_k=-1))
    ind = np.array(ind)

    # Match.
    match = set()
    for a, b in zip(range(ds1.shape[0]), ind):
        for b_i in b:
            match.add((a, b_i))

    return match

# Find mutual nearest neighbors.
def mnn(ds1, ds2, knn=KNN, approx=APPROX):
    # Find nearest neighbors in first direction.
    if approx:
        match1 = nn_approx(ds1, ds2, knn=knn)
    else:
        match1 = nn(ds1, ds2, knn=knn)

    # Find nearest neighbors in second direction.
    if approx:
        match2 = nn_approx(ds2, ds1, knn=knn)
    else:
        match2 = nn(ds2, ds1, knn=knn)
        
    # Compute mutual nearest neighbors.
    mutual = match1 & set([ (b, a) for a, b in match2 ])

    return mutual

# Visualize alignment between two datasets.
def plot_mapping(curr_ds, curr_ref, ds_ind, ref_ind):
    tsne = TSNE(n_iter=400, verbose=VERBOSE, random_state=69)
    
    tsne.fit(curr_ds)
    plt.figure()
    coords_ds = tsne.embedding_[:, :]
    coords_ds[:, 1] += 100
    plt.scatter(coords_ds[:, 0], coords_ds[:, 1])
    
    tsne.fit(curr_ref)
    coords_ref = tsne.embedding_[:, :]
    plt.scatter(coords_ref[:, 0], coords_ref[:, 1])

    x_list, y_list = [], []
    for r_i, c_i in zip(ds_ind, ref_ind):
        x_list.append(coords_ds[r_i, 0])
        x_list.append(coords_ref[c_i, 0])
        x_list.append(None)
        y_list.append(coords_ds[r_i, 1])
        y_list.append(coords_ref[c_i, 1])
        y_list.append(None)
    plt.plot(x_list, y_list, 'b-', alpha=0.3)
    plt.show()
    

# Populate a table (in place) that stores mutual nearest neighbors
# between datasets.
def fill_table(table, i, curr_ds, datasets, base_ds=0,
               knn=KNN, approx=APPROX):
    curr_ref = np.concatenate(datasets)
    if approx:
        match = nn_approx(curr_ds, curr_ref, knn=knn)
    else:
        match = nn(curr_ds, curr_ref, knn=knn, metric_p=1)

    # Build interval tree.
    itree_ds_idx = IntervalTree()
    itree_pos_base = IntervalTree()
    pos = 0
    for j in range(len(datasets)):
        n_cells = datasets[j].shape[0]
        itree_ds_idx[pos:(pos + n_cells)] = base_ds + j
        itree_pos_base[pos:(pos + n_cells)] = pos
        pos += n_cells
    
    # Store all mutual nearest neighbors between datasets.
    for d, r in match:
        interval = itree_ds_idx[r]
        assert(len(interval) == 1)
        j = interval.pop().data
        interval = itree_pos_base[r]
        assert(len(interval) == 1)
        base = interval.pop().data
        if not (i, j) in table:
            table[(i, j)] = set()
        table[(i, j)].add((d, r - base))
        assert(r - base >= 0)

gs_idxs = None
        
# Fill table of alignment scores.
def find_alignments_table(datasets, knn=KNN, approx=APPROX, verbose=VERBOSE,
                          prenormalized=False, harmony=False, geosketch=False,
                          geosketch_max=20000):
    if not prenormalized:
        datasets = [ normalize(ds, axis=1) for ds in datasets ]

    if geosketch:
        # Only match cells in geometric sketches.
        from ample import gs, uniform
        global gs_idxs
        if gs_idxs is None:
            gs_idxs = [ uniform(X, geosketch_max, replace=False)
                        if X.shape[0] > geosketch_max else range(X.shape[0])
                        for X in datasets ]
        datasets = [ datasets[i][gs_idx, :] for i, gs_idx in enumerate(gs_idxs) ]

        if harmony:
            datasets = harmony_integrate(datasets)
    
    table = {}
    for i in range(len(datasets)):
        if len(datasets[:i]) > 0:
            fill_table(table, i, datasets[i], datasets[:i], knn=knn,
                       approx=approx)
        if len(datasets[i+1:]) > 0:
            fill_table(table, i, datasets[i], datasets[i+1:],
                       knn=knn, base_ds=i+1, approx=approx)
    # Count all mutual nearest neighbors between datasets.
    matches = {}
    table1 = {}
    if verbose > 1:
        table_print = np.zeros((len(datasets), len(datasets)))
    for i in range(len(datasets)):
        for j in range(len(datasets)):
            if i >= j:
                continue
            if not (i, j) in table or not (j, i) in table:
                continue
            match_ij = table[(i, j)]
            match_ji = set([ (b, a) for a, b in table[(j, i)] ])
            matches[(i, j)] = match_ij & match_ji
            
            table1[(i, j)] = (max(
                float(len(set([ idx for idx, _ in matches[(i, j)] ]))) /
                datasets[i].shape[0],
                float(len(set([ idx for _, idx in matches[(i, j)] ]))) /
                datasets[j].shape[0]
            ))
            if verbose > 1:
                table_print[i, j] += table1[(i, j)]

            if geosketch:
                # Translate matches within geometric sketches to original indices.
                matches_mnn = matches[(i, j)]
                matches[(i, j)] = [
                    (gs_idxs[i][a], gs_idxs[j][b]) for a, b in matches_mnn
                ]
            
    if verbose > 1:
        print(table_print)
        return table1, table_print, matches
    else:
        return table1, None, matches
    
# Find the matching pairs of cells between datasets.
def find_alignments(datasets, knn=KNN, approx=APPROX, verbose=VERBOSE,
                    alpha=ALPHA, prenormalized=False, harmony=False,
                    geosketch=False, geosketch_max=20000):
    table1, _, matches = find_alignments_table(
        datasets, knn=knn, approx=approx, verbose=verbose,
        prenormalized=prenormalized, harmony=harmony,
        geosketch=geosketch, geosketch_max=geosketch_max
    )

    alignments = [ (i, j) for (i, j), val in reversed(
        sorted(table1.items(), key=operator.itemgetter(1))
    ) if val > alpha ]

    return alignments, matches

# Find connections between datasets to identify panoramas.
def connect(datasets, knn=KNN, approx=APPROX, alpha=ALPHA,
            verbose=VERBOSE):
    # Find alignments.
    alignments, _ = find_alignments(
        datasets, knn=knn, approx=approx, alpha=alpha,
        verbose=verbose
    )
    if verbose:
        print(alignments)
    
    panoramas = []
    connected = set()
    for i, j in alignments:
        # See if datasets are involved in any current panoramas.
        panoramas_i = [ panoramas[p] for p in range(len(panoramas))
                        if i in panoramas[p] ]
        assert(len(panoramas_i) <= 1)
        panoramas_j = [ panoramas[p] for p in range(len(panoramas))
                        if j in panoramas[p] ]
        assert(len(panoramas_j) <= 1)
        
        if len(panoramas_i) == 0 and len(panoramas_j) == 0:
            panoramas.append([ i ])
            panoramas_i = [ panoramas[-1] ]
            
        if len(panoramas_i) == 0:
            panoramas_j[0].append(i)
        elif len(panoramas_j) == 0:
            panoramas_i[0].append(j)
        elif panoramas_i[0] != panoramas_j[0]:
            panoramas_i[0] += panoramas_j[0]
            panoramas.remove(panoramas_j[0])

        connected.add(i)
        connected.add(j)

    for i in range(len(datasets)):
        if not i in connected:
            panoramas.append([ i ])

    return panoramas

# To reduce memory usage, split bias computation into batches.
def batch_bias(curr_ds, match_ds, bias, batch_size=None, sigma=SIGMA):
    if batch_size is None:
        weights = rbf_kernel(curr_ds, match_ds, gamma=0.5*sigma)
        avg_bias = np.dot(weights, bias) / \
                   np.tile(np.sum(weights, axis=1),
                           (curr_ds.shape[1], 1)).T
        return avg_bias
    
    base = 0
    avg_bias = np.zeros(curr_ds.shape)
    denom = np.zeros(curr_ds.shape[0])
    while base < match_ds.shape[0]:
        batch_idx = range(
            base, min(base + batch_size, match_ds.shape[0])
        )
        weights = rbf_kernel(curr_ds, match_ds[batch_idx, :],
                             gamma=0.5*sigma)
        avg_bias += np.dot(weights, bias[batch_idx, :])
        denom += np.sum(weights, axis=1)
        base += batch_size

    avg_bias /= np.tile(denom, (curr_ds.shape[1], 1)).T

    return avg_bias

# Compute nonlinear translation vectors between dataset
# and a reference.
def transform(curr_ds, curr_ref, ds_ind, ref_ind, sigma, cn=False,
              batch_size=None):
    # Compute the matching.
    match_ds = curr_ds[ds_ind, :]
    match_ref = curr_ref[ref_ind, :]
    bias = match_ref - match_ds
    if cn:
        match_ds = match_ds.toarray()
        curr_ds = curr_ds.toarray()
        bias = bias.toarray()

    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            avg_bias = batch_bias(curr_ds, match_ds, bias, sigma=sigma,
                                  batch_size=batch_size)
        except RuntimeWarning:
            sys.stderr.write('WARNING: Oversmoothing detected, will not batch '
                             'correct, consider lowering sigma value.\n')
            return csr_matrix(curr_ds.shape, dtype=float)
        except MemoryError:
            if batch_size is None:
                sys.stderr.write('WARNING: Out of memory, consider turning on '
                                 'batched computation with batch_size parameter.\n')
            else:
                sys.stderr.write('WARNING: Out of memory, consider lowering '
                                 'the batch_size parameter.\n')
            return csr_matrix(curr_ds.shape, dtype=float)
        
    if cn:
        avg_bias = csr_matrix(avg_bias)
    
    return avg_bias
    
# Finds alignments between datasets and uses them to construct
# panoramas. "Merges" datasets by correcting gene expression
# values.
def assemble(datasets, verbose=VERBOSE, view_match=False, knn=KNN,
             sigma=SIGMA, approx=APPROX, alpha=ALPHA, expr_datasets=None,
             ds_names=None, batch_size=None, realign=REALIGN, harmony=False,
             geosketch=False, geosketch_max=20000, alignments=None, matches=None):
    if len(datasets) == 1:
        return datasets

    if alignments is None and matches is None:
        alignments, matches = find_alignments(
            datasets, knn=knn, approx=approx, alpha=alpha, verbose=verbose,
            harmony=harmony, geosketch=geosketch, geosketch_max=geosketch_max
        )
    
    ds_assembled = {}
    panoramas = []
    for i, j in alignments:
        if verbose:
            if ds_names is None:
                print('Processing datasets {}'.format((i, j)))
            else:
                print('Processing datasets {} <=> {}'.
                      format(ds_names[i], ds_names[j]))
        
        # Only consider a dataset a fixed amount of times.
        if not i in ds_assembled:
            ds_assembled[i] = 0
        ds_assembled[i] += 1
        if not j in ds_assembled:
            ds_assembled[j] = 0
        ds_assembled[j] += 1
        if ds_assembled[i] > 3 and ds_assembled[j] > 3:
            continue
                
        # See if datasets are involved in any current panoramas.
        panoramas_i = [ panoramas[p] for p in range(len(panoramas))
                        if i in panoramas[p] ]
        assert(len(panoramas_i) <= 1)
        panoramas_j = [ panoramas[p] for p in range(len(panoramas))
                        if j in panoramas[p] ]
        assert(len(panoramas_j) <= 1)
        
        if len(panoramas_i) == 0 and len(panoramas_j) == 0:
            if datasets[i].shape[0] < datasets[j].shape[0]:
                i, j = j, i
            panoramas.append([ i ])
            panoramas_i = [ panoramas[-1] ]

        # Map dataset i to panorama j.
        if len(panoramas_i) == 0:
            curr_ds = datasets[i]
            curr_ref = np.concatenate([ datasets[p] for p in panoramas_j[0] ])
            
            match = []
            base = 0
            for p in panoramas_j[0]:
                if i < p and (i, p) in matches:
                    match.extend([ (a, b + base) for a, b in matches[(i, p)] ])
                elif i > p and (p, i) in matches:
                    match.extend([ (b, a + base) for a, b in matches[(p, i)] ])
                base += datasets[p].shape[0]

            ds_ind = [ a for a, _ in match ]
            ref_ind = [ b for _, b in match ]
                    
            bias = transform(curr_ds, curr_ref, ds_ind, ref_ind, sigma,
                             batch_size=batch_size)
            datasets[i] = curr_ds + bias
            
            if expr_datasets:
                curr_ds = expr_datasets[i]
                curr_ref = vstack([ expr_datasets[p]
                                    for p in panoramas_j[0] ])
                bias = transform(curr_ds, curr_ref, ds_ind, ref_ind, sigma,
                                 cn=True, batch_size=batch_size)
                expr_datasets[i] = curr_ds + bias
            
            panoramas_j[0].append(i)
            
        # Map dataset j to panorama i.
        elif len(panoramas_j) == 0:
            curr_ds = datasets[j]
            curr_ref = np.concatenate([ datasets[p] for p in panoramas_i[0] ])
            
            match = []
            base = 0
            for p in panoramas_i[0]:
                if j < p and (j, p) in matches:
                    match.extend([ (a, b + base) for a, b in matches[(j, p)] ])
                elif j > p and (p, j) in matches:
                    match.extend([ (b, a + base) for a, b in matches[(p, j)] ])
                base += datasets[p].shape[0]
                
            ds_ind = [ a for a, _ in match ]
            ref_ind = [ b for _, b in match ]
            
            bias = transform(curr_ds, curr_ref, ds_ind, ref_ind, sigma,
                             batch_size=batch_size)
            datasets[j] = curr_ds + bias
            
            if expr_datasets:
                curr_ds = expr_datasets[j]
                curr_ref = vstack([ expr_datasets[p]
                                    for p in panoramas_i[0] ])
                bias = transform(curr_ds, curr_ref, ds_ind, ref_ind, sigma,
                                 cn=True, batch_size=batch_size)
                expr_datasets[j] = curr_ds + bias
                
            panoramas_i[0].append(j)

        # Merge two panoramas together.
        else:
            curr_ds = np.concatenate([ datasets[p] for p in panoramas_i[0] ])
            curr_ref = np.concatenate([ datasets[p] for p in panoramas_j[0] ])

            # Find base indices into each panorama.
            base_i = 0
            for p in panoramas_i[0]:
                if p == i: break
                base_i += datasets[p].shape[0]
            base_j = 0
            for p in panoramas_j[0]:
                if p == j: break
                base_j += datasets[p].shape[0]
            
            # Find matching indices.
            match = []
            base = 0
            for p in panoramas_i[0]:
                if p == i and j < p and (j, p) in matches:
                    match.extend([ (b + base, a + base_j)
                                   for a, b in matches[(j, p)] ])
                elif p == i and j > p and (p, j) in matches:
                    match.extend([ (a + base, b + base_j)
                                   for a, b in matches[(p, j)] ])
                base += datasets[p].shape[0]
            base = 0
            for p in panoramas_j[0]:
                if p == j and i < p and (i, p) in matches:
                    match.extend([ (a + base_i, b + base)
                                   for a, b in matches[(i, p)] ])
                elif p == j and i > p and (p, i) in matches:
                    match.extend([ (b + base_i, a + base)
                                   for a, b in matches[(p, i)] ])
                base += datasets[p].shape[0]
                
            ds_ind = [ a for a, _ in match ]
            ref_ind = [ b for _, b in match ]
            
            # Apply transformation to entire panorama.
            bias = transform(curr_ds, curr_ref, ds_ind, ref_ind, sigma,
                             batch_size=batch_size)
            curr_ds += bias
            base = 0
            for p in panoramas_i[0]:
                n_cells = datasets[p].shape[0]
                datasets[p] = curr_ds[base:(base + n_cells), :]
                base += n_cells
            
            if realign and not expr_datasets is None:
                # Realignment may improve quality of batch correction at cost
                # of potentially higher memory usage.
                curr_ds = vstack([ expr_datasets[p]
                                   for p in panoramas_i[0] ])
                curr_ref = vstack([ expr_datasets[p]
                                    for p in panoramas_j[0] ])
                bias = transform(curr_ds, curr_ref, ds_ind, ref_ind, sigma,
                                 cn=True, batch_size=batch_size)
                curr_ds += bias
                base = 0
                for p in panoramas_i[0]:
                    n_cells = expr_datasets[p].shape[0]
                    expr_datasets[p] = curr_ds[base:(base + n_cells), :]
                    base += n_cells
                
            # Merge panoramas i and j and delete one.
            if panoramas_i[0] != panoramas_j[0]:
                panoramas_i[0] += panoramas_j[0]
                panoramas.remove(panoramas_j[0])

        # Visualize.
        if view_match:
            plot_mapping(curr_ds, curr_ref, ds_ind, ref_ind)

    return datasets

# Non-optimal dataset assembly. Simply accumulate datasets into a
# reference.
def assemble_accum(datasets, verbose=VERBOSE, knn=KNN, sigma=SIGMA,
                   approx=APPROX, batch_size=None):
    if len(datasets) == 1:
        return datasets
    
    for i in range(len(datasets) - 1):
        j = i + 1
        
        if verbose:
            print('Processing datasets {}'.format((i, j)))

        ds1 = datasets[j]
        ds2 = np.concatenate(datasets[:i+1])
        match = mnn(ds1, ds2, knn=knn, approx=approx)
        
        ds_ind = [ a for a, _ in match ]
        ref_ind = [ b for _, b in match ]
                    
        bias = transform(ds1, ds2, ds_ind, ref_ind, sigma,
                         batch_size=batch_size)
        datasets[j] = ds1 + bias

    return datasets

def interpret_alignments(datasets, expr_datasets, genes,
                         verbose=VERBOSE, knn=KNN, approx=APPROX,
                         alpha=ALPHA, n_permutations=None):
    if n_permutations is None:
        n_permutations = float(len(genes) * 30)
    
    alignments, matches = find_alignments(
        datasets, knn=knn, approx=approx, alpha=alpha, verbose=verbose
    )

    for i, j in alignments:
        # Compute average bias vector that aligns two datasets together.
        ds_i = expr_datasets[i]
        ds_j = expr_datasets[j]
        if i < j:
            match = matches[(i, j)]
        else:
            match = matches[(j, i)]
        i_ind = [ a for a, _ in match ]
        j_ind = [ b for _, b in match ]
        avg_bias = np.absolute(
            np.mean(ds_j[j_ind, :] - ds_i[i_ind, :], axis=0)
        )

        # Construct null distribution and compute p-value.
        null_bias = (
            ds_j[np.random.randint(ds_j.shape[0], size=n_permutations), :] -
            ds_i[np.random.randint(ds_i.shape[0], size=n_permutations), :]
        )
        p = ((np.sum(np.greater_equal(
            np.absolute(np.tile(avg_bias, (n_permutations, 1))),
            np.absolute(null_bias)
        ), axis=0, dtype=float) + 1) / (n_permutations + 1))

        print('>>>> Stats for alignment {}'.format((i, j)))
        for k in range(len(p)):
            print('{}\t{}'.format(genes[k], p[k]))
