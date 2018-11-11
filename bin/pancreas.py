import numpy as np
from scanorama import *
from scipy.sparse import vstack
from scipy.stats import f_oneway
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

def entropy_test(datasets_dimred, ds_labels):
    from sklearn.cluster import KMeans
    
    ds_labels = np.array(ds_labels)
    X_dimred = np.concatenate(datasets_dimred)
    embedding = None
    
    for k in range(10, 21):
        km = KMeans(n_clusters=k, n_jobs=-1, verbose=0)
        km.fit(X_dimred)

        if False and k % 5 == 0:
            embedding = visualize(
                datasets_dimred,
                km.labels_, NAMESPACE + '_km{}'.format(k),
                [ str(x) for x in range(k) ],
                embedding=embedding
            )
        
        print('k = {}, average normalized entropy = {}'
              .format(k, avg_norm_entropy(ds_labels, km.labels_)))

def avg_norm_entropy(ds_labels, cluster_labels):
    assert(len(ds_labels) == len(cluster_labels))
    
    clusters = sorted(set(cluster_labels))
    datasets = sorted(set(ds_labels))

    Hs = []
    for cluster in clusters:

        cluster_idx = cluster_labels == cluster
        ds_rep = ds_labels[cluster_idx]
        n_cluster = float(sum(cluster_idx))

        H = 0
        for ds in datasets:
            n_ds = float(sum(ds_rep == ds))
            if n_ds == 0: # 0 log 0 = 0
                continue
            H += (n_ds / n_cluster) * np.log(n_ds / n_cluster)
        H *= -1
        H /= np.log(len(datasets))

        Hs.append(H)
        
    return np.mean(Hs)
        
def log_oneway(X, genes, ds_labels):
    for gene_idx, gene in enumerate(genes):
        ds_names = sorted(set(ds_labels))
        dist = []
        for ds in ds_names:
            dist.append(X[ds_labels == ds, gene_idx])
        sys.stdout.write('{}\t'.format(gene))
        print('{}\t{}'.format(*f_oneway(*dist)))
        
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

    #datasets, genes_list, n_cells = load_names(['data/mnn_corrected_pancreas'])
    
    datasets_dimred, datasets, genes = correct(
        datasets, genes_list, ds_names=data_names,
        return_dimred=True, sigma=150
    )
    
    cell_labels = (
        open('data/cell_labels/pancreas_cluster.txt')
        .read().rstrip().split()
    )
    le = LabelEncoder().fit(cell_labels)
    cell_labels = le.transform(cell_labels)
    cell_types = le.classes_
    
    log_oneway(vstack(datasets).toarray(), genes, labels)

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
    
    # Uncorrected.
    datasets, genes_list, n_cells = load_names(data_names)
    datasets, genes = merge_datasets(datasets, genes_list)
    datasets = [ normalize(ds, axis=1) for ds in datasets ]
    datasets_dimred = dimensionality_reduce(datasets)

    log_oneway(vstack(datasets).toarray(), genes, labels)
    
    embedding = visualize(datasets_dimred, labels,
                          NAMESPACE + '_ds_uncorrected', names,
                          perplexity=100, n_iter=400)
    visualize(None, cell_labels,
              NAMESPACE + '_type_uncorrected', cell_types,
              embedding=embedding,
              perplexity=100, n_iter=400)
