import numpy as np
from scanorama import *
from scipy.stats import f_oneway
from sklearn.cluster import KMeans
import sys

def print_oneway(X, genes, ds_labels):
    for gene_idx, gene in enumerate(genes):
        ds_names = sorted(set(ds_labels))
        dist = []
        for ds in ds_names:
            dist.append(X[ds_labels == ds, gene_idx])
        sys.stdout.write('{}\t'.format(gene))
        print('{}\t{}'.format(*f_oneway(*dist)))
        
def entropy_test(datasets_dimred, ds_labels):
    
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
