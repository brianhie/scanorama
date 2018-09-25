from fbpca import pca
from matplotlib import cm
import numpy as np
from scanorama import *
from scipy.sparse import vstack
import seaborn as sns
from time import time

from process import load_names, process

if __name__ == '__main__':
    from config import data_names

    datasets, genes_list, n_cells = load_names(data_names)
    datasets, genes = merge_datasets(datasets, genes_list)

    # Singular values.

    k = 300
    U, s, Vt = pca(vstack(datasets), k=k)
    
    xticklabels = [ str(i + 1) if (i+1) % 50 == 0 else ''
                    for i in range(k) ]
    xticklabels[-1] = str(k)
    
    plt.figure()
    sns.barplot(x=(np.array(range(len(s))) + 1), y=s, color='powderblue')
    plt.xticks(range(len(s)), xticklabels)
    plt.xlabel('Singular Value Rank')
    plt.ylabel('Singular Values')
    plt.savefig('top_sv.svg')
    
    # t-SNE of zero percentages.

    X = vstack(datasets)
    nonzero_transcripts = np.sum(X != 0, axis=1)
    zero_pct = nonzero_transcripts / float(X.shape[1])
    zero_pct = [ zero_pct[i, 0] for i in range(zero_pct.shape[0]) ]
    
    embedding = np.loadtxt('data/panorama_embedding.txt')
    
    plt.figure()
    plt.scatter(embedding[:, 0], embedding[:, 1],
                c=list(zero_pct), cmap=cm.get_cmap('Reds'),
                s=1, vmin=0, vmax=1)
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.savefig('zero_pct_tsne.svg')

    # Distribution of nonzero transcripts.
    
    process(data_names, min_trans=0)
    datasets, genes_list, n_cells = load_names(data_names)
    datasets, genes = merge_datasets(datasets, genes_list)
    
    X = vstack(datasets)
    nonzero_transcripts = np.sum(X != 0, axis=1)

    plt.figure()
    sns.distplot(nonzero_transcripts, kde=False, norm_hist=False)
    plt.xlabel('Number of transcripts')
    plt.ylabel('Cells')
    plt.savefig('transcript_distribution.svg')
    
    # Reset.
    process(data_names)

