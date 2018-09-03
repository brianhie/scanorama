import numpy as np
from scipy.stats import ttest_ind
from unsupervised import silhouette_samples as sil

from scanorama import plt

if __name__ == '__main__':
    labels = np.array(
        open('data/cell_labels/all.txt').read().rstrip().split()
    )
    
    # Scanorama.
    X = np.loadtxt('data/panorama_embedding.txt')
    idx = np.random.choice(X.shape[0], size=20000, replace=False)
    sil_pan = sil(X[idx, :], labels[idx])
    print(np.median(sil_pan))

    # scran MNN.
    X = np.loadtxt('data/mnn_embedding.txt')
    idx = np.random.choice(X.shape[0], size=20000, replace=False)
    sil_mnn = sil(X[idx, :], labels[idx])
    print(np.median(sil_mnn))

    # Seurat CCA.
    X = np.loadtxt('data/cca_embedding.txt')
    idx = np.random.choice(X.shape[0], size=20000, replace=False)
    sil_cca = sil(X[idx, :], labels[idx])
    print(np.median(sil_cca))


    print(ttest_ind(sil_pan, sil_mnn))
    print(ttest_ind(sil_pan, sil_cca))
    
    plt.figure()
    plt.boxplot([ sil_pan, sil_mnn, sil_cca ], showmeans=True)
    plt.title('Distributions of Silhouette Coefficients')
    plt.xticks([ 1, 2, 3 ], [ 'Scanorama', 'scran MNN', 'Seurat CCA' ])
    plt.ylabel('Silhouette Coefficient')
    plt.savefig('silhouette.svg')
