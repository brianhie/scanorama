import numpy as np
from scanorama import *
from scipy.stats import ttest_ind
from unsupervised import silhouette_samples as sil

from process import load_names

def test_dimred(datasets, genes, labels, idx, sil_mnn, sil_cca):
    distr = [ sil_mnn, sil_cca ]
    xlabels = [ 'scran MNN', 'Seurat CCA' ]
    
    dimreds = [ 10, 20, 50, 200 ]
    for dimred in dimreds:
        datasets_dimred, genes = process_data(datasets, genes,
                                              dimred=dimred)
        datasets_dimred = assemble(datasets_dimred)
        X = np.concatenate(datasets_dimred)
        distr.append(sil(X[idx, :], labels[idx]))
        xlabels.append(str(dimred))
    
    plt.figure()
    plt.boxplot(distr, showmeans=True)
    plt.xticks(range(1, len(xlabels) + 1), xlabels)
    plt.ylabel('Silhouette Coefficient')
    plt.ylim((-1, 1))
    plt.savefig('param_sensitivity_{}.svg'.format('dimred'))

def test_knn(datasets_dimred, genes, labels, idx, sil_mnn, sil_cca):
    distr = [ sil_mnn, sil_cca ]
    xlabels = [ 'scran MNN', 'Seurat CCA' ]
    
    knns = [ 5, 10, 50, 100 ]
    for knn in knns:
        integrated = assemble(datasets_dimred, knn=knn)
        X = np.concatenate(integrated)
        distr.append(sil(X[idx, :], labels[idx]))
        xlabels.append(str(knn))
    
    plt.figure()
    plt.boxplot(distr, showmeans=True)
    plt.xticks(range(1, len(xlabels) + 1), xlabels)
    plt.ylabel('Silhouette Coefficient')
    plt.ylim((-1, 1))
    plt.savefig('param_sensitivity_{}.svg'.format('knn'))

def test_sigma(datasets_dimred, genes, labels, idx, sil_mnn, sil_cca):
    distr = [ sil_mnn, sil_cca ]
    xlabels = [ 'scran MNN', 'Seurat CCA' ]
    
    sigmas = [ 10, 50, 100, 200 ]
    for sigma in sigmas:
        integrated = assemble(datasets_dimred, sigma=sigma)
        X = np.concatenate(integrated)
        distr.append(sil(X[idx, :], labels[idx]))
        xlabels.append(str(sigma))
    
    plt.figure()
    plt.boxplot(distr, showmeans=True)
    plt.xticks(range(1, len(xlabels) + 1), xlabels)
    plt.ylabel('Silhouette Coefficient')
    plt.ylim((-1, 1))
    plt.savefig('param_sensitivity_{}.svg'.format('sigma'))

def test_alpha(datasets_dimred, genes, labels, idx, sil_mnn, sil_cca):
    distr = [ sil_mnn, sil_cca ]
    xlabels = [ 'scran MNN', 'Seurat CCA' ]
    
    alphas = [ 1e-10, 0.05, 0.20, 0.30 ]
    for alpha in alphas:
        integrated = assemble(datasets_dimred, alpha=alpha)
        X = np.concatenate(integrated)
        distr.append(sil(X[idx, :], labels[idx]))
        xlabels.append(str(alpha))
    
    plt.figure()
    plt.boxplot(distr, showmeans=True)
    plt.xticks(range(1, len(xlabels) + 1), xlabels)
    plt.ylabel('Silhouette Coefficient')
    plt.ylim((-1, 1))
    plt.savefig('param_sensitivity_{}.svg'.format('alpha'))

def test_approx(datasets_dimred, genes, labels, idx, sil_mnn, sil_cca):
    distr = [ sil_mnn, sil_cca ]
    xlabels = [ 'scran MNN', 'Seurat CCA' ]
    
    integrated = assemble(datasets_dimred, approx=False)
    X = np.concatenate(integrated)
    distr.append(sil(X[idx, :], labels[idx]))
    xlabels.append('Exact NN')
    
    plt.figure()
    plt.boxplot(distr, showmeans=True)
    plt.xticks(range(1, len(xlabels) + 1), xlabels)
    plt.ylabel('Silhouette Coefficient')
    plt.ylim((-1, 1))
    plt.savefig('param_sensitivity_{}.svg'.format('approx'))

def fit_tsne(X, perplexity=PERPLEXITY, n_iter=N_ITER,
             learn_rate=200., early_exag=12.):
    try:
        from MulticoreTSNE import MulticoreTSNE
        tsne = MulticoreTSNE(
            n_iter=500, perplexity=perplexity,
            learning_rate=learn_rate,
            early_exaggeration=early_exag,
            random_state=69,
            n_jobs=40
        )
    except ImportError:
        tsne = TSNEApprox(
            n_iter=500, perplexity=perplexity,
            learning_rate=learn_rate,
            early_exaggeration=early_exag,
            random_state=69,
        )
    tsne.fit(X)
    embedding = tsne.embedding_
    return embedding

def test_perplexity(datasets_dimred, genes, labels, idx,
                    sil_mnn, sil_cca):
    distr = [ sil_mnn, sil_cca ]
    xlabels = [ 'scran MNN', 'Seurat CCA' ]
    
    X = np.concatenate(datasets_dimred)

    perplexities = [ 10, 100, 500, 2000 ]
    for perplexity in perplexities:
        embedding = fit_tsne(X, perplexity=perplexity)
        distr.append(sil(embedding[idx, :], labels[idx]))
        xlabels.append(str(perplexity))
    
    plt.figure()
    plt.boxplot(distr, showmeans=True)
    plt.xticks(range(1, len(xlabels) + 1), xlabels)
    plt.ylabel('Silhouette Coefficient')
    plt.ylim((-1, 1))
    plt.savefig('param_sensitivity_{}.svg'.format('perplexity'))

def test_learn_rate(datasets_dimred, genes, labels, idx,
                    sil_mnn, sil_cca):
    distr = [ sil_mnn, sil_cca ]
    xlabels = [ 'scran MNN', 'Seurat CCA' ]
    
    X = np.concatenate(datasets_dimred)

    learn_rates = [ 50., 100., 500., 1000. ]
    for learn_rate in learn_rates:
        embedding = fit_tsne(X, learn_rate=learn_rate)
        distr.append(sil(embedding[idx, :], labels[idx]))
        xlabels.append(str(learn_rate))
    
    plt.figure()
    plt.boxplot(distr, showmeans=True)
    plt.xticks(range(1, len(xlabels) + 1), xlabels)
    plt.ylabel('Silhouette Coefficient')
    plt.ylim((-1, 1))
    plt.savefig('param_sensitivity_{}.svg'.format('learn_rate'))

if __name__ == '__main__':
    with open('conf/panorama.txt') as f:
        data_names = f.read().split()

    labels = np.array(
        open('data/cell_labels/all.txt').read().rstrip().split()
    )
    idx = np.random.choice(102923, size=20000, replace=False)
    
    # scran MNN baseline.
    X = np.loadtxt('../assemble-sc/data/corrected_mnn.txt')
    sil_mnn = sil(X[idx, :], labels[idx])
    print(np.median(sil_mnn))

    # Seurat CCA baseline.
    X = np.loadtxt('data/cca_embedding.txt')
    sil_cca = sil(X[idx, :], labels[idx])
    print(np.median(sil_cca))

    datasets, genes_list, n_cells = load_names(data_names)
    datasets, genes = merge_datasets(datasets, genes_list)

    test_dimred(datasets[:], genes, labels, idx, sil_mnn, sil_cca)

    datasets_dimred, genes = process_data(datasets, genes)
    
    test_knn(datasets_dimred[:], genes, labels, idx, sil_mnn, sil_cca)
    test_sigma(datasets_dimred[:], genes, labels, idx, sil_mnn, sil_cca)
    test_alpha(datasets_dimred[:], genes, labels, idx, sil_mnn, sil_cca)
    test_approx(datasets_dimred[:], genes, labels, idx, sil_mnn, sil_cca)

    datasets_dimred = assemble(datasets_dimred)
    
    test_perplexity(datasets_dimred[:], genes, labels, idx, sil_mnn, sil_cca)
    test_learn_rate(datasets_dimred[:], genes, labels, idx, sil_mnn, sil_cca)
