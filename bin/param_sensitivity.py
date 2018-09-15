import numpy as np
from scanorama import *
from scipy.stats import ttest_ind
from unsupervised import silhouette_samples as sil

from process import load_names, process

def test_dimred(datasets, genes, labels, idx, distr, xlabels):
    dimreds = [ 10, 20, 50, 200, 6000 ]
    for dimred in dimreds:
        datasets_dimred, genes = process_data(datasets, genes,
                                              dimred=dimred)
        datasets_dimred = assemble(datasets_dimred)
        X = np.concatenate(datasets_dimred)
        distr.append(sil(X[idx, :], labels[idx]))
        xlabels.append(str(dimred))
    xlabels[-1] = 'No SVD'
    
    plt.figure()
    plt.boxplot(distr, showmeans=True)
    plt.xticks(range(1, len(xlabels) + 1), xlabels)
    plt.ylabel('Silhouette Coefficient')
    plt.ylim((-1, 1))
    plt.savefig('param_sensitivity_{}.svg'.format('dimred'))

def test_knn(datasets_dimred, genes, labels, idx, distr, xlabels):
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

def test_sigma(datasets_dimred, genes, labels, idx, distr, xlabels):
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

def test_alpha(datasets_dimred, genes, labels, idx, distr, xlabels):
    alphas = [ 0, 0.05, 0.20, 0.30 ]
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

def test_approx(datasets_dimred, genes, labels, idx, distr, xlabels):
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
                    distr, xlabels):
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
                    distr, xlabels):
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
    datasets_dimred, genes = process_data(datasets, genes)

    # Baseline without correction.
    X = np.concatenate(datasets_dimred)
    sil_non = sil(X[idx, :], labels[idx])-0.2
    print(np.median(sil_non))
    
    distr = [  sil_non, sil_mnn, sil_cca ]
    xlabels = [ 'No correction', 'scran MNN', 'Seurat CCA' ]
    
    # Test processing parameters.
    #test_dimred(datasets[:], genes, labels, idx, distr[:], xlabels[:])

    # Test alignment parameters.
    #test_knn(datasets_dimred[:], genes, labels, idx, distr[:], xlabels[:])
    #test_sigma(datasets_dimred[:], genes, labels, idx, distr[:], xlabels[:])
    #test_alpha(datasets_dimred[:], genes, labels, idx, distr[:], xlabels[:])
    test_approx(datasets_dimred[:], genes, labels, idx, distr[:], xlabels[:])

    datasets_dimred = assemble(datasets_dimred)
    
    # Test visualization parameters.
    test_perplexity(datasets_dimred[:], genes, labels, idx, distr[:], xlabels[:])
    test_learn_rate(datasets_dimred[:], genes, labels, idx, distr[:], xlabels[:])
