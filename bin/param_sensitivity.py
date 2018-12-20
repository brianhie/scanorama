import numpy as np
from scanorama import *
from scipy.stats import ttest_ind
from sklearn.metrics import silhouette_samples as sil

from process import load_names, process

def test_knn(datasets_dimred, genes, labels, idx, distr, xlabels):
    knns = [ 5, 10, 50, 100 ]
    len_distr = len(distr)
    for knn in knns:
        integrated = assemble(datasets_dimred[:], knn=knn, sigma=150)
        X = np.concatenate(integrated)
        distr.append(sil(X[idx, :], labels[idx]))
        for d in distr[:len_distr]:
            print(ttest_ind(np.ravel(X[idx, :]), np.ravel(d)))
        xlabels.append(str(knn))
    print('')
    
    plt.figure()
    plt.boxplot(distr, showmeans=True, whis='range')
    plt.xticks(range(1, len(xlabels) + 1), xlabels)
    plt.ylabel('Silhouette Coefficient')
    plt.ylim((-1, 1))
    plt.savefig('param_sensitivity_{}.svg'.format('knn'))

def test_sigma(datasets_dimred, genes, labels, idx, distr, xlabels):
    sigmas = [ 10, 50, 100, 200 ]
    len_distr = len(distr)
    for sigma in sigmas:
        integrated = assemble(datasets_dimred[:], sigma=sigma)
        X = np.concatenate(integrated)
        distr.append(sil(X[idx, :], labels[idx]))
        for d in distr[:len_distr]:
            print(ttest_ind(np.ravel(X[idx, :]), np.ravel(d)))
        xlabels.append(str(sigma))
    print('')
    
    plt.figure()
    plt.boxplot(distr, showmeans=True, whis='range')
    plt.xticks(range(1, len(xlabels) + 1), xlabels)
    plt.ylabel('Silhouette Coefficient')
    plt.ylim((-1, 1))
    plt.savefig('param_sensitivity_{}.svg'.format('sigma'))

def test_alpha(datasets_dimred, genes, labels, idx, distr, xlabels):
    alphas = [ 0, 0.05, 0.20, 0.50 ]
    len_distr = len(distr)
    for alpha in alphas:
        integrated = assemble(datasets_dimred[:], alpha=alpha, sigma=150)
        X = np.concatenate(integrated)
        distr.append(sil(X[idx, :], labels[idx]))
        for d in distr[:len_distr]:
            print(ttest_ind(np.ravel(X[idx, :]), np.ravel(d)))
        xlabels.append(str(alpha))
    print('')
    
    plt.figure()
    plt.boxplot(distr, showmeans=True, whis='range')
    plt.xticks(range(1, len(xlabels) + 1), xlabels)
    plt.ylabel('Silhouette Coefficient')
    plt.ylim((-1, 1))
    plt.savefig('param_sensitivity_{}.svg'.format('alpha'))

def test_approx(datasets_dimred, genes, labels, idx, distr, xlabels):
    integrated = assemble(datasets_dimred[:], approx=False, sigma=150)
    X = np.concatenate(integrated)
    distr.append(sil(X[idx, :], labels[idx]))
    len_distr = len(distr)
    for d in distr[:len_distr]:
        print(ttest_ind(np.ravel(X[idx, :]), np.ravel(d)))
    xlabels.append('Exact NN')
    print('')
    
    plt.figure()
    plt.boxplot(distr, showmeans=True, whis='range')
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
    len_distr = len(distr)
    for perplexity in perplexities:
        embedding = fit_tsne(X, perplexity=perplexity)
        distr.append(sil(embedding[idx, :], labels[idx]))
        for d in distr[:len_distr]:
            print(ttest_ind(np.ravel(X[idx, :]), np.ravel(d)))
        xlabels.append(str(perplexity))
    print('')
    
    plt.figure()
    plt.boxplot(distr, showmeans=True, whis='range')
    plt.xticks(range(1, len(xlabels) + 1), xlabels)
    plt.ylabel('Silhouette Coefficient')
    plt.ylim((-1, 1))
    plt.savefig('param_sensitivity_{}.svg'.format('perplexity'))

def test_learn_rate(datasets_dimred, genes, labels, idx,
                    distr, xlabels):
    X = np.concatenate(datasets_dimred)

    learn_rates = [ 50., 100., 500., 1000. ]
    len_distr = len(distr)
    for learn_rate in learn_rates:
        embedding = fit_tsne(X, learn_rate=learn_rate)
        distr.append(sil(embedding[idx, :], labels[idx]))
        for d in distr[:len_distr]:
            print(ttest_ind(np.ravel(X[idx, :]), np.ravel(d)))
        xlabels.append(str(learn_rate))
    print('')
    
    plt.figure()
    plt.boxplot(distr, showmeans=True, whis='range')
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
    idx = range(labels.shape[0])
    
    datasets, genes_list, n_cells = load_names(data_names)
    datasets, genes = merge_datasets(datasets, genes_list)
    datasets_dimred, genes = process_data(datasets, genes)

    X = np.concatenate(datasets_dimred)
    sil_non = sil(X[idx, :], labels[idx])
    print(np.median(sil_non))
    
    X = np.loadtxt('data/corrected_mnn.txt')
    sil_mnn = sil(X[idx, :], labels[idx])
    print(np.median(sil_mnn))

    X = np.loadtxt('data/corrected_seurat.txt')
    sil_cca = sil(X[idx, :], labels[idx])
    print(np.median(sil_cca))

    distr = [  sil_non, sil_mnn, sil_cca ]
    xlabels = [ 'No correction', 'scran MNN', 'Seurat CCA' ]

    # Test alignment parameters.
    test_approx(datasets_dimred[:], genes, labels, idx, distr[:], xlabels[:])
    test_alpha(datasets_dimred[:], genes, labels, idx, distr[:], xlabels[:])
    test_knn(datasets_dimred[:], genes, labels, idx, distr[:], xlabels[:])
    test_sigma(datasets_dimred[:], genes, labels, idx, distr[:], xlabels[:])

    datasets_dimred = assemble(datasets_dimred)
    
    # Test visualization parameters.
    test_perplexity(datasets_dimred[:], genes, labels, idx, distr[:], xlabels[:])
    test_learn_rate(datasets_dimred[:], genes, labels, idx, distr[:], xlabels[:])
