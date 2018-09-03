import numpy as np
import random
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize

from scanorama import plot_clusters, assemble, plt

np.random.seed(0)
random.seed(0)

class SampleGaussian(object):
    def __init__(self, n_clusters, means,
                 sigmas=None, dim=2):
        self.n_clusters = n_clusters
        
        assert(means.shape[0] == n_clusters)
        assert(means.shape[1] == dim)
        self.means = means

        if sigmas == None:
            self.sigmas = np.zeros((n_clusters, dim, dim))
            for n in range(n_clusters):
                A = np.eye(dim)
                self.sigmas[n, :, :] = A
        else:
            assert(sigmas.shape[0] == n_clusters)
            assert(sigmas.shape[1] == dim)
            assert(sigmas.shape[2] == dim)
            self.sigmas = sigmas

        self.dim = dim

    def sample_N(self, N, weights):
        assert(weights.shape[0] == self.n_clusters)
        weights = weights / np.sum(weights)

        clusters = np.random.choice(
            range(self.n_clusters), p=weights, size=N
        )
        units = np.random.multivariate_normal(
            np.zeros(self.dim),
            np.eye(self.dim),
            size=N
        )
        samples = np.zeros((N, self.dim))
        for i in range(N):
            c = clusters[i]
            samples[i, :] = np.dot(self.sigmas[c, :, :], units[i, :]) + \
                            self.means[c, :]
        return samples, clusters

if __name__ == '__main__':
    # Initialized simulated dataset.
    means = np.array([
        [ 0,  30 ],
        [ 5,  20 ],
        [ 18, 11 ],
        [ 30, 0  ],
    ], dtype='float')
    sg = SampleGaussian(4, means)

    # Sample clusters in 2D.
    samples0, clusters0 = sg.sample_N(
        1000, np.array([ 1, 1, 1, 1], dtype='float')
    )
    plot_clusters(samples0, clusters0)
    samples1, clusters1 = sg.sample_N(
        1001, np.array([ 1, 2, 0, 0], dtype='float')
    )
    samples2, clusters2 = sg.sample_N(
        1002, np.array([ 0, 0, 1, 2], dtype='float')
    )
    samples3, clusters3 = sg.sample_N(
        500, np.array([ 0, 1, 1, 0], dtype='float')
    )

    clusters = [ clusters0, clusters1, clusters2, clusters3 ]
    samples = [ samples0, samples1, samples2, samples3 ]

    # Project to higher dimension.
    Z = np.absolute(np.random.randn(2, 100))
    datasets = [ np.dot(s, Z) for s in samples ]

    # Add batch effect "noise."
    datasets = [ ds + np.random.randn(1, 100) for ds in datasets ]

    # Normalize datasets.
    datasets = [ normalize(ds, axis=1) for ds in datasets ]

    tsne = TSNE(n_iter=400, perplexity=100, verbose=2, random_state=69)

    tsne.fit(np.concatenate(datasets[1:]))
    plot_clusters(tsne.embedding_, np.concatenate(clusters[1:]), s=500)
    plt.title('Uncorrected data')
    plt.savefig('simulation_uncorrected.svg')
    
    # Assemble datasets.
    assembled = assemble(datasets[1:], verbose=1, sigma=1, knn=10,
                         approx=True)
    tsne.fit(datasets[1])
    plot_clusters(tsne.embedding_, clusters[1], s=500)
    plt.title('Dataset 1')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.savefig('simulation_ds1.svg')
    
    tsne.fit(datasets[2])
    plot_clusters(tsne.embedding_, clusters[2], s=500)
    plt.title('Dataset 2')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.savefig('simulation_ds2.svg')
    
    tsne.fit(datasets[3])
    plot_clusters(tsne.embedding_, clusters[3], s=500)
    plt.title('Dataset 3')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.savefig('simulation_ds3.svg')

    tsne.fit(np.concatenate(assembled))
    plot_clusters(tsne.embedding_, np.concatenate(clusters[1:]), s=500)
    plt.title('Assembled data')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.savefig('simulation.svg')
    plt.show()
