import networkx as nx
import numpy as np
from scipy.stats import pearsonr, spearmanr
import seaborn as sns
from sklearn.metrics.pairwise import euclidean_distances
import sys

from scanorama import plt

def time_align_visualize(alignments, time, y, namespace='time_align'):
    plt.figure()
    heat = np.flip(alignments + alignments.T +
                   np.eye(alignments.shape[0]), axis=0)
    sns.heatmap(heat, cmap="YlGnBu", vmin=0, vmax=1)
    plt.savefig(namespace + '_heatmap.svg')

    G = nx.from_numpy_matrix(alignments)
    G = nx.maximum_spanning_tree(G)

    pos = {}
    for i in range(len(G.nodes)):
        pos[i] = np.array([time[i], y[i]])

    mst_edges = set(nx.maximum_spanning_tree(G).edges())
    
    weights = [ G[u][v]['weight'] if (not (u, v) in mst_edges) else 8
                for u, v in G.edges() ]
    
    plt.figure()
    nx.draw(G, pos, edges=G.edges(), width=10)
    plt.ylim([-1, 1])
    plt.savefig(namespace + '.svg')

def time_align_correlate(alignments, time):
    time_dist = euclidean_distances(time, time)

    assert(time_dist.shape == alignments.shape)

    time_dists, scores = [], []
    for i in range(time_dist.shape[0]):
        for j in range(time_dist.shape[1]):
            if i >= j:
                continue
            time_dists.append(time_dist[i, j])
            scores.append(alignments[i, j])

    print('Spearman rho = {}'.format(spearmanr(time_dists, scores)))
    print('Pearson rho = {}'.format(pearsonr(time_dists, scores)))
    
def load_alignments_from_log(fname):
    names = []
    alignments = []
    with open(fname) as f:
        row = []
        in_row = False
        while True:
            line = f.readline().strip()
            if not line:
                break
            
            if line.startswith('Loaded'):
                names.append(line.split()[1])
                continue
                
            if line.startswith('[') and not 't-SNE' in line:
                fields = line.replace('[', '').replace(']', '').strip()
                row += [ float(field) for field in fields.split() ]
                in_row = True
                if line.endswith(']'):
                    alignments.append(row)
                    row = []
                    in_row = False
                    continue

            if line.endswith(']'):
                fields = line.replace('[', '').replace(']', '').strip()
                row += [ float(field) for field in fields.split() ]
                if len(alignments) > 0:
                    assert(len(alignments[-1]) == len(row))
                alignments.append(row)
                row = []
                in_row = False

            if in_row and not line.startswith('['):
                fields = line.replace('[', '').replace(']', '').strip()
                row += [ float(field) for field in fields.split() ]

    A = np.array(alignments)
    
    return A, names

def time_dist(datasets_dimred, time):
    time_dist = euclidean_distances(time, time)

    time_dists, scores = [], []
    for i in range(time_dist.shape[0]):
        for j in range(time_dist.shape[1]):
            if i >= j:
                continue
            score = np.mean(euclidean_distances(
                datasets_dimred[i], datasets_dimred[j]
            ))
            time_dists.append(time_dist[i, j])
            scores.append(score)

    print('Spearman rho = {}'.format(spearmanr(time_dists, scores)))
    print('Pearson rho = {}'.format(pearsonr(time_dists, scores)))

if __name__ == '__main__':
    pass
