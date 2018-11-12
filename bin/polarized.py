import numpy as np
from scanorama import *
from scipy.sparse import vstack
from sklearn.preprocessing import normalize, LabelEncoder
import sys

from process import load_names

NAMESPACE = 'polarized'

data_names = [
    'data/polarized/gmcsf',
    'data/polarized/mcsf',
]

if __name__ == '__main__':
    datasets, genes_list, n_cells = load_names(data_names, log1p=True)
    datasets, genes = merge_datasets(datasets, genes_list)
    datasets_dimred, genes = process_data(datasets, genes)
    
    labels = []
    names = []
    curr_label = 0
    for i, a in enumerate(datasets):
        labels += list(np.zeros(a.shape[0]) + curr_label)
        names.append(data_names[i])
        curr_label += 1
    labels = np.array(labels, dtype=int)

    polarized_genes = [
        'PRDX1'
    ]

    #embedding = visualize(datasets_dimred,
    #                      labels, NAMESPACE + '_ds', names,
    #                      gene_names=polarized_genes, genes=genes,
    #                      gene_expr=vstack(datasets),
    #                      perplexity=100, n_iter=400)

    gene = 'PRDX3'
    
    import seaborn as sns
    from scipy.stats import ttest_ind
    gidx = list(genes).index(gene)
    case1 = datasets[0].toarray()[:, gidx]
    case2 = datasets[1].toarray()[:, gidx]
    print(ttest_ind(case1, case2, equal_var=False))
    print(np.mean(case1))
    print(np.mean(case2))
    plt.figure()
    sns.violinplot(data=[ case1, case2 ])
    sns.stripplot(data=[ case1, case2 ], jitter=True, color='black', size=1)
    plt.xticks([0, 1], ['gmcsf', 'mcsf'])
    plt.savefig('{}.png'.format(gene))
