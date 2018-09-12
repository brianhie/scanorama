from fbpca import pca
from process import load_names
from scanorama import *
import seaborn as sns

from time import time
import numpy as np

if __name__ == '__main__':
    from config import data_names

    datasets, genes_list, n_cells = load_names(data_names)
    datasets, genes = merge_datasets(datasets, genes_list)

    U, s, Vt = pca(vstack(datasets), k=100)

    plt.figure()
    sns.barplot(x=(np.array(len(s)) + 1),
                y=s)
    plt.xlabel('Singular Value Rank')
    plt.ylabel('Singular Values')
    plt.savefig('top_sv.svg')
