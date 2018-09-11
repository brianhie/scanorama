import numpy as np
from scanorama import *
from scipy.sparse import vstack
import seaborn as sns
from sklearn.preprocessing import normalize, LabelEncoder
import sys

from process import load_names

NAMESPACE = 'murine_atlases'

with open('conf/murine_atlases.txt') as dn_file:
    data_names = dn_file.read().split()

if __name__ == '__main__':
    datasets, genes_list, n_cells = load_names(data_names)
    
    datasets, genes = merge_datasets(datasets, genes_list)
    datasets_dimred, genes = process_data(datasets, genes)
    
    _, table, _ = find_alignments_table(datasets_dimred, prenormalized=True)
    plt.figure()
    sns.heatmap(table, xticklabels=data_names, yticklabels=data_names)
    plt.tight_layout()
    plt.savefig('murine_heatmap.svg')

    print(connect(datasets_dimred))
    sys.stdout.flush()

    datasets_dimred = assemble(
        datasets_dimred, ds_names=data_names, sigma=50
    )

    labels = []
    names = []
    curr_label = 0
    for i, a in enumerate(datasets):
        labels += list(np.zeros(a.shape[0]) + curr_label)
        names.append(data_names[i])
        curr_label += 1
    labels = np.array(labels, dtype=int)

    embedding = visualize(datasets_dimred,
                          labels, NAMESPACE + '_ds', names)
