import numpy as np
from scanorama import assemble, correct, visualize, process_data
from scanorama import dimensionality_reduce, merge_datasets
from scipy.sparse import vstack
from sklearn.preprocessing import normalize, LabelEncoder
import sys

from process import load_names

NAMESPACE = 'murine_atlases'

with open('conf/murine_atlases.txt') as dn_file:
    data_names = dn_file.read().split()

if __name__ == '__main__':
    datasets, genes_list, n_cells = load_names(data_names)
    
    datasets_dimred, datasets, genes = correct(
        datasets, genes_list, ds_names=data_names,
        return_dimred=True, sigma=50
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
