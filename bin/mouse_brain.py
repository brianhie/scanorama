import numpy as np
from scanorama import *
from scipy.sparse import vstack
from sklearn.preprocessing import normalize, LabelEncoder
import sys
from time import time

from benchmark import write_table
from process import load_names, process

np.random.seed(0)

NAMESPACE = 'mouse_brain'
BATCH_SIZE = 10000

data_names = [
    'data/mouse_brain/nuclei',
    'data/mouse_brain/dropviz/Cerebellum_ALT',
    'data/mouse_brain/dropviz/Cortex_noRep5_FRONTALonly',
    'data/mouse_brain/dropviz/Cortex_noRep5_POSTERIORonly',
    'data/mouse_brain/dropviz/EntoPeduncular',
    'data/mouse_brain/dropviz/GlobusPallidus',
    'data/mouse_brain/dropviz/Hippocampus',
    'data/mouse_brain/dropviz/Striatum',
    'data/mouse_brain/dropviz/SubstantiaNigra',
    'data/mouse_brain/dropviz/Thalamus',
]

if __name__ == '__main__':
    process(data_names, min_trans=100)

    datasets, genes_list, n_cells = load_names(data_names)

    t0 = time()
    datasets_dimred, genes = integrate(
        datasets, genes_list, ds_names=data_names,
        batch_size=BATCH_SIZE,
    )
    print('Integrated panoramas in {:.3f}s'.format(time() - t0))

    t0 = time()
    datasets_dimred, datasets, genes = correct(
        datasets, genes_list, ds_names=data_names,
        return_dimred=True, batch_size=BATCH_SIZE,
    )
    print('Integrated and batch corrected panoramas in {:.3f}s'
          .format(time() - t0))

    labels = []
    names = []
    curr_label = 0
    for i, a in enumerate(datasets_dimred):
        labels += list(np.zeros(a.shape[0]) + curr_label)
        names.append(data_names[i])
        curr_label += 1
    labels = np.array(labels, dtype=int)

    mouse_brain_genes = [
        'Gja1', 'Flt1', 'Gabra6', 'Syt1', 'Gabrb2', 'Gabra1',
        'Meg3', 'Mbp', 'Rgs5',
    ]

    # Downsample for visualization purposes
    for i in range(len(data_names)):
        ds = datasets_dimred[i]
        rand_idx = np.random.choice(ds.shape[0], size=int(ds.shape[0]/10),
                                    replace=False)
        datasets_dimred[i] = ds[rand_idx, :]
        datasets[i] = datasets[i][rand_idx, :]

    embedding = visualize(datasets_dimred,
                          labels, NAMESPACE + '_ds', names,
                          gene_names=mouse_brain_genes, genes=genes,
                          gene_expr=vstack(datasets),
                          multicore_tsne=True,
                          image_suffix='.png')
    np.savetxt('data/{}_embedding.txt'.format(NAMESPACE),
               embedding, delimiter='\t')

    cell_labels = (
        open('data/cell_labels/mouse_brain_cluster.txt')
        .read().rstrip().split()
    )
    le = LabelEncoder().fit(cell_labels)
    cell_labels = le.transform(cell_labels)
    cell_types = le.classes_

    visualize(None,
              cell_labels, NAMESPACE + '_type', cell_types,
              embedding=embedding,  image_suffix='.png')
