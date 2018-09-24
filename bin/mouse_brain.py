import numpy as np
from scanorama import *
from scipy.sparse import vstack
from sklearn.preprocessing import normalize, LabelEncoder
import sys
from time import time

from benchmark import write_table
from process import load_names

np.random.seed(0)

NAMESPACE = 'mouse_brain'
BATCH_SIZE = 10000

data_names = [
    'data/murine_atlases/nuclei',
    'data/murine_atlases/dropviz/Cerebellum_ALT',
    'data/murine_atlases/dropviz/Cortex_noRep5_FRONTALonly',
    'data/murine_atlases/dropviz/Cortex_noRep5_POSTERIORonly',
    'data/murine_atlases/dropviz/EntoPeduncular',
    'data/murine_atlases/dropviz/GlobusPallidus',
    'data/murine_atlases/dropviz/Hippocampus',
    'data/murine_atlases/dropviz/Striatum',
    'data/murine_atlases/dropviz/SubstantiaNigra',
    'data/murine_atlases/dropviz/Thalamus',
]

if __name__ == '__main__':
    datasets, genes_list, n_cells = load_names(data_names)

    # Consider DropViz as a single data set.
    #dropviz, genes = merge_datasets(datasets[1:], genes_list[1:], ds_names=data_names)
    #datasets = [ datasets[0], vstack(dropviz) ]
    #genes_list = [ genes_list[0], genes ]
    #data_names = [ data_names[0], 'data/murine_atlases/dropviz' ]

    datasets, genes = merge_datasets(datasets, genes_list, ds_names=data_names)

    #datasets_dimred, genes = process_data(datasets, genes,
    #                                      hvg=0, verbose=True)

    #data = np.load('{}_dimred_{}.npz'.format(data_names[0], NAMESPACE))
    #datasets_dimred[0] = data['ds']
    
    datasets_dimred = []
    for i in range(len(data_names)):
        data = np.load('{}_dimred_{}.npz'.format(data_names[i], NAMESPACE))
        ds = data['ds']
        rand_idx = np.random.choice(ds.shape[0], size=(ds.shape[0]/10),
                                    replace=False)
        datasets_dimred.append(ds[rand_idx, :])
        datasets[i] = datasets[i][rand_idx, :]
        data.close()

    #t0 = time()
    #datasets_dimred = assemble(
    #    datasets_dimred, batch_size=BATCH_SIZE,
    #)
    #print('Integrated panoramas in {:.3f}s'.format(time() - t0))
    
    #datasets_dimred, datasets, genes = correct(
    #    datasets, genes_list, ds_names=data_names,
    #    return_dimred=True, batch_size=BATCH_SIZE
    #)
    #
    labels = []
    names = []
    curr_label = 0
    for i, a in enumerate(datasets_dimred):
        labels += list(np.zeros(a.shape[0]) + curr_label)
        names.append(data_names[i])
        curr_label += 1
    labels = np.array(labels, dtype=int)

    mouse_brain_genes = [
        'Deptor', 'Rarb', 'Tfap2d', 'Fign', 'Arap1', 'Pax3', 'Ntn1',
        'Pax2', 'Slc6a3', 'Fn1', 'Tspan18', 'Pde11a', 'Dlx6os1',
        'Gabra1', 'Meg3', 'Mbp', 'Pdgfra', 'Dock2', 'Rgs5', 'Col1a2',
        'Aldh1l1', 'Dnah11', 'Mybpc1',
    ]

    #for i, ds in enumerate(datasets_dimred):
    #    np.savez('{}_dimred_{}.npz'.format(data_names[i], NAMESPACE), ds=ds)
    #print('Saved data')

    print(len(labels))
    #exit()

    embedding = visualize(datasets_dimred,
                          labels, NAMESPACE + '_ds', names,
                          gene_names=mouse_brain_genes, genes=genes,
                          gene_expr=vstack(datasets),
                          multicore_tsne=True,
                          image_suffix='.png')
    np.savetxt('data/{}_embedding.txt'.format(NAMESPACE),
               embedding, delimiter='\t')

    #embedding = np.loadtxt('data/mouse_brain_embedding.txt')
    #visualize(None, labels, NAMESPACE + '_ds', names,
    #          gene_names=mouse_brain_genes, genes=genes,
    #          gene_expr=vstack(datasets),
    #          multicore_tsne=True, image_suffix='.png',
    #          embedding=embedding)
