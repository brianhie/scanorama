import numpy as np
from scanorama import assemble, correct, visualize, process_data
from scanorama import dimensionality_reduce, merge_datasets
from scipy.sparse import vstack
from sklearn.preprocessing import normalize, LabelEncoder
import sys

from process import load_names

NAMESPACE = 'mouse_brain_large'

data_names = [
    'data/murine_atlases/dropviz/Cerebellum_ALT',
    'data/murine_atlases/dropviz/Cortex_noRep5_FRONTALonly',
    'data/murine_atlases/dropviz/Cortex_noRep5_POSTERIORonly',
    'data/murine_atlases/dropviz/EntoPeduncular',
    'data/murine_atlases/dropviz/GlobusPallidus',
    'data/murine_atlases/dropviz/Hippocampus',
    'data/murine_atlases/dropviz/Striatum',
    'data/murine_atlases/dropviz/SubstantiaNigra',
    'data/murine_atlases/dropviz/Thalamus',
    'data/murine_atlases/neuron_1M/neuron_1M',
]

if __name__ == '__main__':
    datasets, genes_list, n_cells = load_names(data_names)
    
    datasets, genes = merge_datasets(datasets, genes_list, ds_names=data_names)
    datasets_dimred, genes = process_data(datasets, genes)
    
    datasets_dimred = assemble(
        datasets_dimred[:], ds_names=data_names, sigma=50
    )
    #datasets_dimred, datasets, genes = correct(
    #    datasets, genes_list, ds_names=data_names,
    #    return_dimred=True
    #)

    labels = []
    names = []
    curr_label = 0
    for i, a in enumerate(datasets):
        labels += list(np.zeros(a.shape[0]) + curr_label)
        names.append(data_names[i])
        curr_label += 1
    labels = np.array(labels, dtype=int)

    mouse_brain_genes = [
        'Deptor', 'Rarb', 'Tfap2d', 'Fign', 'Arap1', 'Pax3', 'Ntn1',
        'Pax2', 'Slc6a3', 'Fn1', 'Tspan18', 'Pde11a', 'Dlx6os1',
        'Gabra1'
    ]

    embedding = visualize(datasets_dimred,
                          labels, NAMESPACE + '_ds', names,
                          gene_names=mouse_brain_genes, genes=genes,
                          gene_expr=vstack(datasets))

