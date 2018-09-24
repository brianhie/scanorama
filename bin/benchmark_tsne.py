import numpy as np
from scanorama import *
from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder

def load_data(fname):
    n_cells = 0
    with open(fname) as f:
        genes = f.readline().rstrip().split('\t')

        datasets = []
        names = []
        curr_name = ''
        name = ''
        expr_lists = []
        for line in f:
            fields = line.rstrip().split('\t')
            name = fields[0].rstrip('1234567890').lstrip('X')
            if curr_name != name:
                if curr_name != '':
                    datasets.append(np.array(expr_lists, dtype=float))
                    names.append(curr_name)
                expr_lists = []
                curr_name = name
            expr_lists.append([
                float(x) for x in fields[1:]
            ])
            n_cells += 1

        if curr_name != '':
            datasets.append(np.array(expr_lists, dtype=float))
            names.append(curr_name)
                
    return datasets, genes, names, n_cells

METHOD = 'seurat'
NAMESPACE = 'different3'

if __name__ == '__main__':
    datasets, genes, names, n_cells = load_data(
        'data/corrected_{}_{}.txt'.format(METHOD, NAMESPACE)
    )
    print(len(datasets))
    datasets = [ csr_matrix(ds) for ds in datasets ]
    datasets_dimred, genes = process_data(datasets, genes)
    
    labels = np.zeros(n_cells, dtype=int)
    base = 0
    for i, ds in enumerate(datasets):
        labels[base:(base + ds.shape[0])] = i
        base += ds.shape[0]

    embedding = visualize(
        datasets_dimred, labels,
        '{}_{}'.format(METHOD, NAMESPACE), names,
        perplexity=100, n_iter=400
    )

    cell_labels = (
        open('data/cell_labels/{}_cluster.txt'.format(NAMESPACE))
        .read().rstrip().split()
    )
    le = LabelEncoder().fit(cell_labels)
    cell_labels = le.transform(cell_labels)
    cell_types = le.classes_

    visualize(datasets_dimred,
              cell_labels, NAMESPACE + '_type', cell_types,
              embedding=embedding)
    
