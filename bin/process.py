import gzip
import numpy as np
import os.path
from sklearn.preprocessing import normalize
import sys

MIN_TRANSCRIPTS = 600

def load_tab(fname, max_genes=40000):
    if fname.endswith('.gz'):
        opener = gzip.open
    else:
        opener = open
        
    with opener(fname, 'r') as f:
        if fname.endswith('.gz'):
            header = f.readline().decode('utf-8').rstrip().split('\t')
        else:
            header = f.readline().rstrip().split('\t')
            
        cells = header[1:]
        X = np.zeros((len(cells), max_genes))
        genes = []
        for i, line in enumerate(f):
            if i > max_genes:
                break
            if fname.endswith('.gz'):
                line = line.decode('utf-8')
            fields = line.rstrip().split('\t')
            genes.append(fields[0])
            X[:, i] = [ float(f) for f in fields[1:] ]
    return X[:, range(len(genes))], np.array(cells), np.array(genes)

def load_mtx(dname):
    with open(dname + '/matrix.mtx', 'r') as f:
        f.readline(); f.readline()
        header = f.readline().rstrip().split()
        n_genes, n_cells = int(header[0]), int(header[1])

        X = np.zeros((n_cells, n_genes))
        for line in f:
            fields = line.rstrip().split()
            X[int(fields[1])-1, int(fields[0])-1] = float(fields[2])

    genes = []
    with open(dname + '/genes.tsv', 'r') as f:
        for line in f:
            fields = line.rstrip().split()
            genes.append(fields[1].upper())
    assert(len(genes) == n_genes)

    return X, np.array(genes)

def process_tab(fname, min_trans=MIN_TRANSCRIPTS):
    X, cells, genes = load_tab(fname)

    gt_idx = [ i for i, s in enumerate(np.sum(X != 0, axis=1))
               if s >= min_trans ]
    X = X[gt_idx, :]
    cells = cells[gt_idx]

    if fname.endswith('.txt'):
        cache_prefix = '.'.join(fname.split('.')[:-1])
    elif fname.endswith('.txt.gz'):
        cache_prefix = '.'.join(fname.split('.')[:-2])
    
    cache_fname = cache_prefix + '.npz'
    np.savez(cache_fname, X=X, genes=genes)

    return X, cells, genes

def process_mtx(dname, min_trans=MIN_TRANSCRIPTS):
    X, genes = load_mtx(dname)

    gt_idx = [ i for i, s in enumerate(np.sum(X != 0, axis=1))
               if s >= min_trans ]
    X = X[gt_idx, :]
    
    cache_fname = dname + '/tab.npz'
    np.savez(cache_fname, X=X, genes=genes)

    return X, genes

def load_data(name):
    if os.path.isfile(name + '.npz'):
        data = np.load(name + '.npz')
        X = data['X']
        genes = data['genes']
    elif os.path.isfile(name + '/tab.npz'):
        data = np.load(name + '/tab.npz')
        X = data['X']
        genes = data['genes']
    else:
        sys.stderr.write('Could not find: {}\n'.format(name))
        exit(1)
    return X, genes

def load_names(data_names, norm=True, log1p=False, verbose=True):
    # Load datasets.
    datasets = []
    genes_list = []
    n_cells = 0
    for name in data_names:
        X_i, genes_i = load_data(name)
        if norm:
            X_i = normalize(X_i, axis=1)
        if log1p:
            X_i = np.log1p(X_i)
        datasets.append(X_i)
        genes_list.append(genes_i)
        n_cells += X_i.shape[0]
        if verbose:
            print('Loaded {} with {} genes and {} cells'.
                  format(name, X_i.shape[1], X_i.shape[0]))
    if verbose:
        print('Found {} cells among all datasets'
              .format(n_cells))

    return datasets, genes_list, n_cells

def merge_datasets(datasets, genes, verbose=True):
    # Find genes in common.
    keep_genes = set()
    for gene_list in genes:
        if len(keep_genes) == 0:
            keep_genes = set(gene_list)
        else:
            keep_genes &= set(gene_list)
    if verbose:
        print('Found {} genes among all datasets'
              .format(len(keep_genes)))

    # Only keep genes in common.
    ret_datasets = []
    ret_genes = np.array(sorted(keep_genes))
    for i in range(len(datasets)):
        # Remove duplicate genes.
        uniq_genes, uniq_idx = np.unique(genes[i], return_index=True)
        ret_datasets.append(datasets[i][:, uniq_idx])

        # Do gene filtering.
        gene_sort_idx = np.argsort(uniq_genes)
        gene_idx = [ idx for idx in gene_sort_idx
                     if uniq_genes[idx] in keep_genes ]
        ret_datasets[i] = ret_datasets[i][:, gene_idx]
        assert(np.array_equal(uniq_genes[gene_idx], ret_genes))

    return ret_datasets, ret_genes

def save_datasets(datasets, genes, data_names, verbose=True,
                  truncate_neg=False):
    for i in range(len(datasets)):
        dataset = datasets[i]
        name = data_names[i]

        if truncate_neg:
            dataset[dataset < 0] = 0

        with open(name + '.scanorama_corrected.txt', 'w') as of:
            # Save header.
            of.write('Genes\t')
            of.write('\t'.join(
                [ 'cell' + str(cell) for cell in range(dataset.shape[0]) ]
            ) + '\n')

            for g in range(dataset.shape[1]):
                of.write(genes[g] + '\t')
                of.write('\t'.join(
                    [ str(expr) for expr in dataset[:, g] ]
                ) + '\n')

if __name__ == '__main__':
    from config import data_names

    for name in data_names:
        if os.path.isdir(name):
            process_mtx(name)
        elif os.path.isfile(name):
            process_tab(name)
        elif os.path.isfile(name + '.txt'):
            process_tab(name + '.txt')
        elif os.path.isfile(name + '.txt.gz'):
            process_tab(name + '.txt.gz')
        else:
            sys.stderr.write('Warning: Could not find {}\n'.format(name))
        print('Successfully processed {}'.format(name))
