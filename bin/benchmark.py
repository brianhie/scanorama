from process import load_names, merge_datasets

def write_table(dataset, genes, name, cell_name=None):
    if cell_name is not None:
        prefix = cell_name
    else:
        prefix = name.split('/')[-1]
    with open(name + '_table.txt', 'w') as f:
        header = '\t'.join([ prefix + str(i) for i in range(dataset.shape[0]) ])
        f.write(header + '\n')

        for i in range(dataset.shape[1]):
            line = '\t'.join([ str(j) for j in dataset[:, i] ])
            f.write(genes[i] + '\t' + line + '\n')

if __name__ == '__main__':
    from config import data_names
    
    # Load raw data from files.
    datasets_full, genes_list, n_cells = load_names(data_names)
    datasets, genes = merge_datasets(datasets_full, genes_list)

    for i in range(len(datasets)):
        print('Writing {}'.format(data_names[i]))
        write_table(datasets[i], genes, data_names[i])
