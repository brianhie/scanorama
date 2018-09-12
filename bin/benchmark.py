from process import load_names, merge_datasets

def write_table(dataset, genes, name):
    prefix = name.split('/')[-1]
    with open(name + '_table.txt', 'w') as f:
        header = '\t'.join([ prefix + str(i) for i in range(dataset.shape[0]) ])
        f.write(header + '\n')

        for i in range(dataset.shape[1]):
            line = '\t'.join([ str(j) for j in dataset[:, i] ])
            f.write(genes[i] + '\t' + line + '\n')

data_names = [
    'data/simulation/simulate_nonoverlap/simulate_nonoverlap_A',
    'data/simulation/simulate_nonoverlap/simulate_nonoverlap_B',
    'data/simulation/simulate_nonoverlap/simulate_nonoverlap_C',
]

if __name__ == '__main__':
    # Load raw data from files.
    datasets_full, genes_list, n_cells = load_names(data_names)
    datasets, genes = merge_datasets(datasets_full, genes_list)

    for i in range(len(datasets)):
        print('Writing {}'.format(data_names[i]))
        write_table(datasets[i], genes, data_names[i])
