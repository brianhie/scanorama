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
    'data/pancreas/pancreas_inDrop',
    'data/pancreas/pancreas_multi_celseq2_expression_matrix',
    'data/pancreas/pancreas_multi_celseq_expression_matrix',
    'data/pancreas/pancreas_multi_fluidigmc1_expression_matrix',
    'data/pancreas/pancreas_multi_smartseq2_expression_matrix',
    'data/pbmc/10x/68k_pbmc',
    'data/pbmc/10x/b_cells',
    'data/pbmc/10x/cd14_monocytes',
    'data/pbmc/10x/cd4_t_helper',
    'data/pbmc/10x/cd56_nk',
    'data/pbmc/10x/cytotoxic_t',
    'data/pbmc/10x/memory_t',
    'data/pbmc/10x/regulatory_t',
    'data/pbmc/pbmc_kang',
    'data/pbmc/pbmc_10X',
]

if __name__ == '__main__':
    # Load raw data from files.
    datasets_full, genes_list, n_cells = load_names(data_names)
    datasets, genes = merge_datasets(datasets_full, genes_list)

    for i in range(len(datasets)):
        print('Writing {}'.format(data_names[i]))
        write_table(datasets[i].toarray(), genes, data_names[i])
