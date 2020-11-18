import scanorama
import numpy as np

def data_gen():
    X1 = np.random.rand(100, 10)
    genes1 = [ 'g' + str(i) for i in range(10) ]

    X2 = np.random.rand(200, 12)
    genes2 = list(reversed([ 'g' + str(i) for i in range(12) ]))

    return [ X1, X2 ], [ genes1, genes2 ]

def test_scanorama_integrate():
    """
    Test that Scanorama integration works.
    Ensures that the function call runs and that the returned
    dimensions match up.
    """

    datasets, genes_list = data_gen()

    integrated, genes = scanorama.integrate(datasets, genes_list)

    for X_int, X_orig in zip(integrated, datasets):
        assert(X_int.shape[0] == X_orig.shape[0])
        assert(X_int.shape[1] == len(genes))

def test_scanorama_integrate_scanpy():
    """
    Test that Scanorama integration with Scanpy AnnData works.
    Ensures that the function call runs and dimensions match.
    """
    from anndata import AnnData
    import pandas as pd

    datasets, genes_list = data_gen()

    adatas = []
    for i in range(len(datasets)):
        adata = AnnData(datasets[i])
        adata.obs = pd.DataFrame(list(range(datasets[i].shape[0])),
                                 columns=[ 'obs1' ])
        adata.var = pd.DataFrame(genes_list[i], columns=[ 'var1' ])
        adatas.append(adata)

    scanorama.integrate_scanpy(adatas)

    for adata, X in zip(adatas, datasets):
        assert(adata.obsm['X_scanorama'].shape[0] == X.shape[0])

def test_scanorama_correct_scanpy():
    """
    Test that Scanorama correction with Scanpy AnnData works.
    Ensures that the function call runs, dimensions match, and
    metadata is in the correct order.
    """
    from anndata import AnnData
    import pandas as pd

    datasets, genes_list = data_gen()

    adatas = []
    for i in range(len(datasets)):
        adata = AnnData(datasets[i])
        adata.obs = pd.DataFrame(list(range(datasets[i].shape[0])),
                                 columns=[ 'obs1' ])
        adata.var = pd.DataFrame(genes_list[i], columns=[ 'var1' ])
        adata.var_names = genes_list[i]
        adatas.append(adata)

    corrected = scanorama.correct_scanpy(adatas)

    for adata_cor, adata_orig in zip(corrected, adatas):
        assert(adata_cor.X.shape[0] == adata_orig.X.shape[0])
        assert(adata_cor.X.shape[1] == adatas[0].X.shape[1])
        assert(list(adata_cor.obs['obs1']) == list(adata_orig.obs['obs1']))
        assert(list(adata_cor.var['var1']) == list(adatas[0].var['var1']))
