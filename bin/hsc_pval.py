import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import normalize, LabelEncoder
import sys

from process import load_names
from scanorama import *

NAMESPACE = 'hsc'

data_names = [
    'data/hsc/hsc_mars',
    'data/hsc/hsc_ss2',
]

# Computes the probability that the corrected SS2 dataset
# comes from the original SS2 distribution or from the same
# distribution as the corrected MARS-Seq dataset.
if __name__ == '__main__':
    # Load data.
    datasets, genes_list, n_cells = load_names(data_names, verbose=False)
    datasets, genes = merge_datasets(datasets, genes_list, verbose=False)
    datasets, genes = process_data(datasets, genes)
    datasets = [ normalize(ds, axis=1) for ds in datasets ]
    
    # Fit initial mixture models.
    gm_ss2 = (GaussianMixture(n_components=3, n_init=3)
              .fit(datasets[1]))

    # Do batch correction.
    datasets = assemble(
        datasets,
        verbose=False, knn=KNN, sigma=SIGMA, approx=APPROX
    )
    datasets = [ normalize(ds, axis=1) for ds in datasets ]
    
    # Fit mixture models to other dataset.
    gm_mars_corrected = (
        GaussianMixture(n_components=3, n_init=3)
        .fit(datasets[0])
    )

    # Natural log likelihoods.
    ll_ss2 = gm_ss2.score(datasets[1])
    ll_mars_corrected = gm_mars_corrected.score(datasets[1])

    # Natural log of the likelihood ratio.
    print(ll_ss2 - max(ll_ss2, ll_mars_corrected))
    
