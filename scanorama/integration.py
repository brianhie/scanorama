from ample import gs, uniform, srs, kmeanspp
import numpy as np
from subprocess import Popen
from time import time

from .utils import mkdir_p

def harmony_integrate(datasets_dimred, verbose=1):
    mkdir_p('data/harmony')

    n_samples = sum([ ds.shape[0] for ds in datasets_dimred ])

    embed_fname = 'data/harmony/embedding.txt'
    label_fname = 'data/harmony/labels.txt'
    
    np.savetxt(embed_fname, np.concatenate(datasets_dimred))

    labels = []
    curr_label = 0
    for i, a in enumerate(datasets_dimred):
        labels += list(np.zeros(a.shape[0]) + curr_label)
        curr_label += 1
    labels = np.array(labels, dtype=int)

    np.savetxt(label_fname, labels)

    if verbose:
        print('Integrating with harmony...')

    rcode = Popen('Rscript R/harmony.R {} {} > harmony.log 2>&1'
                  .format(embed_fname, label_fname), shell=True).wait()
    if rcode != 0:
        sys.stderr.write('ERROR: subprocess returned error code {}\n'
                         .format(rcode))
        exit(rcode)

    if verbose:
        print('Done with harmony integration')

    integrated = np.loadtxt('data/harmony/integrated.txt')

    assert(n_samples == integrated.shape[0])
        
    datasets_return = []
    base = 0
    for ds in datasets_dimred:
        datasets_return.append(integrated[base:(base + ds.shape[0])])
        base += ds.shape[0]
        
    return datasets_return
