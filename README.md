# Scanorama

## Overview

Scanorama enables batch-correction and integration of heterogeneous scRNA-seq data sets, which is described in the paper ["Panoramic stitching of single-cell transcriptomic data"](https://www.biorxiv.org/content/early/2018/07/17/371179) by Brian Hie, Bryan Bryson, and Bonnie Berger. This repository contains the Scanorama source code as well as scripts necessary for reproducing the results in the paper.

## Example Usage and API

Here is example usage of Scanorama in Python:

```
# List of data sets:
datasets = [ list of scipy.sparse.csr_matrix or numpy.ndarray ]
# List of gene lists:
genes_list = [ list of list of string ]

import scanorama

# Integration.
integrated, genes = scanorama.integrate(datasets, genes_list)

# Batch correction.
corrected, genes = scanorama.correct(datasets, genes_list)

# Integration and batch correction.
integrated, corrected, genes = scanorama.correct(datasets, genes_list, return_dimred=True)
```

Additional parameter documentation for each method is in the Scanorama source code at the top of [`scanorama/scanorama.py`](scanorama/scanorama.py).

There are also wrappers that make it easy to use Scanorama with [scanpy's AnnData object](https://anndata.readthedocs.io/en/latest/anndata.AnnData.html#anndata.AnnData):

```
# List of data sets:
adatas = [ list of scanpy.api.AnnData ]

import scanorama

# Integration.
integrated = scanorama.integrate_scanpy(adatas)

# Batch correction.
corrected = scanorama.correct_scanpy(adatas)

# Integration and batch correction.
integrated, corrected = scanorama.correct_scanpy(adatas, return_dimred=True)
```

You can also call Scanorama from R using the [`reticulate`](https://rstudio.github.io/reticulate/) package:

```
# List of data sets:
datasets <- list( list of matrices )
# List of gene lists:
genes_list <- list( list of list of string )

library(reticulate)
scanorama <- import('scanorama')

# Integration.
integrated.data <- scanorama$integrate(datasets, genes_list, return_dense=TRUE)

# Batch correction.
corrected.data <- scanorama$correct(datasets, genes_list, return_dense=TRUE)

# Integration and batch correction.
integrated.corrected.data <- scanorama$correct(datasets, genes_list,
                                               return_dimred=TRUE, return_dense=TRUE)
```

Note that `reticulate` has trouble returning sparse matrices, so you should set the `return_dense` flag to `TRUE` (which returns the corrected data in R matrices) when attempting to use Scanorama in R.

## Instructions

This repository contains the Scanorama code as well as some (hopefully) helpful examples to get you started. Reading the instructions below and running at least a small toy example is highly recommended!

### Setup

First, download the Scanorama repository with the command:
```
git clone https://github.com/brianhie/scanorama.git
```

Change into the repository directory:
```
cd scanorama/
```

And install Scanorama with the following command
```
python setup.py install --user
```

### Data set download

All of the data used in our study (around 4 GB) can be downloaded from http://scanorama.csail.mit.edu/data.tar.gz. Download and unpack this data with the command:

```
wget http://scanorama.csail.mit.edu/data.tar.gz
tar xvf data.tar.gz
```

A smaller version of the data (around 720 MB), including 26 heterogeneous data sets, can be similarly downloaded from http://scanorama.csail.mit.edu/data_light.tar.gz.

### Data processing

Scanorama is able to process two file formats. The first is a tab-delimited table format where the columns correspond to cells and the rows correspond to genes. A sample file looks something like:
```
gene	cell_a	cell_b
gene_1	10	10
gene_2	20	20
```
The second is a sparse matrix format used by 10X Genomics (example [here](http://cf.10xgenomics.com/samples/cell-exp/1.1.0/293t/293t_filtered_gene_bc_matrices.tar.gz)). This format has a directory where one file has a list of gene names (`genes.tsv`) and one file has a list of the nonzero transcript counts at certain gene/cell coordinates (`matrix.mtx`).

To ensure a consistent data format, Scanorama first processes these raw files and saves them in numpy archive files. To generate these files, run the command:
```
python bin/process.py conf/panorama.txt
```
The corresponding `.npz` files will be saved in the `data/` directory.

New files can be processed by feeding them into `bin/process.py` via the command line or a configuration file, or by modifying the `data_names` variables at the top of `bin/config.py`.

Currently, Scanorama uses a relatively low-level data representation (for Python at least) where a list of numpy arrays holds the gene expression values for each data set.

### Panorama stitching

#### Toy data sets

For a good illustration of how Scanorama works, we can integrate three toy data sets: 293T cells, Jurkat cells, and a 50:50 293T:Jurkat mixture. To integrate these data sets, run:
```
python bin/293t_jurkat.py
```
By default, this prints a log reporting the alignments the algorithm has found between data sets and saves visualization images to a file in the repository's top-level directory.

#### 26 data sets

We can also stitch a much larger number of cells from many more datsets. To do this, run:
```
python bin/panorama.py conf/panorama.txt
```
The collection of data sets to be integrated is specified in the config file `conf/panorama.txt`. Default parameters are listed at the top of `scanorama/scanorama.py`.

By default, this script will output a verbose log as it finds alignments and applies batch correction. At the end, it will automatically save t-SNE visualized images of the integrated result. The numpy matrices containing the batch-corrected data sets are also available (in memory) to integrate with other single cell pipelines and packages.

#### Runtime performance and memory requirements

Aligning and batch-correcting 26 data sets should complete in around 9 minutes with the process running on 10 cores. The memory usage should be under 10 GB.

Note that the gradient descent portion of the t-SNE visualization step can take a very long time (a few hours) and require a lot of memory (around 13 GB) on more than 100k cells. Other methods for accelerating t-SNE could be used in place of the t-SNE implementation used in this pipeline, such as a faster C++ implementation of [t-SNE](https://github.com/lvdmaaten/bhtsne), [Multicore-TSNE](https://github.com/DmitryUlyanov/Multicore-TSNE), or [net-SNE](https://github.com/hhcho/netsne), a version of t-SNE that uses a neural network to reduce the time required for the gradient descent optimization procedure.

### Additional analyses

Scripts for performing additional analyses of the data are also available in the `bin/` directory.

#### Scanorama implementation

For those interested in the algorithm implementation, `scanorama/scanorama.py` is the main file that handles the mutual nearest neighbors-based matching, batch correction, and panorama assembly.

## Questions

For questions about the pipeline and code, contact brianhie@mit.edu. We will do our best to provide support, address any issues, and keep improving this software. And do not hesitate to submit a pull request and contribute!
