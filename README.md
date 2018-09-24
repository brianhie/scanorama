# Scanorama

## Overview

Scanorama enables batch-correction and integration of heterogeneous scRNA-seq datasets, which is described in the paper ["Panoramic stitching of single-cell transcriptomic data"](https://www.biorxiv.org/content/early/2018/07/17/371179) by Brian Hie, Bryan Bryson, and Bonnie Berger. This repository contains the Scanorama source code as well as scripts necessary for reproducing the results in the paper.

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

### Dataset download

Data for the 26 datasets in our study can be downloaded from http://scanorama.csail.mit.edu/data.tar.gz. Download and unpack this data with the command:

```
wget http://scanorama.csail.mit.edu/data.tar.gz
tar xvf data.tar.gz
```

### Data processing

Scanorama is able to process two file formats. The first is a tab-delimited table format where the columns correspond to cells and the rows correspond to genes. A sample file looks something like:
```
gene	cell_a	cell_b
gene_1	10	10
gene_2	20	20
```
The second is a sparse matrix format used by 10X Genomics (example [here](http://cf.10xgenomics.com/samples/cell-exp/1.1.0/293t/293t_filtered_gene_bc_matrices.tar.gz)). This format has a directory where one file has a list of gene names (`genes.tsv`), one file has a list of cells (`barcodes.tsv`), and one file has a list of the nonzero transcript counts at certain gene/cell coordinates (`matrix.mtx`).

To ensure a consistent data format, Scanorama first processes these raw files and saves them in numpy archive files. To generate these files, run the command:
```
python bin/process.py conf/panorama.txt
```
The corresponding `.npz` files will be saved in the `data/` directory.

New files can be processed by feeding them into `bin/process.py` via the command line or a configuration file, or by modifying the `data_names` variables at the top of `bin/config.py`.

Currently, Scanorama uses a relatively low-level data representation (for Python at least) where a list of numpy arrays holds the gene expression values for each data set.

### Panorama stitching

#### Toy data sets

For a good illustration of how Scanorama works, we can integrate three toy datasets: 293T cells, Jurkat cells, and a 50:50 293T:Jurkat mixture. To integrate these datasets, run:
```
python bin/293t_jurkat.py
```
By default, this prints a log reporting the alignments the algorithm has found between datasets and saves visualization images to a file in the repository's top-level directory.

#### 26 datasets

We can also stitch a much larger number of cells from many more datsets. To do this, run:
```
python bin/panorama.py conf/panorama.txt
```
The collection of datasets to be integrated is specified in the config file `conf/panorama.txt`. Default parameters are listed at the top of `bin/scanorama.py`.

By default, this script will output a verbose log as it finds alignments and applies batch correction. At the end, it will automatically save t-SNE visualized images of the integrated result. The numpy matrices containing the batch-corrected data sets are also available (in memory) to integrate with other single cell pipelines and packages.

#### Runtime performance and memory requirements

Aligning and batch-correcting 26 datasets should complete in under 30 minutes with the process running on 10 cores.

Note that the gradient descent portion of the t-SNE visualization step can take a very long time (a few hours) on more than 100k cells. Other methods for accelerating t-SNE could be used in place of the t-SNE implementation used in this pipeline, such as a faster C++ implementation of [t-SNE](https://github.com/lvdmaaten/bhtsne) or [net-SNE](https://github.com/hhcho/netsne), a version of t-SNE that uses a neural network to reduce the time required for the gradient descent optimization procedure.

Also, note that with the current implementation, the memory usage can be potentially very high, currently processing all 26 datasets averages around 30 GB, peaking at around 50 GB.

### Additional analyses

Scripts for performing additional analyses of the data are also available in the `bin/` directory.

A plethora of other examples can be found in:
```
python bin/pancreas.py
python bin/pbmc.py
python bin/hsc.py
python bin/macrophage.py
```

The script `bin/simulation.py` tests the integrative performance of the method on simulated data.

#### Scanorama implementation

For those interested in the algorithm implementation, `scanorama/scanorama.py` is the main file that handles the mutual nearest neighbors-based matching, batch correction, and panorama assembly.

## Questions

For questions about the pipeline and code, contact brianhie@mit.edu. We will do our best to provide support, address any issues, and keep improving this software. And do not hesitate to submit a pull request and contribute!