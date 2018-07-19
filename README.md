# Scanorama

## Overview

Scanorama enables batch-correction and integration of heterogeneous scRNA-seq datasets, which is described in the paper "Panoramic integration of single-cell transcriptomic data" by Brian Hie, Bryan Bryson, and Bonnie Berger. This repository contains the Scanorama source code as well as scripts necessary for reproducing the results in the paper.

## Instructions

### Setup

First, download the Scanorama repository with the command:
```
git clone https://github.com/brianhie/scanorama.git
```

Change into the repository directory:
```
cd scanorama/
```

And install the required dependencies with the command:
```
python -m pip install -r requirements.txt
```

Scanorama has been tested with Python 2.7 using the following packages:
* [annoy](https://github.com/spotify/annoy) (1.11.5)
* [intervaltree](https://github.com/chaimleib/intervaltree) (2.1.0)
* [numpy](http://www.numpy.org/) (1.11.2)
* [scipy](https://www.scipy.org/) (1.0.0)
* [scikit-learn](http://scikit-learn.org/) (0.19.0)

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
The second is a sparse matrix format used by 10x Genomics (example [here](http://cf.10xgenomics.com/samples/cell-exp/1.1.0/293t/293t_filtered_gene_bc_matrices.tar.gz)). This format has a directory where one file has a list of gene names (`genes.tsv`), one file has a list of cells (`barcodes.tsv`), and one file has a list of the nonzero transcript counts at certain gene/cell coordinates (`matrix.mtx`).

To ensure a consistent data format, Scanorama first processes these raw files and saves them in numpy archive files. To generate these files, run the command:
```
python bin/process.py
```
The corresponding `.npz` files will be saved in the `data/` directory.

New files can be processed by modifying the `data_names` variables at the top of `bin/process.py`.

### Panorama stitching

#### Toy datasets

For a good illustration of how Scanorama works, we can integrate three toy datasets: 293T cells, Jurkat cells, and a 50:50 293T:Jurkat mixture. To integrate these datasets, run:
```
python bin/293t_jurkat.py
```
By default, this prints a log reporting the alignments the algorithm has found between datasets and saves visualization images to a file in the repository's top-level directory.

#### 26 datasets

We can also stitch a much larger number of cells from many more datsets. To do this, run:
```
python bin/panorama.py
```
The collection of datasets to be integrated is specified in the `data_names` variable at the top of `bin/panorama.py`. Changes to the default parameters can also be made to the variables at the top of this file.

By default, this script will output a verbose log as it finds alignments and applies batch correction. At the end, it will automatically save t-SNE visualized images of the integrated result.

#### Runtime performance and memory requirements

Aligning and batch-correcting 26 datasets should complete in under 30 minutes with the process running on 10 cores.

Note that the gradient descent portion of the t-SNE visualization step can take a very long time (a few hours) on more than 100k cells. Other methods for accelerating t-SNE could be used in place of the t-SNE implementation used in this pipeline, such as a faster C++ implementation of [t-SNE](https://github.com/lvdmaaten/bhtsne) or [net-SNE](https://github.com/hhcho/netsne), a version of t-SNE that uses a neural network to reduce the time required for the gradient descent optimization procedure.

Also, note that the peak memory usage required to process all 26 datasets is around 50 GB.

### Additional analyses

Scripts for performing additional analyses of the data are also available in the `bin/` directory.

Examples of analyses within individual groups of cell types can be found at:
```
python bin/pancreas.py
python bin/pbmc.py
python bin/hsc.py
python bin/macrophage.py
```

The script `bin/simulation.py` tests the integrative performance of the method on simulated data.

## Questions

For questions about the pipeline and code, contact brianhie@mit.edu. Do not hesitate to submit a pull request and contribute!