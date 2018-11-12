library(reticulate)

names = list(
    "data/pancreas/pancreas_multi_celseq2_expression_matrix_table.txt",
    "data/pancreas/pancreas_multi_celseq_expression_matrix_table.txt"
)

datasets <- list()
genes_list <- list()
for (i in 1:length(names)) {
    datasets[[i]] <- t(as.matrix(
        read.table(names[[i]], sep="\t")
    ))
    genes_list[[i]] <- colnames(datasets[[i]])
}

print("Done loading")

ptm <- proc.time()

scanorama <- import('scanorama')

# Integration.
integrated.data <- scanorama$integrate(datasets, genes_list)

print("Done integrating")

print(proc.time() - ptm)
