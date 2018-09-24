library(methods)
library(Seurat)

names = list(
    "../../assemble-sc/data/brain/neuron_9k_table.txt",
    "../../assemble-sc/data/macrophage/uninfected_table.txt",
    "../../assemble-sc/data/marrow/marrow_mars_table.txt"
)

data.tables <- list()
for (i in 1:length(names)) {
    data.tables[[i]] <- as(as.matrix(
        read.table(names[[i]])
    ), "dgCMatrix")
}

ob.list <- list()
for (i in 1:length(names)) {
    ob.list[[i]] <- CreateSeuratObject(raw.data = data.tables[[i]])
    ob.list[[i]] <- NormalizeData(ob.list[[i]])
    ob.list[[i]] <- FilterCells(ob.list[[i]], subset.names = "nGene", low.thresholds = 1)
    ob.list[[i]] <- FindVariableGenes(ob.list[[i]], do.plot = F, display.progress = F)
    ob.list[[i]] <- ScaleData(ob.list[[i]])
    ob.list[[i]]@meta.data$tech <- toString(i)
}

print("Done loading")

genes.use <- c()
for (i in 1:length(ob.list)) {
  genes.use <- c(genes.use, head(rownames(ob.list[[i]]@hvg.info), 100000))
}
genes.use <- names(which(table(genes.use) > 1))
for (i in 1:length(ob.list)) {
  genes.use <- genes.use[genes.use %in% rownames(ob.list[[i]]@scale.data)]
}

print("Done intersecting genes")

ptm <- proc.time()

integrated <- RunMultiCCA(ob.list, genes.use = genes.use, num.ccs = 15)

print("CCA Done")

# CC Selection
MetageneBicorPlot(integrated, grouping.var = "tech", dims.eval = 1:15)

# Run rare non-overlapping filtering
integrated <- CalcVarExpRatio(object = integrated, reduction.type = "pca",
                                       grouping.var = "tech", dims.use = 1:15)
#integrated <- SubsetData(integrated, subset.name = "var.ratio.pca",
#                         accept.low = 0.5)

# Alignment
integrated <- AlignSubspace(integrated,
                            reduction.type = "cca",
                            grouping.var = "tech",
                            dims.align = 1:15)

print(proc.time() - ptm)

write.table(integrated@dr$cca@cell.embeddings,
            file = "../data/corrected_seurat_different3.txt",
            quote = FALSE, sep = "\t")
