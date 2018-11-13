suppressMessages(library(monocle))

# Pseudo-time for uncorrected.

expr.matrix <- as.matrix(read.table("data/macrophage/mono_macro_table.txt"))
sample.sheet <- read.table("data/macrophage/mono_macro_meta.txt")
gene.annotation <- read.delim("data/macrophage/mono_macro_genes.txt")

pd <- new("AnnotatedDataFrame", data = sample.sheet)
fd <- new("AnnotatedDataFrame", data = gene.annotation)
obj <- newCellDataSet(expr.matrix, phenoData = pd, featureData = fd,
                      expressionFamily=tobit())

orderingGenes <- scan("data/macrophage/mono_macro_diffexpr.txt",
                      what = typeof(""), sep = "\n")
obj <- setOrderingFilter(obj, orderingGenes)

obj <- reduceDimension(obj, max_components = 2, method = 'DDRTree')

obj <- orderCells(obj)

pdf('uncorrected.pdf')

plot_cell_trajectory(obj, color_by = "Batch")

# Pseudo-time for corrected.

expr.matrix <- as.matrix(read.table("data/macrophage/mono_macro_corrected_table.txt"))
sample.sheet <- read.table("data/macrophage/mono_macro_meta.txt")
gene.annotation <- read.delim("data/macrophage/mono_macro_genes.txt")

pd <- new("AnnotatedDataFrame", data = sample.sheet)
fd <- new("AnnotatedDataFrame", data = gene.annotation)
obj <- newCellDataSet(expr.matrix, phenoData = pd, featureData = fd,
                      expressionFamily=tobit())

orderingGenes <- scan("data/macrophage/mono_macro_diffexpr.txt",
                      what = typeof(""), sep = "\n")
obj <- setOrderingFilter(obj, orderingGenes)

obj <- reduceDimension(obj, max_components = 2, method = 'DDRTree')

obj <- orderCells(obj)

pdf('corrected.pdf')

plot_cell_trajectory(obj, color_by = "Batch")

# Pseudo-time for scran MNN.

expr.matrix <- as.matrix(read.table("data/macrophage/mono_macro_mnn_corrected_table.txt"))
sample.sheet <- read.table("data/macrophage/mono_macro_meta.txt")
gene.annotation <- read.delim("data/macrophage/mono_macro_genes.txt")

pd <- new("AnnotatedDataFrame", data = sample.sheet)
fd <- new("AnnotatedDataFrame", data = gene.annotation)
obj <- newCellDataSet(expr.matrix, phenoData = pd, featureData = fd,
                      expressionFamily=tobit())

orderingGenes <- scan("data/macrophage/mono_macro_diffexpr.txt",
                      what = typeof(""), sep = "\n")
obj <- setOrderingFilter(obj, orderingGenes)

obj <- reduceDimension(obj, max_components = 2, method = 'DDRTree')

obj <- orderCells(obj)

pdf('mnn_corrected.pdf')

plot_cell_trajectory(obj, color_by = "Batch")
