library(methods)
library(scran)

names = list(
    "data/pbmc/10x/68k_pbmc_table.txt",
    "data/pbmc/10x/b_cells_table.txt",
    "data/pbmc/10x/cd14_monocytes_table.txt",
    "data/pbmc/10x/cd4_t_helper_table.txt",
    "data/pbmc/10x/cd56_nk_table.txt",
    "data/pbmc/10x/cytotoxic_t_table.txt",
    "data/pbmc/10x/memory_t_table.txt",
    "data/pbmc/10x/regulatory_t_table.txt",
    "data/pbmc/pbmc_kang_table.txt",
    "data/pbmc/pbmc_10X_table.txt"
    #"data/pancreas/pancreas_inDrop_table.txt",
    #"data/pancreas/pancreas_multi_celseq2_expression_matrix_table.txt",
    #"data/pancreas/pancreas_multi_celseq_expression_matrix_table.txt",
    #"data/pancreas/pancreas_multi_fluidigmc1_expression_matrix_table.txt",
    #"data/pancreas/pancreas_multi_smartseq2_expression_matrix_table.txt"
)

data.tables <- list()
for (i in 1:length(names)) {
    data.tables[[i]] <- as.matrix(
        read.table(names[[i]], sep="\t")
    )
}

print("Done loading")

ptm <- proc.time()

Xmnn <- mnnCorrect(
    data.tables[[1]],
    data.tables[[2]],
    data.tables[[3]],
    data.tables[[4]],
    data.tables[[5]],
    data.tables[[6]],
    data.tables[[7]],
    data.tables[[8]],
    data.tables[[9]],
    data.tables[[10]]
    #data.tables[[11]],
    #data.tables[[12]],
    #data.tables[[13]],
    #data.tables[[14]],
    #data.tables[[15]],
    #data.tables[[16]],
    #data.tables[[17]],
    #data.tables[[18]],
    #data.tables[[19]],
    #data.tables[[20]],
    #data.tables[[21]],
    #data.tables[[22]],
    #data.tables[[23]],
    #data.tables[[24]],
    #data.tables[[25]],
    #data.tables[[26]]
)

print("Done correcting")

print(proc.time() - ptm)

corrected.df <- do.call(cbind.data.frame, Xmnn$corrected)
corrected.mat <- as.matrix(t(corrected.df))

write.table(corrected.mat, file = "/scratch1/brianhie/data/mnn_corrected_pancreas.txt",
            quote = FALSE, sep = "\t")
