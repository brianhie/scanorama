library(methods)
library(splatter)

getLNormFactors <- function(n.facs, sel.prob, neg.prob, fac.loc, fac.scale) {

    is.selected <- as.logical(rbinom(n.facs, 1, sel.prob))
    n.selected <- sum(is.selected)
    dir.selected <- (-1) ^ rbinom(n.selected, 1, neg.prob)
    facs.selected <- rlnorm(n.selected, fac.loc, fac.scale)
    # Reverse directions for factors that are less than one
    dir.selected[facs.selected < 1] <- -1 * dir.selected[facs.selected < 1]
    factors <- rep(1, n.facs)
    factors[is.selected] <- facs.selected ^ dir.selected

    return(factors)
}

splatSimBatchEffects <- function(sim, params) {
    nGenes <- getParam(params, "nGenes")
    nBatches <- getParam(params, "nBatches")
    batch.facLoc <- getParam(params, "batch.facLoc")
    batch.facScale <- getParam(params, "batch.facScale")
    means.gene <- rowData(sim)$GeneMean

    for (idx in seq_len(nBatches)) {
        batch.facs <- getLNormFactors(nGenes, 1, 0.5, batch.facLoc[idx],
                                        batch.facScale[idx])
        batch.means.gene <- means.gene * batch.facs
        rowData(sim)[[paste0("BatchFacBatch", idx)]] <- batch.facs
    }

    return(sim)
}

sim.groups.A <- splatSimulate(
    batchCells = c(1000), group.prob = c(0.50, 0.50, 0, 0),
    method = "groups", verbose = FALSE
)
write.table(add.Gaussian.noise(counts(sim.groups.A)), 'simulate_nonoverlap_A.txt')

sim.groups.B <- splatSimulate(
    batchCells = c(1000), group.prob = c(0, 0, 0.50, 0.50),
    method = "groups", verbose = FALSE
)
sim.groups.A <- splatSimBatchEffects(sim.groups.A, newSplatParams())

sim.groups.C <- splatSimulate(
    batchCells = c(1000), group.prob = c(0, 0.50, 0.50, 0),
    method = "groups", verbose = FALSE
)
sim.groups.A <- splatSimBatchEffects(sim.groups.A, newSplatParams())

write.table(counts(sim.groups.B), 'simulate_nonoverlap_B.txt')
write.table(counts(sim.groups.C), 'simulate_nonoverlap_C.txt')
