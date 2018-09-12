library(methods)
library(splatter)

sim.groups <- splatSimulate(
    batchCells = c(
        1000, 1000, 1000, 1000, 1000,
        1000, 1000, 1000, 1000, 1000
    ),
    group.prob = c(0.99, 0.01), method = "groups", verbose = FALSE
)

write.table(t(counts(sim.groups)), 'simulate_rare_full.txt')
