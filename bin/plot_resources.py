from sklearn.linear_model import LinearRegression
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

n_cells = np.reshape(np.array([
    10547,
    26369,
    52738,
    105476
]), (4, 1))

pano_memory = np.reshape(np.array([
    1.1,
    2.8,
    6.1,
    12.9
]), (4, 1))

cca_memory = np.reshape(np.array([
    5.0,
    12.9,
    26.1,
    54.2
]), (4, 1))

mnn_memory = np.reshape(np.array([
    5.1,
    13.1,
    25.7,
    49.5
]), (4, 1))

pano_runtime = np.reshape(np.array([
    40.9,
    110.1,
    204.8,
    469.6,
]) / 3600., (4, 1))

cca_runtime = np.reshape(np.array([
    9724.1,
    24622.1,
    49243.7,
    99683.4,
]) / 3600., (4, 1))

mnn_runtime = np.reshape(np.array([
    15669.2,
    39899.7,
    78677.3,
    157212.6
]) / 3600., (4, 1))

line_x = np.array(range(n_cells[-1]), dtype=int)
line_x = line_x.reshape(-1, 1)

# Memory plot.
plt.figure()
plt.plot(line_x, LinearRegression().fit(n_cells, pano_memory)
         .predict(line_x), 'k')
pano = plt.scatter(n_cells, pano_memory, marker='o')
plt.plot(line_x, LinearRegression().fit(n_cells, cca_memory)
         .predict(line_x), 'k')
cca = plt.scatter(n_cells, cca_memory, marker='^')
plt.plot(line_x, LinearRegression().fit(n_cells, mnn_memory)
         .predict(line_x), 'k')
mnn = plt.scatter(n_cells, mnn_memory, marker='s')
plt.legend((pano, cca, mnn),
           ('Scanorama', 'Seurat CCA', 'scran MNN'))
plt.xlabel('Number of cells')
plt.ylabel('Memory (GB)')
plt.savefig('benchmark_memory.svg')

# Memory plot.
plt.figure()
plt.plot(line_x, LinearRegression().fit(n_cells, pano_runtime)
         .predict(line_x), 'k')
pano = plt.scatter(n_cells, pano_runtime, marker='o')
plt.plot(line_x, LinearRegression().fit(n_cells, cca_runtime)
         .predict(line_x), 'k')
cca = plt.scatter(n_cells, cca_runtime, marker='^')
plt.plot(line_x, LinearRegression().fit(n_cells, mnn_runtime)
         .predict(line_x), 'k')
mnn = plt.scatter(n_cells, mnn_runtime, marker='s')
plt.legend((pano, cca, mnn),
           ('Scanorama', 'Seurat CCA', 'scran MNN'))
plt.yscale('log')
plt.xlabel('Number of cells')
plt.ylabel('Runtime (hours)')
plt.savefig('benchmark_runtime.svg')
