import fileinput
from scanorama import plt
plt.rcParams.update({'font.size': 22})
import sys

methods = [
    'scanorama',
    'mnn',
    'seurat',
    'uncorrected'
]
data = {}

curr_method = 'scanorama'
ks = []
entropies = []

f = open(sys.argv[1])

for line in fileinput.input():
    line = line.rstrip()
    if line in methods:
        if len(ks) > 0:
            data[curr_method] = entropies
            ks = []
            entropies = []
        curr_method = line
        continue
    
    fields = line.split()

    k = int(fields[2].rstrip(','))
    if k < 10:
        continue
    
    ks.append(k)
    entropies.append(float(fields[-1]))
    
data[curr_method] = entropies

plt.figure()

for method in methods:
    plt.plot(ks, data[method], label=method)
    plt.scatter(ks, data[method])

plt.legend()
plt.xlabel('k-means, number of clusters')
plt.ylabel('Normalized Shannon entropy')
plt.savefig('entropies.svg')
