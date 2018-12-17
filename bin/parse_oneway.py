import numpy as np
from scanorama import plt
plt.rcParams.update({'font.size': 25})
import sys

scano, mnn, uncor = {}, {}, {}

in_scano = True
in_mnn = False
for line in open(sys.argv[1]):
    fields = line.rstrip().split()
    if len(fields) > 3 or len(fields) < 2:
        continue
    try:
        F = float(fields[1])
    except ValueError:
        continue

    gene = fields[0]

    if gene.startswith("b'"):
        gene = gene[2:].replace("'", '')

    if in_scano:
        scano[gene] = F
    elif in_mnn:
        mnn[gene] = F
    else:
        uncor[gene] = F

    if gene == 'ZZZ3':
        if in_scano:
            in_scano = False
            in_mnn = True
        elif in_mnn:
            in_mnn = False

# Scanorama.

scanorama, uncorrected = [], []
for gene in set(scano.keys()) & set(uncor.keys()):
    scanorama.append(scano[gene])
    uncorrected.append(uncor[gene])
scanorama = np.array(scanorama)
uncorrected = np.array(uncorrected)

below = sum(scanorama > uncorrected + 50)
above = sum(scanorama < uncorrected - 50)

print('{}% above line'.format(float(above) / float(above + below) * 100))
    
name = sys.argv[1].split('.')[0]
line = max(min(max(scanorama), max(uncorrected)), 2100)

from scipy.stats import pearsonr
print(pearsonr(scanorama, uncorrected))
        
plt.figure()
plt.scatter(scanorama, uncorrected, s=10)
plt.plot([0, line], [0, line], 'r--')
plt.xlim([ 0, 2100 ])
plt.tight_layout()
plt.savefig('oneway_scanorama.png')

# scran MNN.

scranmnn, uncorrected = [], []
for gene in set(mnn.keys()) & set(uncor.keys()):
    scranmnn.append(mnn[gene])
    uncorrected.append(uncor[gene])
scranmnn = np.array(scranmnn)
uncorrected = np.array(uncorrected)

below = sum(scranmnn > uncorrected + 50)
above = sum(scranmnn < uncorrected - 50)

print('{}% above line'.format(float(above) / float(above + below) * 100))
    
name = sys.argv[1].split('.')[0]
line = max(min(max(scranmnn), max(uncorrected)), 2100)

from scipy.stats import pearsonr
print(pearsonr(scranmnn, uncorrected))
        
plt.figure()
plt.scatter(scranmnn, uncorrected, s=10)
plt.plot([0, line], [0, line], 'r--')
plt.xlim([ 0, 2100 ])
plt.tight_layout()
plt.savefig('oneway_mnn.png')

