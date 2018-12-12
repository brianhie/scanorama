import numpy as np
from scanorama import plt
plt.rcParams.update({'font.size': 25})
import sys

scano, uncor = {}, {}

in_scano = True
for line in open(sys.argv[1]):
    fields = line.rstrip().split()
    if len(fields) > 3:
        continue
    try:
        F = float(fields[1])
    except ValueError:
        continue

    if in_scano:
        scano[fields[0]] = F
    else:
        uncor[fields[0]] = F

    if fields[0] == 'ZZZ3':
        in_scano = False

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
        
plt.figure()
plt.scatter(scanorama, uncorrected, s=10)
plt.plot([0, line], [0, line], 'r--')
plt.xlim([ 0, 2100 ])
plt.tight_layout()
plt.savefig('oneway_{}.png'.format(name))
