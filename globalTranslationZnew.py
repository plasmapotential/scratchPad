import numpy as np
import sys
import os

EFITPath = '/home/tom/source'
HEATPath = '/home/tom/source/HEAT/github/source'
sys.path.append(EFITPath)
sys.path.append(HEATPath)
import MHDClass



gF = '/home/tom/HEATruns/SPARC/globalPerturbations/sparc/g000001.00001'
MHD = MHDClass.setupForTerminalUse(gF)


#translation [m]
d = 0.005

#[m]
z1 = -1.3214
z2 = -1.51
r1 = 1.5872
r2 = 1.72

z3 = z1
z4 = z2
r3 = d + r1
r4 = d + r2

#original SP location
rSP1 = 1.6745
zSP1 = -1.454

#Bfield at SP1
Br = MHD.ep.BRFunc(rSP1, zSP1)[0][0]
Bz = MHD.ep.BZFunc(rSP1, zSP1)[0][0]

x34 = np.array([r4-r3, z4-z3])
dVec = np.array([d,0])

x34_norm = x34 / np.linalg.norm(x34)
d_norm = dVec / np.linalg.norm(dVec)

alpha = np.arccos(np.dot(x34_norm, d_norm))

Bvec = np.array([Br, -Bz])
BvecNorm = Bvec / np.linalg.norm(Bvec)
print(x34_norm)
beta = np.arccos(np.dot(-BvecNorm, x34_norm))
print(np.degrees(beta))
gamma = np.pi - beta - alpha
a = d * np.sin(alpha) / np.sin(beta)
deltaR = a * np.cos(gamma)*1000.0
deltaZ = a * np.sin(gamma)*1000.0
print(np.degrees(gamma))
print(deltaR)
print(deltaZ)

print(MHD.ep.g['RmAxis'])
