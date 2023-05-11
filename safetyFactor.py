#safetyFactor.py
#Description:   calculates safety factor
#Engineer:      T Looby
#Date:          20230209

import numpy as np
import sys
import os
import shutil
from scipy.interpolate import interp1d

EFITPath = '/home/tlooby/source'
HEATPath = '/home/tlooby/source/HEAT/github/source'
sys.path.append(EFITPath)
sys.path.append(HEATPath)
import MHDClass


#gFile = '/home/tlooby/projects/VDEs/g000001.00000'
#gFile = '/home/tlooby/Downloads/g000001.00000'
gFile = '/home/tlooby/projects/VDEs/GEQDSKs_wide_120ms/g000001.01010'
psiN = 0.95 #95% flux surface
#contactPoint = [1.893376, 0.995052]
#contactPoint = [2.044, 0.6]
#contactPoint = [2.432538, 0.0]



MHD = MHDClass.setupForTerminalUse(gFile=gFile)
ep = MHD.ep


Bt0 = ep.g['Bt0']
R0 = ep.g['RmAxis']
Ip = ep.g['Ip'] / 1e6


#LCFS values
lcfs = ep.g['lcfs']
#surf = ep.getBs_FluxSur(1.0)
#Rs = surf['Rs'][~np.isnan(surf['Rs'])]
#Zs = surf['Rs'][~np.isnan(surf['Zs'])]
#lcfs = np.vstack((Rs, Zs)).T
Rmax = np.max(lcfs[:,0])
idxRmax = np.argmax(lcfs[:,0])
maxSep = lcfs[idxRmax]
Rmin = np.min(lcfs[:,0])
Zmax = np.max(lcfs[:,1])
Zmin = np.min(lcfs[:,1])
Rgeo = (Rmax + Rmin)/2.0
a = Rmax - ep.g['RmAxis']
idx = np.argmax(lcfs[:,1])
R_upper = lcfs[idx,0]
idx = np.argmin(lcfs[:,1])
R_lower = lcfs[idx,0]
b = Zmax / a


#ellipticity
k = (Zmax - Zmin)/(2.0*a)

#triangularity
d_upper = (Rgeo - R_upper) / a
d_lower = (Rgeo - R_lower) / a
d = (d_upper + d_lower) / 2.0

print("Elongation at LCFS: {:F}".format(k))
print("Triangularity at LCFS: {:f}".format(d))

Bp = ep.BpFunc(maxSep[0], maxSep[1])[0][0]
Bt = ep.BtFunc(maxSep[0], maxSep[1])[0][0]


term2 = ( 1 + k**2*(1 + 2*d**2 - 1.2*d**3) ) / 2.0
q_uckan = (5 * a**2 *Bt0) / (R0 * Ip) * term2
print("Uckan Safety Factor using k,d at LCFS: {:f}".format(q_uckan))
eps = a / R0
q95 = q_uckan * ( (1.17-0.65*eps)/(1-eps**2)**2 )
print("Uckan q95 from LCFS: {:f}".format(q95))

#95% flux surface
axis = np.array([ep.g['RmAxis'],ep.g['ZmAxis']])
vec = maxSep - axis
RZ = vec*psiN + axis
a = np.linalg.norm(vec)
Bp = ep.BpFunc(RZ[0], RZ[1])[0][0]
Bt = ep.BtFunc(RZ[0], RZ[1])[0][0]

##simple TEST
#k=1.97
#d = 0.54
#Ip = 8.7
#Bt0 = 12.2
#R0 = 1.85
#a = 0.57
#q95 = np.abs( (a * Bt0) / (R0 * Bp) )
#print("Simple safety Factor q95: {:f}".format(q95))


#95% flux surface
surf = ep.getBs_FluxSur(psiN)
Ridx = ~np.isnan(surf['Rs'])
Zidx = ~np.isnan(surf['Zs'])
idxs = np.logical_and(Ridx,Zidx)
Rs = surf['Rs'][idxs]
Zs = surf['Zs'][idxs]
lcfs = np.vstack((Rs, Zs)).T
#print(lcfs)
#print(surf['Rs'])


Rmax = np.max(lcfs[:,0])
idxRmax = np.argmax(lcfs[:,0])
maxSep = lcfs[idxRmax]
Rmin = np.min(lcfs[:,0])
Zmax = np.max(lcfs[:,1])
Zmin = np.min(lcfs[:,1])
Rgeo = (Rmax + Rmin)/2.0
a = Rmax - ep.g['RmAxis']
idx = np.argmax(lcfs[:,1])
R_upper = lcfs[idx,0]
idx = np.argmin(lcfs[:,1])
R_lower = lcfs[idx,0]
b = Zmax / a
#ellipticity
k = (Zmax - Zmin)/(2.0*a)
#triangularity
d_upper = (Rgeo - R_upper) / a
d_lower = (Rgeo - R_lower) / a
d = (d_upper + d_lower) / 2.0

Bp = ep.BpFunc(maxSep[0], maxSep[1])[0][0]
Bt = ep.BtFunc(maxSep[0], maxSep[1])[0][0]


print("Elongation at psiN=0.95: {:F}".format(k))
print("Triangularity at psiN=0.95: {:f}".format(d))

term2 = ( 1 + k**2*(1 + 2*d**2 - 1.2*d**3) ) / 2.0
q_uckan = (5 * a**2 *Bt0) / (R0 * Ip) * term2
print("Uckan Safety Factor using k,d at psiN=0.95: {:f}".format(q_uckan))
eps = a / R0
q95 = q_uckan * ( (1.17-0.65*eps)/(1-eps**2)**2 )
print("Uckan q95: {:f}".format(q95))

