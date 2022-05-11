#script for printing various quantities related to the gFile.  can
#be used to test if gFile psiRZ is normalized by using Ampere's Law

import sys
from scipy import interpolate
import numpy as np
EFITPath = '/home/tom/source'
HEATPath = '/home/tom/source/HEAT/github/source'
sys.path.append(EFITPath)
sys.path.append(HEATPath)
import MHDClass

gFilePath = '/home/tom/HEATtest/STEP/darioHEATruns/g000001.00001'
ep = MHDClass.setupForTerminalUse(gFile=gFilePath).ep


#R=0.621875
R=3.98
Z = 0.0
Raxis = ep.g['RmAxis']
a = R - Raxis

Bp = ep.BpFunc.ev(R,Z)
Bt = ep.BtFunc.ev(R,Z)
psi1 = ep.psiFunc.ev(R,Z)




Ip = ep.g['Ip']
mu = 4*np.pi*10**-7
Bp1 = mu * Ip / (2*np.pi*a)
print("Bp [T] calculated @R using Ampere's Law:".format(R))
print(Bp1)
print("Bp [T] calculated @R from gfile:")
print(Bp)

print("TEDST")
print(ep.g['Bt0'])
print(ep.g['Fpol'][-1] / R)
print(Bt)
print(ep.g['Bt0']*Raxis/R)


psi = ep.g['psi']
q = ep.g['q']
f = interpolate.interp1d(psi,q)
q1 = f(psi1)
q1_calc = (a * Bt) / (R * Bp)

print("\nq calculated @R from gfile:".format(R))
print(q1)
print("q calculated @R from Ip and Bt:")
print(q1_calc)

print(Raxis)
# For putting in a google slide / Latex:
#Bp = \frac{\mu_0 I_p}{2 \pi a} = \frac{(4 \pi \times 10^{-7} \text{T m/A}) (2 \text{MA}) }{2 \pi (0.12726 \text{m})} = 3.1433 \text{ T}
#q = \frac{aB_t}{RB_p} = \frac{(0.12726 \text{m}) (2.1137 \text{T})}{(0.621875 \text{m})(0.72173 \text{T})} = 0.5993
