import numpy as np

import sys
EFITPath = '/home/tom/source'
HEATPath = '/home/tom/source/HEAT/github/source'
sys.path.append(EFITPath)
sys.path.append(HEATPath)
import MHDClass

gFilePath = '/home/tom/HEATtest/NSTXU/limiter_GyroTerminal/nstx/g204118.00113_newLCFS_3220mm'
MHD = MHDClass.setupForTerminalUse(gFile=gFilePath)

Rlim = MHD.ep.g['lcfs'][:,0]
Zlim = MHD.ep.g['lcfs'][:,1]

#calculate cross sectional area using shoelace formula
i=np.arange(len(Rlim))
crossSecArea=np.abs(np.sum(Rlim[i-1]*Zlim[i]-Rlim[i]*Zlim[i-1])*0.5)

#calculate volume
vol = crossSecArea * 2 * np.pi * MHD.ep.g['RmAxis']

ep = MHD.ep
#assuming plasma is centered in machine here
zMin = ep.g['ZmAxis'] - 0.25
zMax = ep.g['ZmAxis'] + 0.25
rLCFS = ep.g['lcfs'][:,1]
zLCFS = ep.g['lcfs'][:,1]
#this prevents us from getting locations not at midplane
idx = np.where(np.logical_and(zLCFS>zMin,zLCFS<zMax))
Rmax = ep.g['lcfs'][:,0][idx].max()
Rmin = ep.g['lcfs'][:,0][idx].min()
# geometric quantities
Rgeo = (Rmax + Rmin) / 2.0
a = (Rmax - Rmin) / 2.0
aspect = a/Rgeo

#maximum z point and elongation
idx2 = np.where(np.logical_and(rLCFS>Rmin,rLCFS<Rmax))
b = ep.g['lcfs'][:,1][idx].max() #assumes equatorial plane is z=0
k = b / a
#lambda q from Horacek engineering scaling figure 6a
Ptot = 10e6
lqCF = 10 * (Ptot / vol)**(-0.38) * aspect**(1.3) * k**(-1.3)

print(lqCF)
