#matlab2GEQDSK.py
#Description:   Writes gfile from matlab file
#Engineer:      T Looby
#Date:          20191206 (updated 20220619 for CREATE)

import os
import numpy as np
import scipy.io
import scipy.integrate as integ
import scipy.interpolate as interp
import matplotlib.pyplot as plt
import sys
#you need equilParams_class to run this script
EFITpath = '/home/tom/source'
sys.path.append(EFITpath)
import EFIT.equilParams_class as EP
import matplotlib.pyplot as plt

matlab_file = '/home/tom/work/CFS/projects/CREATE/CarMa0NL_Shot10_v6.mat'
gIn = '/home/tom/work/CFS/projects/CREATE/g000001.00010'
gOut = '/home/tom/work/CFS/projects/CREATE/create.geqdsk'
mat = scipy.io.loadmat(matlab_file)
writeMask = True
plotMask = True

#These are all defined in the MATLAB file
R = mat['R']
Z = mat['Z']
r = np.unique(R)
z = np.unique(Z)
timeSIM = mat['timeSIM']
t = 0
psiRZ = mat['psi_tot_grid'][:,:,t]
jRZ = mat['j_zeta_grid'][t]
psiAxis = mat['psi_a'][0,t]
psiSep = mat['psi_b'][0,t]
psiLim = mat['psi_l'][0,t]
BtFlux = mat['phi_tor_pl']
rlim = mat['r_fw']
zlim = mat['z_fw']
haloCurrent = mat['halo_current']
rCS = mat['r_CS']
zCS = mat['z_CS']
Ieq = mat['Ieq']

print(psiRZ)

#these are defined in the PRD FreeGS equilibrium
ep = EP.equilParams(gIn)
Fpol = ep.g['Fpol']
qPsi = ep.g['qpsi']
Bt0 = ep.g['Bt0']

levels = sorted(np.append([0.0,0.05,0.1,0.25,0.5,0.75,0.95, 1.0], np.linspace(0.99,psiRZ.max(),15)))
CS = plt.contourf(R,Z,psiRZ,levels,cmap=plt.cm.cividis)
lcfsCS = plt.contour(CS, levels = [1.0])
for i in range(len(lcfsCS.allsegs[0])):
    rlcfs = lcfsCS.allsegs[0][i][:,0]
    zlcfs = lcfsCS.allsegs[0][i][:,1]
lcfs = np.vstack((rlcfs, zlcfs)).T

#These we derive from what came out of the matlab file and GEQDSK file
wall = np.vstack((rlim, zlim)).T
Nwall = len(rlim)
Nr = len(r)
Nz = len(z)
R1 = np.min(R)
Xdim = np.max(R) - np.min(R)
Zdim = np.max(Z) - np.min(Z)
R0 = Xdim / 2.0
Zmid = Zdim / 2.0
Nlcfs = len(lcfs[0])
psiN1D = np.linspace(psiAxis, psiSep, ep.g['NR'])
psi1D = psiN1D * (psiSep - psiAxis) + psiAxis
#psiN2D = (psiRZ - psiAxis) / (psiSep - psiAxis)
Fprime = np.diff(Fpol) / np.diff(psi1D)

#Interpolate the first point in FFprime
deltaFp = Fprime[0] - Fprime[1]
Fp0 = deltaFp + Fprime[0]
FFprime = np.insert(Fprime,0,Fp0) * Fpol

if plotMask:
    print(FFprime)
    plt.plot(FFprime, label='FFprime')
    plt.plot(psi1D, label='psi')
    plt.plot(Fpol, label='Fpol')
    plt.legend()
    plt.show()



#These we don't have so we just write arbitrary values in
KVTOR = 0
RVTOR = 0
NMASS = 0
RHOVN = np.zeros((Nr))
Ip = ep.g['Ip']
RmAxis = 1.0
ZmAxis = 0.0
Pprime = np.zeros((Nr))
Pres = np.ones((Nr))

# --- _write_array -----------------------
# write numpy array in format used in g-file:
# 5 columns, 9 digit float with exponents and no spaces in front of negative numbers
def write_array(x, f):
    N = len(x)
    rows = int(N/5)  # integer division
    rest = N - 5*rows
    for i in range(rows):
        for j in range(5):
                f.write('% .9E' % (x[i*5 + j]))
        f.write('\n')
    if(rest > 0):
        for j in range(rest):
                f.write('% .9E' % (x[rows*5 + j]))
        f.write('\n')



shot = 1
time = 1
# Now, write to file using same style as J. Menard script (listed above)
# Using function in WRITE_GFILE for reference

if writeMask:
    with open(gOut + 'g' + format(shot, '06d') + '.' + format(time,'05d'), 'w') as f:
        f.write('  EFITD    xx/xx/xxxx    #' + str(shot) + '  ' + str(time) + 'ms        ')
        f.write('   3 ' + str(Nr) + ' ' + str(Nz) + '\n')
        f.write('% .9E% .9E% .9E% .9E% .9E\n'%(Xdim, Zdim, R0, R1, Zmid))
        f.write('% .9E% .9E% .9E% .9E% .9E\n'%(RmAxis, ZmAxis, psiAxis, psiSep, Bt0))
        f.write('% .9E% .9E% .9E% .9E% .9E\n'%(Ip, psiAxis, 0, RmAxis, 0))
        f.write('% .9E% .9E% .9E% .9E% .9E\n'%(ZmAxis,0,psiSep,0,0))
        write_array(Fpol, f)
        write_array(Pres, f)
        write_array(FFprime, f)
        write_array(Pprime, f)
        write_array(psiRZ.flatten(), f)
        write_array(qPsi, f)
        f.write(str(Nlcfs) + ' ' + str(Nwall) + '\n')
        write_array(lcfs.flatten(), f)
        write_array(wall.flatten(), f)
        f.write(str(KVTOR) + ' ' + format(RVTOR, ' .9E') + ' ' + str(NMASS) + '\n')
        write_array(RHOVN, f)
