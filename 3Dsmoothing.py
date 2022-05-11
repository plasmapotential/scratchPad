#script for testing smoothing algorithms
#smooths 3D pointcloud data

import numpy as np
import pandas as pd
import sys
import os
from scipy.ndimage import gaussian_filter

HEATPath = '/home/tom/source/HEAT/github/source'
sys.path.append(HEATPath)
import toolsClass
tools = toolsClass.tools()
tools.rootDir = HEATPath

#f = '/home/tom/results/limiter_gyro/666_100eV_lq3mm_S1mm_2mmres/T010/HF_gyro.csv'
#f = '/home/tom/results/limiter_gyro/666_100eV_lq3mm_S1mm_2mmres/T010/HF_optical.csv'
#fOut = '/home/tom/results/limiter_gyro/666_100eV_lq3mm_S1mm_2mmres/T010/HF_gyro_SMOOTH.csv'

f = '/home/tom/results/castellationTopDiamagnetic/nstx_204118_10eV_444/castellationTop/HF_gyro.csv'
fOut = '/home/tom/results/castellationTopDiamagnetic/nstx_204118_10eV_444/castellationTop/HF_smooth.csv'


data = pd.read_csv(f)
xyz = data.values[:,0:3]
print(xyz.shape)
HF = data.values[:,-1]
#hfPC = data.values
N = len(data)
convHF = np.zeros((N))


d = np.zeros((N))
blurredHF = np.zeros((N))
thresh = 0.2
for i in range(N):
    d = np.linalg.norm(xyz[i] - xyz, axis=1)
    #blur via averaging
    use = np.where(d<thresh)[0]
    blurredHF[i] = np.sum(HF[use]) / len(use)
    ##blur via max (like softmax)
    #use = np.where(d<thresh)[0]
    #blurredHF[i] = np.max(HF[use])

#gaussian blur matrix / convolution matrix
#sigma = 1.0
#use = np.where(HF > 0.0)[0]
#for i in range(len(use)):
#    blurredHF[use[i]] = 1.0 / (sigma * np.sqrt(2*np.pi)) * np.sum(np.exp(-( d[use[i],:]**2 / ( 2.0 * sigma**2 ) ) * HF[use]))
#blurredHF = blurredHF / np.sum(HF)




#use this to create a VTK object with points used in plot
vtk = True
if vtk == True:
    pc = np.zeros((N, 4))
    pc[:,0] = xyz[:,0]
    pc[:,1] = xyz[:,1]
    pc[:,2] = xyz[:,2]
    pc[:,3] = blurredHF
    head = "X,Y,Z,HF"
    np.savetxt(fOut, pc, delimiter=',',fmt='%.10f', header=head)
    #make pointcloud of x and y for viewing in PV
    pvpythonCMD = '/opt/paraview/oldRevs/ParaView-5.9.0-RC1-MPI-Linux-Python3.8-64bit/bin/pvpython'
    os.environ["pvpythonCMD"] = pvpythonCMD
    tools.createVTKOutput(fOut, 'points', 'HF_smooth')
