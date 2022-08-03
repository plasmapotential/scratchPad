import numpy as np
import pandas as pd
import sys, os
HEATPath = '/home/tom/source/HEAT/github/source'
sys.path.append(HEATPath)
import toolsClass
tools = toolsClass.tools()
tools.rootDir = HEATPath


#heat fluxes we want in new files above thresh
file1 = '/home/tom/results/NSTXU/NF_gyro_paper/castellation/nstx_204118_castellationGyro_555/000010/castellationTop/HF_gyro.csv'
file2 = '/home/tom/results/NSTXU/NF_gyro_paper/castellation/nstx_204118_castellationGyro_555/000010/castellationTop/HF_optical.csv'

#heat fluxes we are using as a boundary condition at lower resolution
file3 = '/home/tom/results/NSTXU/NF_gyro_paper/castellation/lowRes_entireCastellation_oldAlgorithmsOct2021/castellation/HF_optical.csv'

#threshold.  points above this threshold will be taken from file1 and file2
#points below it will be taken from file 3
z_thresh = -1608.0

#heat loads at high resolution
data = pd.read_csv(file1)
xyz1 = data.iloc[:,0:3].values
q_gyro = data.iloc[:,3].values

data2 = pd.read_csv(file2)
xyz2 = data2.iloc[:,0:3].values
q_opt = data2.iloc[:,3].values

#heat loads at low resolution
data3 = pd.read_csv(file3)
xyz3 = data3.iloc[:,0:3].values
q_low = data3.iloc[:,3].values

#get all points with z value above thresh
useUp = np.where(xyz1[:,-1] > z_thresh)[0]
useDown = np.where(xyz3[:,-1] <= z_thresh)[0]

#create new PCs
N_up = len(useUp)
N_down = len(useDown)
N = N_up + N_down
q_gyroNew = np.zeros((N))
q_optNew = np.zeros((N))
q_allNew = np.zeros((N))
q_gyroNew = np.append(q_gyro[useUp], q_low[useDown])
q_optNew = np.append(q_opt[useUp], q_low[useDown])
xyzNew = np.append(xyz1[useUp,:], xyz3[useDown,:], axis=0)
q_allSourcesNew = q_gyroNew + q_optNew

#save new pointclouds
newDir = '/home/tom/results/NSTXU/NF_gyro_paper/castellation/newPointClouds/'
newFile1 = newDir + 'HF_optical.csv'
newFile2 = newDir + 'HF_gyro.csv'
newFile3 = newDir + 'HF_allSources.csv'
head = "X,Y,Z,HeatFlux"
pc = np.zeros((N,4))
print(pc.shape)
pc[:,0:3] = xyzNew
pc[:,3] = q_optNew
np.savetxt(newFile1, pc, delimiter=',',fmt='%.10f', header=head)
pc[:,3] = q_gyroNew
np.savetxt(newFile2, pc, delimiter=',',fmt='%.10f', header=head)
pc[:,3] = q_allSourcesNew
np.savetxt(newFile3, pc, delimiter=',',fmt='%.10f', header=head)

#create vtk files
pvpythonCMD = '/opt/paraview/ParaView-5.10.1-MPI-Linux-Python3.9-x86_64/bin/pvpython'
os.environ["pvpythonCMD"] = pvpythonCMD
tools.createVTKOutput(newFile1, 'points', 'HF_optical')
tools.createVTKOutput(newFile2, 'points', 'HF_gyro')
tools.createVTKOutput(newFile3, 'points', 'HF_allSources')
