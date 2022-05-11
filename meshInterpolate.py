#meshInterpolate.py
#Description:   interpolate point cloud to an STL mesh
#Engineer:      T Looby
#Date:          20220420

import sys
import numpy as np
import pandas as pd
import sklearn

stlFile = '/home/tom/work/CFS/projects/ansysTest/ANSYSmesh843.stl'
pcFile = '/home/tom/work/CFS/projects/ansysTest/HF_optical.csv'

#HEAT path
HEATpath = '/home/tom/source/HEAT/github/source'
sys.path.append(HEATpath)

#load HEAT environment
import launchHEAT
launchHEAT.loadEnviron()

#load HEAT CAD module and STP file
import CADClass
CAD = CADClass.CAD(os.environ["rootDir"], os.environ["dataPath"])

#load STL
mesh = CAD.loadExternalSTL(stlFile)

#Now get face centers, normals, areas
norms, ctrs, areas = CAD.normsCentersAreas(mesh)

#load point cloud data
pc = pd.read_csv(pcFile, names=['X','Y','Z','Val'], skiprows=[0])
pcData = pc.values

gp = sklearn.gaussian_process.GaussianProcessRegressor(kernel=RBF(10, (1e-2, 1e2)))
gp.fit()
