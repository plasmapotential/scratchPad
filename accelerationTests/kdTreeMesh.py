#script for testing out a kdtree algorithm using freecad kdtrees

import sys
import numpy as np
from scipy.spatial import KDTree
FreeCADPath = '/usr/lib/freecad-daily-python3/lib'
sys.path.append(FreeCADPath)
import FreeCAD
import Mesh
from scipy import spatial


#file = '/home/tom/source/test/raysect/cube.stl'
#file = '/home/tom/source/test/raysect/cube1mm.stl'
#file = '/home/tom/HEAT/data/NSTX/STLs/T000___3.0mm.stl'
#file = '/home/tom/HEAT/data/NSTX/STLs/narrowSlice___standardmm.stl'
file = '/home/tom/HEAT/data/NSTX/STLs/narrowSlice001___standardmm.stl'
mesh = Mesh.Mesh(file)


#rayOrig = np.array([50.0, -100.0, 50.0])
#rayTerm = np.array([40.0, 500.0, 50.0])
rayOrig = np.array([604.928, 0, -1609.06])
rayTerm = np.array([604.985, 10.56, -1608.35])


#triangle = np.asarray(mesh.Facets[0].Points)
#face centers
#rayOrig = np.zeros((3))
#rayOrig[0] = np.sum(triangle[:,0])/3.0
#rayOrig[1] = np.sum(triangle[:,1])/3.0
#rayOrig[2] = np.sum(triangle[:,2])/3.0
#rayTerm = np.asarray(mesh.Facets[0].Normal)


rayVec = rayTerm - rayOrig
rayDist = np.linalg.norm(rayVec)
rayDir = rayVec / rayDist

#=== Using toroidal angle filter
def xyz2cyl(x,y,z):
    """
    Converts x,y,z coordinates to r,z,phi
    """
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)
    r = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y,x)
    #phi = np.radians(phi)
    return r,z,phi


q1 = np.array([rayOrig])
q2 = np.array([rayTerm])
i=0

targetPoints = []
for face in mesh.Facets:
    targetPoints.append(face.Points)

targetPoints = np.asarray(targetPoints)/1000.0 #scale to m
p1 = targetPoints[:,0,:]  #point 1 of mesh triangle
p2 = targetPoints[:,1,:]  #point 2 of mesh triangle
p3 = targetPoints[:,2,:]  #point 3 of mesh triangle
#Prepare for toroidal angle filtering
R,Z,phi = xyz2cyl(p1[:,0],p1[:,1],p1[:,2])
phiP1 = phi
R,Z,phi = xyz2cyl(p2[:,0],p2[:,1],p2[:,2])
phiP2 = phi
R,Z,phi = xyz2cyl(p3[:,0],p3[:,1],p3[:,2])
phiP3 = phi

#filter by toroidal angle
R,Z,phi = xyz2cyl(q1[i,0],q1[i,1],q1[i,2])
phiMin = phi
R,Z,phi = xyz2cyl(q2[i,0],q2[i,1],q2[i,2])
phiMax = phi

#target faces outside of this toroidal slice
test0 = np.logical_and(phiP1 < phiMin, phiP2 < phiMin, phiP3 < phiMin)
test1 = np.logical_and(phiP1 > phiMax, phiP2 > phiMax, phiP3 > phiMax)
test = np.logical_or(test0,test1)
use = np.where(test == False)[0]




#np.logical_and(phiP1 > phiMin, phiP2 < phiMax)
#np.logical_and(phiP2 > phiMin, phiP2 < phiMax)
#np.logical_and(phiP3 > phiMin, phiP3 < phiMax)
#test = np.logical_and(test1,test2,test3)
#use = np.where(test==True)[0]






print(use)


#=== using scipy.spatial.KDTree algorithm
N_facets = mesh.CountFacets
x = np.zeros((N_facets,3))
y = np.zeros((N_facets,3))
z = np.zeros((N_facets,3))
for i,facet in enumerate(mesh.Facets):
    #mesh points
    for j in range(3):
        x[i][j] = facet.Points[j][0]
        y[i][j] = facet.Points[j][1]
        z[i][j] = facet.Points[j][2]

x = x.flatten()
y = y.flatten()
z = z.flatten()
X,Y,Z = np.meshgrid(x, y, z, indexing='ij')

points = np.c_[X.ravel(), Y.ravel(), Z.ravel()]
tree = spatial.KDTree(np.array([x,y,z]).T)
sorted(tree.query_ball_point(rayOrig, rayDist))



#=== using FreeCAD internal algorithm
#intersect
intersect = mesh.nearestFacetOnRay((rayOrig[0],rayOrig[1],rayOrig[2]),(rayDir[0],rayDir[1],rayDir[2]))
idx = list(intersect.keys())[0]
loc = list(intersect.values())[0]
newRay = loc - rayOrig
print(intersect)
# remove self intersections
mesh1 = mesh.copy()
frontFaces = np.array([21])
mesh1.removeFacets(frontFaces)
mesh1.nearestFacetOnRay((rayOrig[0],rayOrig[1],rayOrig[2]),(rayDir[0],rayDir[1],rayDir[2]))
