#calculates shadowed and wetted area for a given VTP shadowMask

import numpy as np
import vtk


fVTP = '/home/tlooby/results/boundaryOps/qProfiles/shadowMask_all_mesh.vtp'

r1 = 1.57
z1 = -1.297
r2 = 1.72
z2 = -1.51

conicA = np.pi*r2*(r2 + np.sqrt(z2**2+r2**2)) - np.pi*r1*(r1 + np.sqrt(z1**2 + r1**2))

def triangle_area(p1, p2, p3):
    """Calculate the area of a triangle with vertices p1, p2, and p3."""
    a = [p2[i] - p1[i] for i in range(3)]
    b = [p3[i] - p1[i] for i in range(3)]
    cross_product = [
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0]
    ]
    return 0.5 * ((cross_product[0]**2 + cross_product[1]**2 + cross_product[2]**2) ** 0.5)


def faceNormals(vtxs):
    AC = vtxs[1] - vtxs[0]
    BC = vtxs[2] - vtxs[0]
    N = np.cross(AC,BC)
    mag = np.linalg.norm(N)
    return N/mag

# Read the VTP file
reader = vtk.vtkXMLPolyDataReader()
reader.SetFileName(fVTP)
reader.Update()

# Extract the mesh
mesh = reader.GetOutput()

# Extract the scalar binary variable (assuming the data array is named 'shadowMask')
shadowMaskArray = mesh.GetPointData().GetArray("shadowMask")
shadowMask = [shadowMaskArray.GetValue(i) for i in range(shadowMaskArray.GetNumberOfTuples())]

# Calculate the normals
normalGenerator = vtk.vtkPolyDataNormals()
normalGenerator.SetInputData(mesh)
normalGenerator.ComputePointNormalsOff()
normalGenerator.ComputeCellNormalsOn()
normalGenerator.Update()

norms = np.array(normalGenerator.GetOutput().GetPointData().GetNormals())
#norms = [[normsArray.GetTuple3(i)[j] for j in range(3)] for i in range(normsArray.GetNumberOfTuples())]

# Calculate the total area of triangles where shadowMask=0
total_area = 0.0
N = mesh.GetNumberOfCells()
areas = np.zeros((N))

print(N)
for i in range(N):
    points = mesh.GetCell(i).GetPoints()
    p1, p2, p3 = [points.GetPoint(j) for j in range(3)]
    areas[i] = triangle_area(p1, p2, p3)

print("Total area of triangles with shadowMask=0:", np.sum(areas)*1e-6)




idx = 4148786
normIdx = np.repeat(norms[idx, np.newaxis], N, axis=0)
dotProds = np.dot(norms, normIdx)
use1 = np.where(dotProds > 0.0)[0]
use2 = np.where(np.abs(dotProds) < 1e-3 )[0]
use = np.intersect1d(use1, use2)

print(norms[idx])
pIdx = [points.GetPoint(j) for j in range(3)]

A2 = 0.0
for i in use:
    if shadowMask[i] == 0:
        points = mesh.GetCell(i).GetPoints()
        p1, p2, p3 = [points.GetPoint(j) for j in range(3)]
        A2 += triangle_area(p1, p2, p3)

print(A2)