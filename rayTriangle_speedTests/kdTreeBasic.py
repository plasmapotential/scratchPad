import numpy as np
import open3d as o3d
import sys
N=1
Nt = 10
STLfile1 = '/home/tom/source/dummyOutput/SOLID843___5.00mm.stl'
STLfile2 = '/home/tom/source/dummyOutput/SOLID844___standard.stl'

##load CAD via HEAT
#HEATPath = '/home/tom/source/HEAT/github/source'
#sys.path.append(HEATPath)
##load HEAT environment
#import launchHEAT
#launchHEAT.loadEnviron()
##load HEAT CAD module and STP file
#import CADClass
#CAD = CADClass.CAD(os.environ["rootDir"], os.environ["dataPath"])
#mesh1 = CAD.load1Mesh(STLfile1)
#mesh2 = CAD.load1Mesh(STLfile2)
#combinedMesh = CAD.createEmptyMesh()
#print(combinedMesh)
#combinedMesh.addFacets(mesh1.Facets)
#combinedMesh.addFacets(mesh2.Facets)

#open3d
mesh = o3d.io.read_triangle_mesh(STLfile1)
mesh.compute_vertex_normals()
#print(mesh)
#kd_tree = o3d.geometry.KDTreeFlann(mesh)

#on SOLID843
q1 = np.array([458.2, 253.5, -1500.0])
q2 = np.array([458.2, 253.5, -1700.0])
#on SOLID844
#q1 = np.array([500.0, 130.0, -1500.0])
#q2 = np.array([500.0, 130.0, -1700.0])

rayOrig = q1
rayTerm = q2
rayVec = rayTerm - rayOrig
rayDist = np.linalg.norm(rayVec)
rayDir = rayVec / rayDist

#open3D ray tracing
mesh = o3d.io.read_triangle_mesh(STLfile1)
mesh.compute_vertex_normals()
mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
#kd_tree = o3d.geometry.KDTreeFlann(mesh)
scene = o3d.t.geometry.RaycastingScene()
mesh_id = scene.add_triangles(mesh)
rays = o3d.core.Tensor([np.hstack([q1,rayDir])],dtype=o3d.core.Dtype.Float32)
ans = scene.cast_rays(rays)
print(ans.keys())

class dummy:
    def __init__(self, mesh):
        self.m = mesh
        return


class parallel:
    def __init__(self, mesh):
        self.mesh = [mesh]
        return

    def multiFunc(self, i):
        print(mesh.nearestFacetOnRay((rayOrig[0],rayOrig[1],rayOrig[2]),(rayDir[0],rayDir[1],rayDir[2])))
        print("Inside: {:d}".format(i))
        return

    def runMulti(self):
        import multiprocessing
        pool = multiprocessing.Pool(4)
        pool.map(self.multiFunc, np.arange(10))
        pool.close()
        pool.join()
        return

#d = dummy(combinedMesh)
#p = parallel(d.m)
#p.runMulti()



#random ray/triangles:
#initialize triangles and rays
#print("N = {:d}".format(N))
#print("Nt = {:d}".format(Nt))
#All random
#q1 = np.random.random(size=(N,3))
#q2 = np.random.random(size=(N,3))
#t1 = np.random.random(size=(Nt,3))
#t2 = np.random.random(size=(Nt,3))
#t3 = np.random.random(size=(Nt,3))
#triangles = np.hstack([t1,t2,t3])
#kd1 = cKDTree(triangles)
#print(dir(kd1))
