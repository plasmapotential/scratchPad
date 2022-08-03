import numpy as np
import time
import sys
N=1000000
Nt = 10
STLfile1 = '/home/tom/source/dummyOutput/SOLID843___5.00mm.stl'
STLfile2 = '/home/tom/source/dummyOutput/SOLID844___standard.stl'
STLfile3 = '/home/tom/HEAT/data/NSTXU/STLs/Cube___standard.stl'

O3Dpath = '/opt/open3d/Open3D/build/lib/python_package/open3d'
sys.path.append(O3Dpath)
import open3d as o3d

#=== Single Ray
#on SOLID843
#q1 = np.array([458.2, 253.5, -1500.0])
#q2 = np.array([458.2, 253.5, -1700.0])
#on SOLID844
#q1 = np.array([500.0, 130.0, -1500.0])
#q2 = np.array([500.0, 130.0, -1700.0])
#on both
#N=2
#q1 = np.array([[458.2, 253.5, -1500.0], [500.0, 130.0, -1500.0], [458.2, 253.5, -1500.0]])
#q2 = np.array([[458.2, 253.5, -1700.0], [500.0, 130.0, -1700.0], [458.2, 253.5, -1700.0]])

#All random
q1 = np.random.random(size=(N,3))
q2 = np.random.random(size=(N,3))

rayOrig = q1
rayTerm = q2
rayVec = rayTerm - rayOrig
rayDist = np.linalg.norm(rayVec, axis=1)
rayDir = rayVec / rayDist[:,np.newaxis]

#open3D ray tracing
mesh = o3d.io.read_triangle_mesh(STLfile1)

#mesh.compute_vertex_normals()
mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
#kd_tree = o3d.geometry.KDTreeFlann(mesh)
scene = o3d.t.geometry.RaycastingScene()
mesh_id = scene.add_triangles(mesh)
#vs = np.array(mesh.vertices, dtype=np.float32)
#ts = np.array(mesh.triangles, dtype=np.uint32)
#mesh_id = scene.add_triangles(vs, ts)
t0 = time.time()
rays = o3d.core.Tensor([np.hstack([q1,rayDir])],dtype=o3d.core.Dtype.Float32)
ans = scene.cast_rays(rays)
print("time: {:f}".format(time.time() - t0))
print(ans['primitive_ids'][0])
#print(ans['geometry_ids'])
#print(ans['primitive_ids'])
input()


class dummy:
    def __init__(self, mesh):
        self.vertices = np.array(mesh.vertices, dtype=np.float32)
        self.triangles = np.array(mesh.triangles, dtype=np.uint32)
        return

class SerializableMesh:
    def __init__(self, mesh: o3d.geometry.TriangleMesh):
        self.vertices = np.array(mesh.vertices, dtype=np.float32)
        self.triangles = np.array(mesh.triangles, dtype=np.uint32)

    def to_open3d(self) -> o3d.geometry.TriangleMesh:
        return o3d.geometry.TriangleMesh(
            vertices=o3d.utility.Vector3dVector(self.vertices),
            triangles=o3d.utility.Vector3iVector(self.triangles),
        )



class parallel:
    def __init__(self, mesh, q1, rayDir):
        self.mesh = SerializableMesh(mesh)
        self.vertices = np.array(mesh.vertices, dtype=np.float32)
        self.triangles = np.array(mesh.triangles, dtype=np.uint32)
        self.q1 = q1
        self.rayDir = rayDir
        return

    def multiFunc(self, i):
        print(i)
        scene = o3d.t.geometry.RaycastingScene()
        #mesh = self.mesh.to_open3d()
        #mesh = o3d.t.geometry.TriangleMesh.from_legacy(self.mesh.to_open3d())
        #vs = np.array(mesh.vertices, dtype=np.float32)
        #ts = np.array(mesh.triangles, dtype=np.uint32)
        vs = d.vertices
        ts = d.triangles

        print(vs)
        print(ts)

        mesh_id = scene.add_triangles(vs,ts)
        rays = o3d.core.Tensor([np.hstack([self.q1[i],self.rayDir[i]])],dtype=o3d.core.Dtype.Float32)
        ans = scene.cast_rays(rays)
        print(ans['geometry_ids'])
        print(ans['primitive_ids'])
        return

    def runMulti(self):
        print("Running")
        import multiprocessing
        pool = multiprocessing.Pool(1)
        pool.map(self.multiFunc, np.arange(2))
        pool.close()
        pool.join()
        print("Done")
        return

d = dummy(mesh)
p = parallel(mesh, q1, rayDir)
print(p.vertices)
p.runMulti()



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
