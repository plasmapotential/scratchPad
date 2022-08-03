import numpy as np

import sys
N=1
Nt = 10
STLfile1 = '/home/tom/source/dummyOutput/SOLID843___5.00mm.stl'
STLfile2 = '/home/tom/source/dummyOutput/SOLID844___standard.stl'
STLfile3 = '/home/tom/HEAT/data/NSTXU/STLs/Cube___standard.stl'

O3Dpath = '/opt/open3d/Open3D/build/lib/python_package/open3d'
sys.path.append(O3Dpath)
import open3d as o3d

#=== Single Ray
#on SOLID843
q1 = np.array([458.2, 253.5, -1500.0])
q2 = np.array([458.2, 253.5, -1600.0])
#on SOLID844
#q1 = np.array([500.0, 130.0, -1500.0])
#q2 = np.array([500.0, 130.0, -1700.0])
#on both
#q1 = np.array([[458.2, 253.5, -1500.0], [500.0, 130.0, -1500.0]])
#q2 = np.array([[458.2, 253.5, -1700.0], [500.0, 130.0, -1700.0]])
rayOrig = q1
rayTerm = q2
rayVec = rayTerm - rayOrig
rayDist = np.linalg.norm(rayVec)
rayDir = rayVec / rayDist


#All random
#q1 = np.random.random(size=(N,3))
#q2 = np.random.random(size=(N,3))
#t1 = np.random.random(size=(Nt,3))
#t2 = np.random.random(size=(Nt,3))
#t3 = np.random.random(size=(Nt,3))



#open3D ray tracing
mesh = o3d.io.read_triangle_mesh(STLfile1)
#mesh.compute_vertex_normals()
mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
#kd_tree = o3d.geometry.KDTreeFlann(mesh)
scene = o3d.t.geometry.RaycastingScene()
mesh_id = scene.add_triangles(mesh)
rays = o3d.core.Tensor([np.hstack([q1,rayVec])],dtype=o3d.core.Dtype.Float32)
ans = scene.cast_rays(rays)
print(ans['geometry_ids'])
print(ans['primitive_ids'])
print(ans['t_hit'])

print(type(ans['geometry_ids']))
print(ans['geometry_ids'][0].numpy())
