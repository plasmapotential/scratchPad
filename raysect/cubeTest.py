import numpy as np
from raysect.primitive.mesh import stl
from raysect.optical import World
from raysect.core.ray import Ray as CoreRay
from raysect.core.intersection import Intersection
from raysect.core import Point3D, Vector3D
from raysect.core import MulticoreEngine

file = '/home/tom/source/test/raysect/cube.stl'
name = 'tomas'


world = World()
mesh = stl.import_stl(file, scaling=1.0, mode='auto', parent=world)
mesh.name = name

rayOrig = np.array([50.0, -100.0, 50.0])
rayTerm = np.array([50.0, 500.0, 50.0])

rayVec = rayTerm - rayOrig
rayDist = np.linalg.norm(rayVec) # use norm from raysect.core
rayDir = rayVec / rayDist

ray = CoreRay()
ray.origin = Point3D(rayOrig[0], rayOrig[1], rayOrig[2])
ray.direction = Vector3D(rayDir[0], rayDir[1], rayDir[2])
ray.direction = Vector3D(rayVec[0], rayVec[1], rayVec[2]).normalise()
ray.max_distance = rayDist

intersection = world.hit(ray)
print(ray.origin)
print(ray.max_distance)
print(ray.direction)
print(intersection.hit_point)
print(intersection.primitive.name)
