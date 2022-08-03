import bpy

f1 = '/home/tom/source/dummyOutput/SOLID843___5.00mm.stl'
m1 = bpy.ops.import_mesh.stl(filepath=f1)

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
