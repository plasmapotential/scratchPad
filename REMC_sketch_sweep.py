#script for sweeping an RZ sketch.  this one for the REMC

import numpy as np
#3d coordinates
#bracket
p1 = np.array([-2265.79242243956, -9.8864472259878, 725.224707322426])
p2 = np.array([-2356.36572751177, -10.2816503310082, 574.483979098982])
vec = p2-p1
d = np.linalg.norm(vec)
vecN = vec / d
mid = vecN * d/2.0 + p1
#joint
p3 = np.array([-2277.8867836526, 219.335550299657, 662.404729434871])
p4 = np.array([-2318.04613453925, 223.202455972762, 595.259221128946])
vecJ = p4-p3
dJ = np.linalg.norm(vecJ)
vecNJ = vecJ / dJ
midJ = vecNJ * d/2.0 + p3


#RZ coordinates
p1RZ = np.array([np.sqrt(p1[0]**2+p1[1]**2), p1[2]])
p2RZ = np.array([np.sqrt(p2[0]**2+p2[1]**2), p2[2]])
vec = p2RZ-p1RZ
d = np.linalg.norm(vec)
vecN = vec / d
mid = vecN * d/2.0 + p1RZ

p3RZ = np.array([np.sqrt(p3[0]**2+p3[1]**2), p3[2]])
p4RZ = np.array([np.sqrt(p4[0]**2+p4[1]**2), p4[2]])
vecJ = p4RZ-p3RZ
dJ = np.linalg.norm(vecJ)
vecNJ = vecJ / dJ
midJ = vecNJ * dJ/2.0 + p3RZ


#calculate projection of middle of REMC onto VSC bracket plane
v2 = midJ - p1RZ
l2 = np.linalg.norm(v2)
v2N = v2/l2
d2 = np.dot(v2N, vecN)
mid2 = vecN * l2 + p1RZ


btm = vecN * 65 + mid2
top = -1.0*vecN * 65 + mid2

top
btm




