#calculates conic section area given two points that represent
#a line segment in R,Z space
import numpy as np


#1 is point closer to origin in z-coord (the smaller cone)

def conicArea(r1,z1,r2,z2):
    return np.pi*r2*(r2 + np.sqrt(z2**2+r2**2)) - np.pi*r1*(r1 + np.sqrt(z1**2 + r1**2))


##===V2_FirstWall.txt===
##T4
##r1 = 1.477 #from .txt, not end of tile in CALC
##z1 = -1.175
#r1 = 1.576
#z1 = -1.3
#r2 = 1.75
#z2 = -1.52
#A4 = conicArea(r1,z1,r2,z2)
##T6
##r1 = 1.646 #from .txt, not end of tile in CALC
##z1 = -1.2191
#r1 = 1.682
#z1 = -1.3
#r2 = 1.73
#z2 = -1.41
#A6 = conicArea(r1,z1,r2,z2)

#===V3c===
#T4
r1 = 1.57
z1 = -1.297
r2 = 1.72
z2 = -1.51
A4 = conicArea(r1,z1,r2,z2)
#T6
r1 = 1.6585
z1 = -1.2177
r2 = 1.695
z2 = -1.38
A6 = conicArea(r1,z1,r2,z2)


print(A4)
print(A6)
print(A4+A6)
