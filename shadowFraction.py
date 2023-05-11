#shadowFraction.py
#Description:   given tile polygon and shadow polygon, calculate shdowed fraction
#               assumes trapezoidal tile
#Engineer:      T Looby
#Date:          20230209

import numpy as np
from shapely.geometry import Polygon

#theoretical area
#annulus
rMax = 1840.0
rMin = 1720.0
N = 720 #mode number of tiles
A_theory = np.pi * (rMax**2-rMin**2) / 720


#width of tile at narrow (tile) end
a = 14.09
#width of tile at wide end
b = 15.06
#side length
c = 119.98
#calculate tile length (not side length)
d = (b-a)/2.0
e = np.sqrt(c**2 - d**2)

#build x,y coordinates assuming one point is 0,0
x = np.array([0.0, b, a+d, d])
y = np.array([0.0, 0.0, e, e])
tile = Polygon(zip(x, y))
#calculate tile area
#A = e*(d+a)

#calculate shadow area (assumes also polygonal and linear shadow)
#example
#===edit these for each shadow case:
#fat end of shadow
s1 = 1.47
#skinny end of shadow
s2 = 1.2

x = np.array([0.0, s2, s1+d, d])
y = np.array([0.0, 0.0, e, e])
shadow = Polygon(zip(x, y))

fracShad = shadow.area / tile.area
fracWet = 1-fracShad

print("Tile Top Surface Area: {:f} [mm^2]".format(tile.area))
print("Shadow Top Surface Area: {:f} [mm^2]".format(shadow.area))
print("Wetted Fraction: {:f}".format(fracWet))
print("Shadowed Fraction: {:f}".format(fracShad))
print("Theoretical Shadowed Frac: {:f}".format(shadow.area / A_theory))
