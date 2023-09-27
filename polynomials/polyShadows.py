#polynomialShadow.py
#Description:   Calculates quantities for shadowing with polynomials
#Engineer:      T Looby
#Date:          20230208

import numpy as np
import plotly.graph_objects as go

#=== inputs
#gap size [mm]
g = 0.5
#angle of incidence
alpha = np.radians(1.0) #degrees
#choose sample points for tile surface at apex and edge of tile
#sample points occur at (0,h) and (w,0)
h = 1.0 #height of curve [mm]
w = 25.0 #half-width of tile [mm]

#=== polynomial coefficients
N = 4
x = np.array([0.0, w])
#Vandermonde Matrix
A = np.vander(x, N)
#print(A)

#exclude odd coefficients if desired
#A[:,0::2] = 0.0
#print(A)

b = np.array([h, 0.0])
#use Moore-Penrose Pseudo Inverse Matrix to find coefficients
coeffs = np.matmul(np.linalg.pinv(A),b)


#6th order polynomial here
#example even polynomial
#coeffs = [ 
#   0.0000000e+00,
#  -9.9990001e-05,
#   0.0000000e+00,
#  -9.9990001e-07,
#   0.0000000e+00,
#   1.0000000e+00,
# ]
#example odd polynomial
#coeffs = [
# -9.9e-06,
# -9.9e-07,
# -9.9e-08,
# -9.9e-09,
# -9.9e-10,
#  1.0e+00,
#  ]
#other example
#coeffs = [
#    -0.1,
#    0.0,
#    0.0,
#    0.0,
#    -0.1,
#    0.0,
#    1.0,
#    ]
#

#define the polynomial function
fs = np.poly1d(coeffs)
dfs_dx = np.polyder(fs)
dCoeffs = np.polyder(coeffs)

#plot the polynomial if desired
#fig = go.Figure()
#t = np.linspace(0,w,100)
#fig.add_trace(go.Scatter(x=t, y=fs(t)))
#fig.update_layout(
#    xaxis_title="x [mm]",
#    yaxis_title="y [mm]",
#    font=dict(
#        size=18,
#        ),
#    )
#fig.show()

#add polynomial to csv file
#fOut = '/home/tlooby/projects/polyShadows/poly.csv'
#with open(fOut, 'a') as f:
#    tStr = np.char.mod('%f',t)
#    tLine = ",".join(tStr)
#    yStr = np.char.mod('%f',fs(t))
#    yLine = ",".join(yStr)
#    f.write(tLine + '\n')
#    f.write(yLine + '\n')


print("Polynomial:")
print(fs)
print("Test at x=w (should be close to 0):")
print(fs(w))



#half of 1 tile's toroidal with (ie horizontal distance from apex to edge)
#calculated to check polynomial
rts = np.roots(coeffs)
use = np.where(np.logical_and(rts.real>0.0, rts.imag == 0.0))[0]
w1 = float(rts[use].real)
print("PFC half width as root: {:f} [mm]".format(w1))

#slope of field line (downward into tile surface)
m = -1.0 * np.tan(alpha)
fB0 = np.poly1d([m, 0.0])

#we define x=0 to be the location of the middle of the tile (the apex)
#find y0, the y intercept of a field line that strikes the downstream
#PFCs leading edge corner
y0 = -1.0 * fB0(w+g)

#build a polynomial for this field line
B0_coeffs = np.array([m])
B_order = 0 #linear

#find the x location of the last shadow on the upstream PFC, x_tangent
fsMinusfB = dCoeffs.copy()
fsMinusfB[-(B_order+1):] -= B0_coeffs
rts = np.roots(fsMinusfB)
use = np.where(np.logical_and(rts.real>0.0, rts.imag == 0.0))[0]
x_tangent = float(rts[use].real)
print("X location of last shadow: {:f} [mm]".format(x_tangent))

#find the y-intercept for the field line that goes through x_tangent
y1 = fs(x_tangent) - fB0(x_tangent)
print("y-axis intercept for tangent field line: {:f} [mm]".format(y1))

#evaluate the elevation that the downstream PFC would need to move 
# for the leading edge corner to be loaded
B1_coeffs = np.array([m, y1])
fB1 = np.poly1d(B1_coeffs)
delta_h = fB1(g+w)
print("Maximum elevation of tile to remain shadowed: {:f} [mm]".format(delta_h))



#print(coeffs)

