import numpy as np
import plotly.graph_objects as go

#unit conversions
kg2eV = 5.609e35 #1kg = 5.609e35 eV/c^2
eV2K = 1.160e4 #1ev=1.160e4 K

#Constants: these definitions follow Freidberg Section 7.7
AMU = 931.494e6 #ev/c^2
Mp_eV = 1.007*AMU #eV/c^2
Md_eV = 2.014*AMU  #eV/c^2
Z = 1 #1 for deuterium
kB = 8.617e-5 #ev/K
B = 5 #tesla
e = 1.602e-19 # C
T0_eV = 15000 # eV
c = 299792458 #m/s

#velocities calculated from thermal velocity
v_perp0 = np.sqrt(2*T0_eV/(Md_eV/c**2))
v_parallel = np.sqrt(2*T0_eV/(Md_eV/c**2))
#frequencies and periods
omegaGyro = Z * e * B / (Md_eV / kg2eV)
fGyro = omegaGyro/(2*np.pi)
TGyro = 1/fGyro
#gyro radius
rGyro = v_perp0 / omegaGyro

#Start / End point of this line segment
p0 = np.array([0.123,0.056,-1.604])
p1 = np.array([0.250,0.178,-1.500])
delP = p1 - p0
magP = np.sqrt(delP[0]**2 + delP[1]**2 + delP[2]**2)
#number of discrete linear segments we will approx. gyro orbit with
N_sampleGyro = 8
#time it takes to transit line segment
delta_t = magP / v_parallel
#period at which we sample particle position (must be better than Nyquist)
Tsample = TGyro / 8
#Number of steps in entire line segment
NgyroSteps = int(delta_t / Tsample)
#length (in time) along guiding center
t = np.linspace(0,delta_t,NgyroSteps+1)
#guiding center location
xGC = np.linspace(p0[0],p1[0],NgyroSteps+1)
yGC = np.linspace(p0[1],p1[1],NgyroSteps+1)
zGC = np.linspace(p0[2],p1[2],NgyroSteps+1)

# construct orthogonal system for coordinate transformation
w = delP
u = np.cross(w,[0,0,1])
v = np.cross(w,u)
#normalize
u = u / np.sqrt(u.dot(u))
v = v / np.sqrt(v.dot(v))
w = w / np.sqrt(w.dot(w))
xfm = np.vstack([u,v,w]).T

def getHelixPath(rGyro, omegaGyro, gyroPhase, t):
    x = rGyro*np.cos(omegaGyro*t + gyroPhase)
    y = rGyro*np.sin(omegaGyro*t + gyroPhase)
    z = np.zeros((len(t)))
    return x,y,z

#helix equations (first along z-axis=t)
x_helix,y_helix,z_helix = getHelixPath(rGyro, omegaGyro, np.pi, t)
#gyroPhase = 0
#x_helix = rGyro*np.cos(omegaGyro*t + gyroPhase)
#y_helix = rGyro*np.sin(omegaGyro*t + gyroPhase)
#z_helix = np.zeros((len(t)))
#perform rotation
helix = np.vstack([x_helix,y_helix,z_helix]).T
helix_rot = np.zeros((len(helix),3))
for i,coord in enumerate(helix):
    helix_rot[i,:] = helix[i,0]*u + helix[i,1]*v + helix[i,2]*w
#perform translation
helix_rot[:,0] += xGC
helix_rot[:,1] += yGC
helix_rot[:,2] += zGC

fig = go.Figure(data=[go.Scatter3d(x=[p0[0], p1[0]],
                                   y=[p0[1], p1[1]],
                                   z=[p0[2], p1[2]],
                                   mode='lines+markers')])
fig.add_trace(go.Scatter3d(x=helix_rot[:,0],
                           y=helix_rot[:,1],
                           z=helix_rot[:,2],
                           mode='lines'))
fig.show()


print("V_perp = {:f} [m/s]".format(v_perp0))
print("Cyclotron Freq = {:f} [rad/s]".format(omegaGyro))
print("Cyclotron Freq = {:f} [Hz]".format(fGyro))
print("Gyro Radius = {:f} [m]".format(rGyro))
print("Number of gyro points = {:f}".format(len(t)))
print("Longitudinal dist between orbits = {:f} [m]".format(magP/NgyroSteps))
print("Line segment length: {:f} [m]".format(magP))
