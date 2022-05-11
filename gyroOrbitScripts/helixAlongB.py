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

N_sampleGyro = 20
Tsample = TGyro / N_sampleGyro

file = './struct.csv'
structdata = np.genfromtxt(file,comments='#',delimiter=',')
BtraceXYZ = np.zeros((len(structdata),3))
BtraceXYZ[:,0] = structdata[:,0]/1000.0
BtraceXYZ[:,1] = structdata[:,1]/1000.0
BtraceXYZ[:,2] = structdata[:,2]/1000.0

#Loop thru B field trace while tracing gyro orbit
gyroPhase = 1.5*np.pi
helixTrace = None
for i in range(len(BtraceXYZ)-1):
#for i in range(10):
    #calculate B at this point
    ###DO WE NEED TO CALCULATE ALL B AND FREQ QUANTITIES HERE?!
    B = 5
    #points in this iteration
    p0 = BtraceXYZ[i,:]
    p1 = BtraceXYZ[i+1,:]
    #vector
    delP = p1 - p0
    #magnitude
    magP = np.sqrt(delP[0]**2 + delP[1]**2 + delP[2]**2)
    #time it takes to transit line segment
    delta_t = magP / v_parallel
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
    #get helix path along (proxy) z axis reference frame
    x_helix = rGyro*np.cos(omegaGyro*t + gyroPhase)
    y_helix = -rGyro*np.sin(omegaGyro*t + gyroPhase)
    z_helix = np.zeros((len(t)))
    #perform rotation to field line reference frame
    helix = np.vstack([x_helix,y_helix,z_helix]).T
    helix_rot = np.zeros((len(helix),3))
    for j,coord in enumerate(helix):
        helix_rot[j,:] = helix[j,0]*u + helix[j,1]*v + helix[j,2]*w
    #perform translation to field line reference frame
    helix_rot[:,0] += xGC
    helix_rot[:,1] += yGC
    helix_rot[:,2] += zGC
    #update gyroPhase variable so next iteration starts here
    gyroPhase = omegaGyro*t[-1] + gyroPhase
    #append to helix trace
    if helixTrace is None:
        helixTrace = helix_rot
    else:
        helixTrace = np.vstack([helixTrace,helix_rot])

    if i==0:
        #create gyro-radius mesh plane
        gPs = np.linspace(0,2*np.pi,100)
        xC = rGyro * np.cos(gPs)
        yC = rGyro * np.sin(gPs)
        zC = np.zeros((len(xC)))
        circArr = np.vstack([xC,yC,zC]).T
        circRot = np.zeros((len(gPs), 3))
        for k,phase in enumerate(gPs):
            circRot[k,:] = xC[k]*u + yC[k]*v + zC[k]*w
        circRot[:,0]+=BtraceXYZ[0,0]
        circRot[:,1]+=BtraceXYZ[0,1]
        circRot[:,2]+=BtraceXYZ[0,2]



fig = go.Figure(data=[go.Scatter3d(x=BtraceXYZ[:,0],
                                   y=BtraceXYZ[:,1],
                                   z=BtraceXYZ[:,2],
                                   mode='lines',
                                   line=dict(color='royalblue', width=6),
                                   name="Optical"),
                                   ],
                                   )
fig.add_trace(go.Scatter3d(x=helixTrace[:,0],
                           y=helixTrace[:,1],
                           z=helixTrace[:,2],
                           mode='lines',
                           line=dict(color='firebrick', width=6),
                           name="Gyro Orbit"))

fig.add_trace(go.Scatter3d(x=[helixTrace[0,0]],
                           y=[helixTrace[0,1]],
                           z=[helixTrace[0,2]],
                           mode='markers',
                           marker=dict(size=12),
                           name="Launch Point"))

fig.add_trace(go.Mesh3d(x=circRot[:,0],
                           y=circRot[:,1],
                           z=circRot[:,2],
                           opacity=0.5,
                           ))

#for white bkgd
fig.update_scenes(xaxis_visible=False, yaxis_visible=False,zaxis_visible=False )
fig.update_layout(
    scene=dict(#the default values are 1.25, 1.25, 1.25
           xaxis=dict(),
           yaxis=dict(),
           zaxis=dict(),
           aspectmode='manual',
           aspectratio=dict(x=1, y=1, z=0.2),
           bgcolor='white',
           ),
           plot_bgcolor='rgb(255,255,255)'
           )


fig.update_layout(legend=dict(
    yanchor="middle",
    y=0.9,
    xanchor="right",
    x=0.80
    ),
    )



fig.show()

print("V_perp = {:f} [m/s]".format(v_perp0))
print("Cyclotron Freq = {:f} [rad/s]".format(omegaGyro))
print("Cyclotron Freq = {:f} [Hz]".format(fGyro))
print("Gyro Radius = {:f} [m]".format(rGyro))
print("Number of gyro points = {:f}".format(len(helixTrace)))
print("Longitudinal dist between gyro points = {:f} [m]".format(magP/NgyroSteps))
print("Each line segment length ~ {:f} [m]".format(magP))
