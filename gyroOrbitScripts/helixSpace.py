import numpy as np
import plotly.graph_objects as go

#user input
vTot = 1000.0
omega = 2*np.pi*10

#bfield line
q1 = np.array([0,0,0])
q2 = np.array([0,200,0])
Bvec = np.vstack([q1,q2])
delQ = q2 - q1
Bnorm = delQ / np.linalg.norm(delQ)


pt = np.array([[0.0, 150, 1],
               [5, 120, 8],
               [12, 150, 5]])

magPt = np.linalg.norm(pt, axis=1)
normPt = pt / magPt



# construct orthogonal system for coordinate transformation
w = delQ / np.linalg.norm(delQ)
print(w)
if np.all(w==[0,0,1]):
    u = np.cross(w,[0,1,0]) #prevent failure if bhat = [0,0,1]
else:
    u = np.cross(w,[0,0,1]) #this would fail if bhat = [0,0,1] (rare)
v = np.cross(u,w)
#normalize
u = u / np.sqrt(u.dot(u))
v = v / np.sqrt(v.dot(v))
w = w / np.sqrt(w.dot(w))
xfm = np.flip(np.vstack([u,v,w]), 0)

B2pt = pt-q1
B2ptNorm = B2pt / np.linalg.norm(B2pt)
#gyroPhase angle
theta = np.arctan2(np.dot(B2ptNorm, u), np.dot(B2ptNorm, v))

#distance along field line
d = np.dot(pt, Bnorm)
#radius
r = np.sqrt(magPt**2 - d**2)
#period
L = np.sqrt((vTot**2 / omega**2) - r**2)
#velocity phase
vP = np.arccos(r*omega/vTot)
#gyro phase at q1
gP = theta - omega * d/(vTot*np.sin(vP))

#helix
def helix(v,vP,r,d):
    tMax = d / (v*np.sin(vP))
    t = np.linspace(0,tMax,100)
    x = r*np.cos(omega*t+gP)
    y = r*np.sin(omega*t+gP)
    z = v*np.sin(vP) * t
    return x,y,z

xH,yH,zH = helix(vTot,vP,r,d)

fig = go.Figure(data=[go.Scatter3d(x=xH[:,0], y=yH[:,0], z=zH[:,0],
                                   mode='markers')])
fig.add_trace(go.Scatter3d(x=xH[:,1], y=yH[:,1], z=zH[:,1],
                                   mode='markers'))
fig.add_trace(go.Scatter3d(x=xH[:,2], y=yH[:,2], z=zH[:,2],
                                   mode='markers'))

Bvec = np.matmul(Bvec, xfm)

fig.add_trace(go.Scatter3d(x=Bvec[:,0], y=Bvec[:,1], z=Bvec[:,2],
                                   mode='lines'))
#starting triangle
pt = np.matmul(pt,xfm)
triX = np.append(pt[:,0], pt[0,0])
triY = np.append(pt[:,1], pt[0,1])
triZ = np.append(pt[:,2], pt[0,2])
fig.add_trace(go.Mesh3d(x=triX, y=triY, z=triZ))

#ending triangle
endTri = np.zeros((3,3))
endTri[0,0] = xH[0,0]
endTri[1,0] = xH[0,1]
endTri[2,0] = xH[0,2]
endTri[0,1] = yH[0,0]
endTri[1,1] = yH[0,1]
endTri[2,1] = yH[0,2]
endTri[0,2] = zH[0,0]
endTri[1,2] = zH[0,1]
endTri[2,2] = zH[0,2]
endTri = np.vstack([endTri, endTri[0]])
fig.add_trace(go.Mesh3d(x=endTri[:,0], y=endTri[:,1], z=endTri[:,2]))


fig.update_layout(
    font=dict(
        family="Courier New, monospace",
        size=18,
        color="RebeccaPurple"
    )
)


fig.show()
