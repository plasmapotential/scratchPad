import numpy as np
import plotly.graph_objects as go




x = np.linspace(0,150,50)
y = np.linspace(0,150,50)
z = np.linspace(0,100,50)
#y = np.zeros((50))
#z = np.zeros((50))

line = np.vstack([x,y,z]).T

fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z,
                                   mode='lines+markers')])

p0 = np.array([x[0],y[0],z[0]])
p1 = np.array([x[-1],y[-1],z[-1]])
delP = p1 - p0


l = np.sqrt(delP[0]**2 + delP[1]**2 + delP[2]**2)
t = np.linspace(0,l,50)

w = delP
u = np.cross(w,[0,0,1])
v = np.cross(w,u)
print(u)
print(v)
print(w)
u = u / np.sqrt(u.dot(u))
v = v / np.sqrt(v.dot(v))
w = w / np.sqrt(w.dot(w))
xfm = np.vstack([u,v,w]).T
print(xfm)


a = 10
omega = 4

x_helix = a*np.cos(omega*t)
y_helix = a*np.sin(omega*t)
z_helix = t

helix = np.vstack([x_helix,y_helix,z_helix]).T

helix_rot = np.zeros((len(helix),3))
for i,coord in enumerate(helix):
    helix_rot[i,:] = helix[i,0]*u + helix[i,1]*v + helix[i,2]*w

fig.add_trace(go.Scatter3d(x=helix_rot[:,0], y=helix_rot[:,1], z=helix_rot[:,2],mode='lines+markers'))
fig.add_trace(go.Scatter3d(x=[160,200,180,160], y=[100,100,100,100], z=[100,100,150,100]))
fig.show()
