#calculate intersection point of curve and plane
import numpy as np
from scipy.interpolate import interp1d
import plotly.graph_objects as go
import plotly.express as px

def triangleNorm(p1,p2,p3):
    #calculates triangle norm
    n = np.cross((p1-p2),(p3-p2))
    nHat = n / np.linalg.norm(n)
    return nHat

def tPoint(x,y,z,t,norm, p1):
    #calculates point of t where <x,y,z> curve intersects triangle plane
    helix = np.vstack([x,y,z]).T
    dotProduct = np.sum(norm*helix, axis=1)
    fInverse = interp1d(dotProduct, t, bounds_error=False, fill_value="extrapolate")
    c = np.dot(norm, p1)
    print("Constant: {:f}".format(c))
#    tPoint = fInverse(c)
#    print(min(dotProduct))
#    print(max(dotProduct))
    tPoint = inverseCurveInterp(t,dotProduct,c)
    print(tPoint)

    plotMask=False
    if plotMask == True:
        fig = px.scatter(x=t, y=dotProduct)
        fig.add_hline(y=c)
        #fig = px.scatter(x=dotProduct, y=t)
        #fig.add_vline(x=c)
        #fig.add_hline(y=tPoint)
        fig.show()
    return tPoint

def inverseCurveInterp(x,y,val):
    """
    performs inverse interpolation on non-monotonic curve
    and finds first point where y=val

    x is independent variable, 1D array
    y is dependent variable, 1D array matching length of x
    val is requested interpolation value of y, scalar
    returns all values of x where y=val, or nan if y never becomes val
    """
    #find index where curve crosses val
    idx = []
    for i in range(len(x)-1):
        if y[i] > val and y[i+1] < val:
            idx.append(i)

    #interpolate real x value from curve
    if len(idx) == 0:
        xVal = [np.nan]
    else:
        xVal = []
        for i in range(len(idx)):
            frac = (val-y[idx[i]]) / (y[idx[i]+1]-y[idx[i]])
            xVal.append((x[idx[i]+1] - x[idx[i]]) *frac + x[idx[i]])
    return np.array(xVal)

def barycentricCoords(pt,p1,p2,p3):
    """
    finds barycentric coordinates of pt w.r.t triangle
    with vertices, p1,p2,p3
    """
    #FINISH THIS FUNCTION!
    T = np.zeros((2,2))



def insideTriangle(pt,p1,p2,p3):
    """
    pt is point we are checking
    p1,p2,p3 are vertices of triangle
    determines if a point is inside triangle
    returns boolean, true if inside triangle

    THIS DOESNT WORK!!!
    """
    #transform pt to same relative location to p1 from origin
    P = pt-p1

    #get coordinates
    u = p2-p1
    v = p3-p1
    uMag = np.linalg.norm(u)
    vMag = np.linalg.norm(v)
    uHat = u / uMag
    vHat = v / vMag

    #dot products
    PdotU = np.dot(P,uHat)
    PdotV = np.dot(P,vHat)

    print(PdotU)
    print(uMag)
    print(PdotV)
    print(vMag)

    #check for pt inside triangle
    if (PdotU > uMag) or (PdotU < 0):
        test1=False
    else:
        test1=True

    sumP = PdotV+PdotU
    if (sumP > (uMag+vMag)) or (sumP < 0):
        test2=False
    else:
        test2=True

    shadowMask = test1 or test2
    print(shadowMask)
    return shadowMask

#triangle points
#perpendicular triangle
#p1 = np.array([-1,-1,0])
#p2 = np.array([1,-1,0])
#p3 = np.array([0,1,0])
#parallel triangle
p1 = np.array([-1,-1,0])
p2 = np.array([-1,10,0])
p3 = np.array([-1,0,-5])
triPts = np.vstack([p1,p2,p3])
norm = triangleNorm(p1,p2,p3)
print(norm)

#origin point of curve
q1 = np.array([0,0,-5])

omega = 2
vPar = 1
tMax = 10
tMin = 0
t = np.linspace(tMin,tMax,1000)


fX = lambda x: 6.0*np.cos(omega*x) + q1[0]
fY = lambda x: 6.0*np.sin(omega*x) + q1[1]
fZ = lambda x: vPar * x + q1[2]

x = fX(t)
y = fY(t)
z = fZ(t)

tPt = tPoint(x,y,z,t,norm, p1)

fig = go.Figure()
fig.add_trace(go.Mesh3d(x=triPts[:,0], y=triPts[:,1], z=triPts[:,2], color='lightpink', opacity=0.50))
fig.add_trace(go.Scatter3d(x=x,y=y,z=z,mode='lines',line=dict(color='blue', width=6),name="Gyro Orbit"))
for i in range(len(tPt)):
    if tPt[i] == np.nan:
        print('No Intersections')
        pass
    else:
        fig.add_trace(go.Scatter3d(x=[fX(tPt[i])], y=[fY(tPt[i])], z=[fZ(tPt[i])], mode='markers',name="Intersect"))
fig.add_trace(go.Scatter3d(x=[fX(0)], y=[fY(0)], z=[fZ(0)], mode='markers',name="Start"))

#perpendicular triangle
xP= -1.0*np.ones((10))
yP= np.linspace(-10, 10, 10)
zP= np.linspace(-5, 5, 10)
#parallel triangle
#xP= np.linspace(-10, 10, 10)
#yP= np.linspace(-10, 10, 10)
#zP= np.zeros((10))

rc = np.random.rand(10,10)  # random surface colors
rc = np.reshape(['#aa9ce2']*100, (10,10))


cs = [[0, '#aa9ce2'],
      [5, '#aa9ce2']]

fig.add_trace(go.Surface(x=xP, y=yP, z=np.array([zP]*len(xP)), surfacecolor=rc, opacity=0.5))
fig.show()

print("time to intersection #0 = {:f}".format(tPt[0]))
print("distance to intersection #0= {:f}".format(tPt[0] * vPar))
print("x at intersection #0 = {:f}".format(fX(tPt[0])))
print("y at intersection #0 = {:f}".format(fY(tPt[0])))
print("z at intersection #0 = {:f}".format(fZ(tPt[0])))


#test for intersection - DOES NOT WORK!
#i=0
#pt = np.array([fX(tPt)[i], fY(tPt)[i], fZ(tPt)[i]])
#shadowMask = insideTriangle(pt,p1,p2,p3) #THIS DOESNT WORK.  FIX IT!
