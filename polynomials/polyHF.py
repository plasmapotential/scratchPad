import numpy as np
import plotly.graph_objects as go
import polyClass


topPt = np.array([0.0, 6.0])
btmPt = np.array([67.5, 0.0])
x = np.linspace(topPt[0],btmPt[0],50)



#number of polys
N = 5
#lq in mm
lq = 3.0

c0 = 6.0
c2Max = -c0 / btmPt[0]**2
c4Max = -c0 / btmPt[0]**4
c2 = np.linspace(c2Max, 0, N)

def c2Fromc4(c0, y, x, c4):
    return (y-c0-c4*x**4)/x**2

def c4Fromc2(c0, y, x, c2):
    return (y-c0-c2*x**2)/x**4







#use this to plot all solutions (some not zero intersecting)
#c4 = np.linspace(c4Max, 0, N)
#use this to only plot zero intersection solutions
c4 = c4Fromc2(c0, 0.0, 67.5, c2)

coeffs = np.zeros((N, 5))
coeffs[:,0] = c0
coeffs[:,2] = c2
coeffs[:,4] = c4
#c = coeffs[0,:]

print(coeffs[2,:])


polys = []
for i,c in enumerate(coeffs):
    p1 = polyClass.poly(c)
    p1.evalOnX(x)
    p1.localPhi(2480)
    p1.bdotn = np.sum(np.multiply(p1.norms, p1.phi), axis=1)
    p1.qPar = p1.qParallel(lq)
    polys.append(p1)


#idx = 50
#ptX = x[idx]
#ptY = p(ptX)
#dpPt = dp(ptX)
#dx = 5.0
#pt2X = ptX + dx
#pt2Y = ptY + dpPt*dx
#fig.add_trace(go.Scatter(x=[ptX, pt2X], y=[ptY, pt2Y]))


fig = go.Figure()
fig1 = go.Figure()
fig2 = go.Figure()
fig3 = go.Figure()

#for visualizing normal vectors
#p1.buildNormalEndPoints(5.0)
#for i,pt in enumerate(p1.endPts):
#    vecX = [p1.ctrs[i,0], p1.endPts[i,0]]
#    vecY = [p1.ctrs[i,1], p1.endPts[i,1]]
#    fig.add_trace(go.Scatter(x=vecX, y=vecY))
#
#for visualizing phi vectors
#p1.buildPhiEndPoints(5.0)
#for i,pt in enumerate(p1.endPtsPhi):
#    vecX = [p1.ctrs[i,0], p1.endPtsPhi[i,0]]
#    vecY = [p1.ctrs[i,1], p1.endPtsPhi[i,1]]
#    fig.add_trace(go.Scatter(x=vecX, y=vecY))


for i,p in enumerate(polys):
    #for visualizing polynomial
    fig.add_trace(go.Scatter(x=x, y=p.yArr[:,1], name="poly{:d}".format(i)))
    #for visualizing bdotn
    fig1.add_trace(go.Scatter(x=x, y=p.bdotn, name="bdotn{:d}".format(i)))
    #for visualizing q(psi)
    fig2.add_trace(go.Scatter(x=x, y=p.qPar, name="qPar{:d}".format(i)))
    #for visualizing HF (qPar * bdotn)
    fig3.add_trace(go.Scatter(x=x, y=p.qPar*p.bdotn, name="qDiv{:d}".format(i)))

#fig.update_yaxes(scaleanchor = "x",scaleratio = 1,)
fig.show()
fig1.show()
fig2.show()
fig3.show()


