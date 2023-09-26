import numpy as np
import plotly.graph_objects as go
import polyClass
import bezierClass

polyMask = True
bezierMask = False

torCurve = True
polCurve = False

#lq in mm
lq = 80.0
#number of curves
N = 5
#q||0
q0 = 6000.0 #MW/m2
#gap between lcfs and tile surface [mm]
gap = 0.0

if torCurve == True:
    topPt = np.array([0.0, 6.0])
    btmPt = np.array([67.5, 0.0])
    x = np.linspace(topPt[0],btmPt[0],50)

if polCurve == True:
    topPt = np.array([0.0, 1.0])
    btmPt = np.array([13.42/2.0, 0.0])
    x = np.linspace(topPt[0],btmPt[0],50)


#================================================
#              Polynomials
#================================================
if polyMask == True:
    c0 = topPt[1]
    c2Max = -c0 / btmPt[0]**2
    c4Max = -c0 / btmPt[0]**4
    c2 = np.linspace(c2Max, 0, N)

    def c2Fromc4(c0, y, x, c4):
        return (y-c0-c4*x**4)/x**2

    def c4Fromc2(c0, y, x, c2):
        return (y-c0-c2*x**2)/x**4

    #use this to plot all solutions (some not zero intersecting)
    c4 = np.linspace(c4Max, 0, N)
    #use this to only plot zero intersection solutions
    #c4 = c4Fromc2(c0, 0.0, btmPt[0], c2)

    coeffs = np.zeros((N, 5))
    coeffs[:,0] = c0
    coeffs[:,2] = c2
    coeffs[:,4] = c4
    #c = coeffs[0,:]

    curves = []
    for i,c in enumerate(coeffs):
        p1 = polyClass.poly(c)
        p1.evalOnX(x)
        p1.localPhi(2480)
        p1.bdotn = np.sum(np.multiply(p1.norms, p1.phi), axis=1)
        p1.qPar = p1.qParallel(lq, gap)
        curves.append(p1)
    
    print(coeffs)


#================================================
#              Bezier Curves
#================================================
if bezierMask == True:
    scanVals = np.linspace(0,1,N)
    
    #points that define the block we will machine (part outline)
    # change x axis control point
    #basePts = np.array([[0.0,6.0], [np.nan, 6.0], [67.5, 0.0]])
    basePts = np.array([[0.0,topPt[1]], [np.nan, topPt[1]], [btmPt[0]*0.9, topPt[1]*0.67],  [btmPt[0], 0.0]])

    # change y axis control point
    #basePts = np.array([[0.0,6.0], [67.5, 6.0], [67.5, np.nan], [67.5, 0.0]])

    dX = np.nanmax(basePts[:,0])
    dY = np.nanmax(basePts[:,1])
    print(dX)
    print(dY)
    points = []
    for i,s in enumerate(scanVals):
        newPts = basePts.copy()
        newPts[1,0] = s*dX
        points.append(newPts)

    print(points)
    #points = np.array([[ [0,6], [10, 6.0], [40, 6.0], [67.5, 6], [67.5, 5],[67.5,0] ]])
    curves = []
    for i,p in enumerate(points):
        p1 = bezierClass.bezier(p)
        p1.evalOnX()
        p1.localPhi(2480)
        p1.bdotn = np.sum(np.multiply(p1.norms, p1.phi), axis=1)
        p1.qPar = p1.qParallel(lq)
        curves.append(p1)

#    fig = go.Figure()
#    fig.add_trace(go.Scatter(x=xvals, y=yvals))
#    fig.add_trace(go.Scatter(x=xpoints, y=ypoints, mode='markers'))
#    fig.show()



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

#for visualizing phi vectors
#p1.buildPhiEndPoints(5.0)
#for i,pt in enumerate(p1.endPtsPhi):
#    vecX = [p1.ctrs[i,0], p1.endPtsPhi[i,0]]
#    vecY = [p1.ctrs[i,1], p1.endPtsPhi[i,1]]
#    fig.add_trace(go.Scatter(x=vecX, y=vecY))

for i,p in enumerate(curves):
    #for visualizing curve
    fig.add_trace(go.Scatter(x=p.yArr[:,0], y=p.yArr[:,1], name="poly{:d}".format(i)))
    #if bezierMask==True:
    #    fig.add_trace(go.Scatter(x=p.pts[:,0], y=p.pts[:,1], mode='markers', marker={'color':'black'}))
    #for visualizing bdotn
    fig1.add_trace(go.Scatter(x=p.yArr[:,0], y=p.bdotn, name="bdotn{:d}".format(i)))
    #for visualizing q(psi)
    fig2.add_trace(go.Scatter(x=p.yArr[:,0], y=p.qPar, name="qPar{:d}".format(i)))
    #for visualizing HF (qPar * bdotn)
    fig3.add_trace(go.Scatter(x=p.yArr[:,0], y=p.qPar*p.bdotn, name="qDiv{:d}".format(i)))
    fig3.update_yaxes(range=[-0.5, 0])
#fig.update_yaxes(scaleanchor = "x",scaleratio = 1,)

fig.update_xaxes(title='[mm]')
fig1.update_xaxes(title='[mm]')
fig2.update_xaxes(title='[mm]')
fig3.update_xaxes(title='[mm]')
fig.update_yaxes(title='[mm]')
fig1.update_yaxes(title='bdotn')
fig2.update_yaxes(title='qParallel_normalized')
fig3.update_yaxes(title='qDiv_normalized')

fig.show()
#fig1.show()
#fig2.show()
fig3.show()