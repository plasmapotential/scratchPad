#Description:   generates user specified number of polynomials 
#               for top surface PFC shape. then plots HF, bdotn, 
#               psi, on the top surface for use in top surface 
#               optimization workflows
#Eningeer: T Looby
#Date: 20230926
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import polyClass
import bezierClass


#================================================
#              User Inputs
#================================================
#polynomial or bezier curves
polyMask = True
if polyMask == False:
    bezierMask = True
else:
    bezierMask = False

#toroidal or poloidal contour
torCurve = True
polCurve = False

#should we calculate shadows
shadowMask = False
maxHFcsv = '/home/tom/work/CFS/projects/dummy/hfMax.csv'

#inner or outer limiter
#limType = 'outerLim'
limType = 'innerLim'

#lq in mm
lq = 0.6
#number of curves
N = 1
#q||0
q0 = 6000.0 #MW/m2
#gap between lcfs and tile surface [mm]
gap = 0.0
#gap between neighboring tiles [mm]
g = 0.5
#half width of tile [mm]
if torCurve == True:
    w = 67.5 #toroidal
else:
    w = 13.42/2.0 #poloidal
#angle of incidence [degrees]
a = 10.0
alpha = np.radians(a)
#toroidal direction of power flow
pwrDir = -1.0
#number of points in curve
Nx = 50


if torCurve == True:
    topPt = np.array([0.0, 6.0]) #apex
    btmPt = np.array([w, 0.0]) #end point
    x = np.linspace(topPt[0],btmPt[0],Nx)

if polCurve == True:
    topPt = np.array([0.0, 1.0])
    btmPt = np.array([w, 0.0])
    x = np.linspace(topPt[0],btmPt[0],Nx)


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
    #c4 = np.linspace(c4Max, 0, N)
    #use this to only plot zero intersection solutions
    c4 = c4Fromc2(c0, 0.0, btmPt[0], c2)

    coeffs = np.zeros((N, 5))
    coeffs[:,0] = c0
    coeffs[:,2] = c2
    coeffs[:,4] = c4
    #c = coeffs[0,:]

    curves = []
    for i,c in enumerate(coeffs):
        p1 = polyClass.poly(c)
        p1.evalOnX(x)
        p1.localPhi(2480, limType, alpha, pwrDir)
        print(p1.phi)
        p1.bdotn = np.sum(np.multiply(p1.norms, p1.phi), axis=1)
        p1.qPar = p1.qParallel(lq, gap)
        curves.append(p1)
    
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
    points = []
    for i,s in enumerate(scanVals):
        newPts = basePts.copy()
        newPts[1,0] = s*dX
        points.append(newPts)

    #points = np.array([[ [0,6], [10, 6.0], [40, 6.0], [67.5, 6], [67.5, 5],[67.5,0] ]])
    curves = []
    for i,p in enumerate(points):
        p1 = bezierClass.bezier(p)
        p1.evalOnX(Nx)
        p1.localPhi(2480, limType, alpha)
        p1.bdotn = np.sum(np.multiply(p1.norms, p1.phi), axis=1)
        p1.qPar = p1.qParallel(lq)
        curves.append(p1)

#    fig = go.Figure()
#    fig.add_trace(go.Scatter(x=xvals, y=yvals))
#    fig.add_trace(go.Scatter(x=xpoints, y=ypoints, mode='markers'))
#    fig.show()



#================================================
#              Shadowing
#================================================
HFarray = []
shadowEnds = []
shadowIdx = []
if shadowMask == True:
    for i,p in enumerate(curves):
        #p.calculateShadow(alpha, w, g)
        p.calculateShadow2(alpha, p.yArr[:,0], p.yArr[:,1])
        hot = np.where(p.ctrs[:,0]<p.x_tangent)[0]
        maxHF = max( np.abs(p.qPar[hot]*p.bdotn[hot]) )
        HFarray.append( maxHF )
        shadowEnds.append(p.yArr[hot[-1]])
        shadowIdx.append(hot[-1])

#save max HF line in csv file
with open(maxHFcsv,'a') as fd:
    row = '{:0.3f},'.format(a) + ','.join([str(np.round(s, 6)) for s in HFarray]) + '\n'
    fd.write(row)


#================================================
#              Plots
#================================================

fig = go.Figure()
fig1 = go.Figure()
fig2 = go.Figure()
fig3 = go.Figure()
fig4 = go.Figure()
#for visualizing normal vectors
#p1.buildNormalEndPoints(5.0)
#for i,pt in enumerate(p1.endPts):
#    vecX = [p1.ctrs[i,0], p1.endPts[i,0]]
#    vecY = [p1.ctrs[i,1], p1.endPts[i,1]]
#    fig.add_trace(go.Scatter(x=vecX, y=vecY))

#for visualizing phi vectors
p1.buildPhiEndPoints(5.0)
for i,pt in enumerate(p1.endPtsPhi):
    vecX = [p1.ctrs[i,0], p1.endPtsPhi[i,0]]
    vecY = [p1.ctrs[i,1], p1.endPtsPhi[i,1]]
    fig.add_trace(go.Scatter(x=vecX, y=vecY))

#for visualizing alpha vectors
if shadowMask == True:
    length = 5.0
    for i,p in enumerate(curves):
        pt = shadowEnds[i]
        idx = shadowIdx[i]
        norm = np.array([0.0, 0.0, 0.0])
        norm[0] = p.norms[idx,0]
        norm[1] = p.norms[idx,1]
        phi = np.array([0.0, 0.0, 1.0])
        tangents = np.cross(norm, phi)*length
        xVec = [p.ctrs[idx,0], p.ctrs[idx,0]+tangents[0]]
        yVec = [p.ctrs[idx,1], p.ctrs[idx,1]+tangents[1]]
        fig.add_trace(go.Scatter(x=xVec, y=yVec))




color = px.colors.qualitative.Plotly
for i,p in enumerate(curves):

    #for visualizing curve
    if shadowMask == True:
        hot = np.where(p.yArr[:,0]<p.x_tangent)[0]
        cold = np.where(p.yArr[:,0]>p.x_tangent)[0]
        fig.add_trace(go.Scatter(x=p.yArr[hot,0], y=p.yArr[hot,1],
                                 line=dict(color=color[i]),
                                 mode = 'lines+markers',
                                 name="poly{:d} loaded".format(i)))
        fig.add_trace(go.Scatter(x=p.yArr[cold,0], y=p.yArr[cold,1], 
                                 line=dict(color=color[i], dash='dash'),
                                 mode = 'lines',
                                 name="poly{:d} shadow".format(i)))
    else:
        fig.add_trace(go.Scatter(x=p.yArr[:,0], y=p.yArr[:,1], name="poly{:d}".format(i)))
        #if bezierMask==True:
    #       fig.add_trace(go.Scatter(x=p.pts[:,0], y=p.pts[:,1], mode='markers', marker={'color':'black'}))
    #for visualizing bdotn
    fig1.add_trace(go.Scatter(x=p.yArr[:,0], y=p.bdotn, name="bdotn{:d}".format(i)))
    #for visualizing q(psi)
    fig2.add_trace(go.Scatter(x=p.yArr[:,0], y=p.qPar, name="qPar{:d}".format(i)))
    #for visualizing HF (qPar * bdotn)
    if shadowMask == True:
        hot = np.where(p.ctrs[:,0]<p.x_tangent)[0]
        fig3.add_trace(go.Scatter(x=p.ctrs[hot,0], y=p.qPar[hot]*p.bdotn[hot], mode='lines', name="qDiv{:d}".format(i)))
    else:
        fig3.add_trace(go.Scatter(x=p.yArr[:,0], y=p.qPar*p.bdotn, name="qDiv{:d}".format(i)))
    #fig3.update_yaxes(range=[-0.1, 0])

#max HF plot
fig4.add_trace(go.Scatter(x=np.arange(N)+1, y=HFarray, mode='lines+markers', name="Max Surface Flux"))



fig.update_yaxes(scaleanchor = "x",scaleratio = 1,)

fig.update_xaxes(title='[mm]')
fig1.update_xaxes(title='[mm]')
fig2.update_xaxes(title='[mm]')
fig3.update_xaxes(title='[mm]')
fig.update_yaxes(title='[mm]')
fig1.update_yaxes(title='bdotn')
fig2.update_yaxes(title='qParallel_normalized')
fig3.update_yaxes(title='qDiv_normalized')
fig4.update_xaxes(title='Polynomial #')
fig4.update_yaxes(title='Max HF Fraction of q0')

fig.show()
#bdotn plot
fig1.show()
#normalized q plot
#fig2.show()
#HF along contour plot
fig3.show()
#max HF plot
#fig4.show()


