#Description:   generates user specified number of polynomials 
#               for top surface PFC shape. then plots HF, bdotn, 
#               psi, on the top surface for use in top surface 
#               optimization workflows
#               plots full top surface
#Eningeer: T Looby
#Date: 20230928
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import polyClass
import bezierClass

#================================================
#              User Inputs
#================================================
#polynomial or bezier curves
polyMask = False
if polyMask == False:
    bezierMask = True
else:
    bezierMask = False

#calculate shadowing
shadowMask = True
#save maxHF csv
saveCSVmaxHF = True
saveCSVmaxHFlq = False
maxHFcsv = '/home/tlooby/projects/dummy/hfMax.csv'
#tile width [mm]
w = 135.0
#gap between tiles [mm]
tileGap = 1.0
#maximum height [mm]
h = 6.0
#we assume y co-aligns to radial direction at PFC ctr.
leftPt = np.array([-w/2.0, 0.0])
ctrPt = np.array([0.0, h])
rightPt = np.array([w/2.0, 0.0])
#number of points in curve
Nx = 100
x = np.linspace(leftPt[0],rightPt[0],Nx)
#number of polynomials to analyze
N = 5
#angle of incidence [degrees]
a = 1.0
alpha = np.radians(a)
#toroidal direction of b (+ means left to right) 
bDir = 1.0
#inner or outer limiter
limType = 'outerLim'
#limType = 'innerLim'
#lq in mm
lq = 0.6
#get between wall and lcfs
gapLCFS = 0.0


#for only showing specific curve #s (set to all false then true the ones you want)
#all
polyNmask = [True]*N
#user specified 
#polyNmask = [False]*N
#polyNmask[3] = True

#================================================
#              Polynomials
#================================================
if polyMask == True:
    c0 = ctrPt[1]
    c2Max = -c0 / rightPt[0]**2
    c4Max = -c0 / rightPt[0]**4
    c2 = np.linspace(c2Max, 0, N)

    def c2Fromc4(c0, y, x, c4):
        return (y-c0-c4*x**4)/x**2

    def c4Fromc2(c0, y, x, c2):
        return (y-c0-c2*x**2)/x**4

    #use this to plot all solutions (some not zero intersecting)
    #c4 = np.linspace(c4Max, 0, N)
    #use this to only plot zero intersection solutions
    c4 = c4Fromc2(c0, 0.0, rightPt[0], c2)

    coeffs = np.zeros((N, 5))
    coeffs[:,0] = c0
    coeffs[:,2] = c2
    coeffs[:,4] = c4
    #c = coeffs[0,:]

    curves = []
    for i,c in enumerate(coeffs):
        p1 = polyClass.poly(c)
        p1.evalOnX(x)
        p1.localPhi(2480, limType, alpha, bDir)
        p1.bdotn = np.sum(np.multiply(p1.norms, p1.phi), axis=1)
        p1.qPar = p1.qParallel(lq, gapLCFS)
        curves.append(p1)

    #print(coeffs)

#================================================
#              Bezier Curves
#================================================
if bezierMask == True:
    scanVals = np.linspace(0,1,N)
    
    #points that define the block we will machine (part outline)
    # change x axis control point
    #basePts = np.array([[0.0,6.0], [np.nan, 6.0], [67.5, 0.0]])
    basePts = np.array([[0.0,ctrPt[1]], [np.nan, ctrPt[1]], [rightPt[0]*0.9, ctrPt[1]*0.8],  [rightPt[0], 0.0]])
    print(basePts)
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
        p1.localPhi(2480, limType, alpha, bDir)
        p1.bdotn = np.sum(np.multiply(p1.norms, p1.phi), axis=1)
        p1.qPar = p1.qParallel(lq)
        curves.append(p1)



#================================================
#              Shadowing
#================================================
HFarray = []
shadowEnds = []
shadowIdx = []
if shadowMask == True:
    for i,p in enumerate(curves):
        if polyNmask[i] == False:
            continue
        #find downstream shadow edge
        sign_changes = np.where(np.diff(np.sign(p.bdotn)))[0]
        idx1 = sign_changes[0] + 1 if len(sign_changes) > 0 else None
        lineSeg = np.array([p.yArr[idx1], p.yArr[idx1]+p.phi[idx1]*w])       
        #build poly model of downstream tile
        arr = p.yArr.copy()
        arr[:,0] += tileGap + w
        #find upstream shadow edge
        idx0 = p.find_intersection(arr, lineSeg)
        #identify portion of tile surface that is loaded
        p.hot = np.arange(Nx)[idx0:idx1]
        p.cold = np.setdiff1d(np.arange(Nx), p.hot)
        p.maxHF = max( np.abs(p.qPar[p.hot]*p.bdotn[p.hot]) )
        p.idx0 = idx0
        p.idx1 = idx1
        HFarray.append( p.maxHF )


#save max HF line in csv file with alpha data
if saveCSVmaxHF==True:
    with open(maxHFcsv,'a') as fd:
        row = '{:0.3f},'.format(a) + ','.join([str(np.round(s, 6)) for s in HFarray]) + '\n'
        fd.write(row)

#save max HF line in csv file with lq data
if saveCSVmaxHFlq==True:
    with open(maxHFcsv,'a') as fd:
        row = '{:0.3f},'.format(lq) + ','.join([str(np.round(s, 6)) for s in HFarray]) + '\n'
        fd.write(row)


#================================================
#              Plots
#================================================
fig = go.Figure()
fig1 = go.Figure()
fig2 = go.Figure()
fig3 = go.Figure()
fig4 = go.Figure()


color = px.colors.qualitative.Plotly
for i,p in enumerate(curves):
    if polyNmask[i] == False:
        continue
    #for visualizing curve
    if shadowMask == True:
        fig.add_trace(go.Scatter(x=p.yArr[p.hot,0], y=p.yArr[p.hot,1],
                                 line=dict(color=color[i]),
                                 mode = 'lines+markers',
                                 name="poly{:d} loaded".format(i)))
        fig.add_trace(go.Scatter(x=p.yArr[:,0], y=p.yArr[:,1], 
                                 line=dict(color=color[i], dash='dash'),
                                 mode = 'lines',
                                 name="poly{:d} shadow".format(i)))
    else:
        fig.add_trace(go.Scatter(x=p.yArr[:,0], y=p.yArr[:,1], name="curve{:d}".format(i)))

    #for visualizing bdotn
    fig1.add_trace(go.Scatter(x=p.yArr[:,0], y=p.bdotn, name="bdotn{:d}".format(i)))


    #for visualizing HF (qPar * bdotn)
    if shadowMask == True:
        fig3.add_trace(go.Scatter(x=p.yArr[p.hot,0], y=np.abs(p.qPar[p.hot]*p.bdotn[p.hot]), mode='lines', name="qDiv{:d}".format(i)))
    else:
        fig3.add_trace(go.Scatter(x=p.yArr[:,0], y=np.abs(p.qPar*p.bdotn), name="qDiv{:d}".format(i)))


##for visualizing phi vectors on surface plot
#p1.buildPhiEndPoints(5.0)
#for i,pt in enumerate(p1.endPtsPhi):
#    vecX = [p1.ctrs[i,0], p1.endPtsPhi[i,0]]
#    vecY = [p1.ctrs[i,1], p1.endPtsPhi[i,1]]
#    fig.add_trace(go.Scatter(x=vecX, y=vecY))




fig.update_xaxes(title='[mm]')
fig.update_yaxes(title='[mm]')
fig1.update_xaxes(title='[mm]')
fig1.update_yaxes(title='bdotn')
fig3.update_xaxes(title='[mm]')
fig3.update_yaxes(title='qDiv_normalized')
#fig.update_yaxes(scaleanchor = "x",scaleratio = 1,)

#top surface shape
fig.show()
#bdotn plot
#fig1.show()
#hf plot
#fig3.show()
