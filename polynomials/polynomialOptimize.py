#
import numpy as np
import plotly.graph_objects as go


class poly:
    """
    polynomial object and associated functions
    """
    def __init__(self, coeffs):
        self.c = coeffs
        self.dc = self.polyDerivative(c)
        self.p = np.polynomial.Polynomial(self.c, domain=None, window=None)
        self.dp = np.polynomial.Polynomial(self.dc, domain=None, window=None)
        return

    def polyDerivative(self, c):
        """
        calculates coefficients of derivative of polynomial with coefficients, c
        """
        dcdx = np.array([c[i] * i for i in range(1, len(c))])
        return dcdx
    
    def evaluateOnX(self):
        return
    
    def evalOnX(self, x):
        """
        evaluates polynomial, derivative, ctrs, normals, for user defined x values
        ctrs and norms are of dimension len(x)-1
        """
        y = self.p(x)
        dy = self.dp(x)

        #build arrays
        self.yArr = np.vstack([x,y]).T
        self.dyArr = np.vstack([x,dy]).T

        #calculate norms, ctrs
        self.ctrs = self.centers(self.yArr)
        self.norms = np.zeros((self.ctrs.shape))
        normTmp = self.normals(self.yArr)
        self.norms[:,0] = normTmp[:,0]
        self.norms[:,1] = normTmp[:,1]        

        return
    

    def normals(self, rawdata):
        N = len(rawdata) - 1
        norms = np.zeros((N,3))
        for i in range(N):
            RZvec = rawdata[i+1] - rawdata[i]
            vec1 = np.array([[RZvec[0], RZvec[1], 0.0]])
            vec2 = np.array([[0.0, 0.0, -1.0]])
            n = np.cross(vec1, vec2)
            norms[i,:] = n / np.linalg.norm(n,axis=1)
        return norms

    def buildNormalEndPoints(self, mag):
        """
        builds an array of endpoints for normal vectors (for plotting)
        """
        self.endPts = self.ctrs + self.norms*mag
        return

    def centers(self, rz):
        centers = np.zeros((len(rz)-1, 2))
        dR = np.diff(rz[:,0])
        dZ = np.diff(rz[:,1])
        centers[:,0] = rz[:-1,0] + dR/2.0
        centers[:,1] = rz[:-1,1] + dZ/2.0
        return centers

        






topPt = np.array([0.0, 6.0])
btmPt = np.array([67.5, 0.0])
x = np.linspace(topPt[0],btmPt[0],50)

#define flux vector
qVec = np.array([0.0, 1.0, 0.0])



#number of polys
N = 1

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
c = coeffs[0,:]


p1 = poly(c)
p1.evalOnX(x)
p1.buildNormalEndPoints(5.0)

#idx = 50
#ptX = x[idx]
#ptY = p(ptX)
#dpPt = dp(ptX)
#dx = 5.0
#pt2X = ptX + dx
#pt2Y = ptY + dpPt*dx
#fig.add_trace(go.Scatter(x=[ptX, pt2X], y=[ptY, pt2Y]))


fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=p1.yArr[:,1]))

#for visualizing normal vectors
for i,pt in enumerate(p1.endPts):
    vecX = [p1.ctrs[i,0], p1.endPts[i,0]]
    vecY = [p1.ctrs[i,1], p1.endPts[i,1]]
    fig.add_trace(go.Scatter(x=vecX, y=vecY))


fig.update_yaxes(scaleanchor = "x",scaleratio = 1,)
fig.show()


