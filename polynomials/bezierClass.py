import numpy as np
from scipy.special import comb


class bezier:
    """
    bezier curve object and associated functions
    """
    def __init__(self, points):
        self.pts = points
        return

    def bernstein_poly(self, i, n, t):
        """
         The Bernstein polynomial of n, i as a function of t
        """
        return comb(n, i) * ( t**(n-i) ) * (1 - t)**i

    def bezier_curve(self, points, nTimes=100):
        """
           Given a set of control points, return the
           bezier curve defined by the control points.

           points should be a list of lists, or list of tuples
           such as [ [1,1], 
                     [2,3], 
                     [4,5], ..[Xn, Yn] ]
            nTimes is the number of time steps, defaults to 1000

            See http://processingjs.nihongoresources.com/bezierinfo/
        """

        nPoints = len(points)
        xPoints = np.array([p[0] for p in points])
        yPoints = np.array([p[1] for p in points])

        t = np.linspace(0.0, 1.0, nTimes)

        polynomial_array = np.array([ self.bernstein_poly(i, nPoints-1, t) for i in range(0, nPoints)   ])

        xvals = np.dot(xPoints, polynomial_array)
        yvals = np.dot(yPoints, polynomial_array)

        return xvals, yvals
  

    def evalOnX(self, Nx, fullTile=True, rotation=0.0):
        """
        evaluates bezier, derivative, ctrs, normals, for user defined x values
        ctrs and norms are of dimension len(x)-1

        fullTile is a boolean that indicates if we are doing a half or full tile
        if your control points correspond to half a tile then set to True
        """
        if fullTile==True:
            m = 2.0
        else:
            m=1.0
        x1, y1 = self.bezier_curve(self.pts, int(Nx/m))

        x = np.flip(x1)
        y = np.flip(y1)

        if fullTile == True:
            x = np.unique(np.concatenate([-1.0*np.flip(x), x]))
            y = np.concatenate([np.flip(y[1:]), y])

        dy = np.diff(y) / np.diff(x)
        dy = np.insert(dy, 0, 0)

        #build arrays
        self.yArr = np.vstack([x,y]).T
        self.dyArr = np.vstack([x,dy]).T

        #calculate norms, ctrs
        self.ctrs = self.centers(self.yArr)
        self.norms = np.zeros((self.ctrs.shape))
        normTmp = self.normals(self.yArr)
        self.norms[:,0] = normTmp[:,0]
        self.norms[:,1] = normTmp[:,1]        


        #rotate norms if necessary
        # Rotation matrix
        if rotation != 0.0:
            rotation_matrix = np.array([
                [np.cos(rotation), -np.sin(rotation)],
                [np.sin(rotation), np.cos(rotation)]
            ])
            self.norms = self.norms @ rotation_matrix.T  # Matrix multiplication
        return
    

    def normals(self, rawdata):
        N = len(rawdata) - 1
        norms = np.zeros((N,3))
        for i in range(N):
            RZvec = rawdata[i+1] - rawdata[i]
            vec1 = np.array([[RZvec[0], RZvec[1], 0.0]])
            vec2 = np.array([[0.0, 0.0, 1.0]])
            n = np.cross(vec1, vec2)
            norms[i,:] = n / np.linalg.norm(n,axis=1)
        return norms

    def buildNormalEndPoints(self, mag):
        """
        builds an array of endpoints for normal vectors (for plotting)
        """
        self.endPts = self.ctrs + self.norms*mag
        return

    def buildPhiEndPoints(self, mag):
        """
        builds an array of endpoints for normal vectors (for plotting)
        """
        self.endPtsPhi = self.ctrs + self.phi*mag
        return

    def centers(self, rz):
        centers = np.zeros((len(rz)-1, 2))
        dR = np.diff(rz[:,0])
        dZ = np.diff(rz[:,1])
        centers[:,0] = rz[:-1,0] + dR/2.0
        centers[:,1] = rz[:-1,1] + dZ/2.0
        return centers
    
    def localPhi(self, R0, mode, alpha=0.0, bDir=1.0):
        """
        calculate the local phi vector at each ctr point
        given a radius of R0 at the polynomial apex
        mode is outerLim or innerLim

        x coordinate corresponds to distance along poly
        y coordinate corresponds to radial coordinate
        z coordinate is orthogonal
        """
        rVec = np.zeros((len(self.ctrs), 3))
        if mode == 'innerLim':
            rVec[:,1] = (self.pts[0,1] - self.ctrs[:,1]) + R0
        else:
            rVec[:,1] = R0 - (self.pts[0,1] - self.ctrs[:,1])
        rVec[:,0] = self.ctrs[:,0]
        rMag = np.linalg.norm(rVec, axis=1)
        rVec[:,0] = rVec[:,0] / rMag
        rVec[:,1] = rVec[:,1] / rMag

        zVec = np.array([0.0, 0.0, -1.0])
        phi = np.cross(rVec, zVec)
        phi[:,0]*=-1.0*bDir #use bDir to flip toroidal coordinate

        #add alpha to phi
        a1 = np.arctan2(phi[:,1], phi[:,0]) - alpha
        self.phi = np.zeros((self.ctrs.shape))
        self.phi[:,1] = np.tan(a1)
        self.phi[:,0] = bDir
        phiMag = np.linalg.norm(self.phi, axis=1)
        self.phi = self.phi / phiMag[:,np.newaxis]
        return

    def localPhi0(self, R0):
        """
        calculate the local phi vector at each ctr point
        given a radius of R0 at the polynomial apex
        """
        rVec = np.zeros((len(self.ctrs), 3))
        rVec[:,0] = self.pts[0,1] - self.ctrs[:,1] + R0
        rVec[:,1] = self.ctrs[:,0]
        rMag = np.linalg.norm(rVec, axis=1)
        rVec[:,0] = rVec[:,0] / rMag
        rVec[:,1] = rVec[:,1] / rMag

        zVec = np.array([0.0, 0.0, 1.0])
        phi = np.cross(rVec, zVec)

        self.phi = np.zeros((self.ctrs.shape))
        self.phi[:,0] = phi[:,1]
        self.phi[:,1] = -phi[:,0]
        return
    
    def qParallel(self, lq, gap=0.0):
        """
        calculates q|| given lq, the decay length for an exponential profile
        assumes separatrix is on tile apex unless a gap is specified in mm
        """
        r = self.pts[0,1] - self.ctrs[:,1] + gap
        q = np.exp(-r / lq)

        return q

    def calculateShadow2(self, alpha, x, y):
        """
        takes an angle of incidence, alpha, and calculates the width of the tile
        top surface that is loaded, and the corresponding x coordinate
        where the shadow begins (location of last shadow) x_tangent.  dependent
        upon tile half width, w, and gap size, g 
        """
        #calculate the derivative of the profile, y
        dydx = np.diff(y) / np.diff(x)
        dydx = np.insert(dydx, 0, 0)

        idx = np.argmin( np.abs(dydx + np.tan(alpha)) )

        self.x_tangent = x[idx]

        print("X location of last shadow (#2): {:f} [mm]".format(self.x_tangent))
        return     

    def ccw(self, A, B, C):
        """Check the orientation of the triplet (A, B, C)"""
        return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

    def do_intersect(self, A, B, C, D):
        """Check if line segments AB and CD intersect"""
        return (self.ccw(A, C, D) != self.ccw(B, C, D)) and (self.ccw(A, B, C) != self.ccw(A, B, D))

    def find_intersection(self, curve_points, line_segment):
        """Check if the line segment intersects the curve.

        Args:
            curve_points: List of points representing the curve.
            line_segment: List of two points representing the line segment.

        Returns:
            Index of the first point of the curve segment that was intersected.
            Returns None if there's no intersection.
        """
        for i in range(len(curve_points) - 1):
            if self.do_intersect(curve_points[i], curve_points[i+1], line_segment[0], line_segment[1]):
                return i
        return None
