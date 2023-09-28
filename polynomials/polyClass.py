#Description:   class that defines a polynomial.  for use in workflows comparing
#               bidirectional top surface shaping polynomials
#Engineer:      T Looby
#Date:          20230926

import numpy as np

class poly:
    """
    polynomial object and associated functions
    """
    def __init__(self, coeffs):
        self.c = coeffs
        self.dc = self.polyDerivative(self.c)
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
        """
        calculates normals
        """
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
            rVec[:,1] = (self.c[0] - self.ctrs[:,1]) + R0
        else:
            rVec[:,1] = R0 - (self.c[0] - self.ctrs[:,1])
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
    
    def qParallel(self, lq, gap=0.0):
        """
        calculates q|| given lq, the decay length for an exponential profile
        assumes separatrix is on tile apex unless a gap is specified in mm
        """
        #r = self.c[0] - self.ctrs[:,1] + gap
        r = self.c[0] - self.yArr[:,1] + gap
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


    def calculateShadow(self, alpha, w, g):
        """
        takes an angle of incidence, alpha, and calculates the width of the tile
        top surface that is loaded, and the corresponding x coordinate
        where the shadow begins (location of last shadow) x_tangent.  dependent
        upon tile half width, w, and gap size, g

        alpha:  angle of incidence [radians]
        x_tangent: x coordinate of last shadow [mm]
        """
        #slope of field line (downward into tile surface)
        m = -1.0 * np.tan(alpha)
        fB0 = np.poly1d([m, 0.0])

        #we define x=0 to be the location of the middle of the tile (the apex)
        #find y0, the y intercept of a field line that strikes the downstream
        #PFCs leading edge corner
        y0 = -1.0 * fB0(w+g)

        #build a polynomial for this field line
        B0_coeffs = np.array([m])
        B_order = 0 #linear

        #find the x location of the last shadow on the upstream PFC, x_tangent
        fsMinusfB = self.dc.copy()
        fsMinusfB[-(B_order+1):] -= B0_coeffs
        rts = np.roots(fsMinusfB)
        use = np.where(np.logical_and(rts.real>0.0, rts.imag == 0.0))[0]

        print(fsMinusfB)

        if len(use) > 0:
            x_tangent = float(rts[use].real)
        else:
            x_tangent = w

        #find the y-intercept for the field line that goes through x_tangent
        y1 = self.p(x_tangent) - fB0(x_tangent)

        #evaluate the elevation that the downstream PFC would need to move 
        # for the leading edge corner to be loaded
        B1_coeffs = np.array([m, y1])
        fB1 = np.poly1d(B1_coeffs)
        delta_h = fB1(g+w)


        print("X location of last shadow: {:f} [mm]".format(x_tangent))
        print("y-axis intercept for tangent field line: {:f} [mm]".format(y1))
        print("Maximum elevation of tile to remain shadowed: {:f} [mm]".format(delta_h))


        self.x_tangent = x_tangent
        self.delta_h = delta_h
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


