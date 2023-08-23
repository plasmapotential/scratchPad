#enhancementFactor.py
#Description:   Analytically calculates enhancement factor for multiple mode number fish-scale
#               assumes fish-scale is poloidally uniform
#Engineer:      T Looby
#Date:          20230403
import numpy as np
import plotly.graph_objects as go
#=== User inputs =====
#fish-scale heights T5B
#tileName = 'T5B'
#h_s = 58.0e-6 # [m], fish-scale height
#h_c = 232.0e-6 #[m]
#h_p = 579.0e-6 #[m]
#R_sp = 1.72 #[m]
#aoi = 4.31 #degrees
aoi = 9.0 #degrees

#T4
tileName = 'T4'
h_s = 138e-6
h_c = 440e-6
h_p = 1175e-6
R_sp = 1.66474 #[m]
#aoi = 1.02 #[degrees]
aoi = 0.3 #[degrees]
aoiDesign = 2.0 #degrees
#angle of incidence

alpha = np.radians(aoi)
alphaDesign = np.radians(aoiDesign)
#======================






class tileClass:
    def __init__(self):
        return
    
    def setupTile(self, name:str, R_sp:float):
        """
        sets up tile object based upon user supplied name

        gaps provided below in [m]
        """
        if name=='T5B':
            self.g_s = 500e-6 #[m]
            self.g_c = 2000e-6 #[m]
            self.g_p = 5000e-6 #[m]
            self.sMode = 720
            self.cMode = 144
            self.pMode = 36
            self.Rctr = 1.76 #[m]
        elif name=='T4':
            self.g_s = 500e-6 #[m]
            self.g_c = 2000e-6 #[m]
            self.g_p = 5000e-6 #[m]
            self.sMode = 720
            self.cMode = 144
            self.pMode = 36
            self.Rctr = R_sp
        
        
        
        self.g_tot = self.g_s*self.sMode + self.g_c*self.cMode + self.g_p*self.pMode

        return

    def setupHeight(self, h_s, h_c, h_p):
        """
        sets up height for slice, carrier, pedestal

        heights should be provided in [m]
        """
        self.h_s = h_s
        self.h_c = h_c
        self.h_p = h_p
        return

    def calculateWidths(self, w_s=None, w_c=None, w_p=None):
        """
        sets up the widths for each mode number
        """
        #fish-scale + carrier + pedestal
        if w_p==None:
            self.w_p = (2*np.pi*self.Rctr - self.g_p*self.pMode) / self.pMode
        else:
            self.w_p = w_p
        #fish-scale + carrier
        if w_c==None:
            #self.w_c = (2*np.pi*self.Rctr - self.g_c*self.cMode + self.g_p*self.pMode) / self.cMode
            self.w_c = self.w_p * self.pMode / self.cMode - self.g_c
        else:
            self.w_c = None
        #slice only
        if w_s==None:
            #self.w_s = (2*np.pi*self.Rctr - self.g_tot) / self.sMode
            self.w_s = self.w_c * self.cMode / self.sMode - self.g_s
        else:
            #only slice fish-scale
            self.w_s = w_s

        return

    def calculateBetas(self):
        self.beta_s = self.calculateBeta(self.w_s, self.h_s)
        self.beta_c = self.calculateBeta(self.w_c, self.h_c)
        self.beta_p = self.calculateBeta(self.w_p, self.h_p)
        self.beta_tot = self.beta_s+self.beta_c+self.beta_p
        return


    def calculateEFs(self, alpha):
        """
        calculates all EFs
        """
        self.EF_s =   self.calculateEF(alpha, self.beta_s)
        self.EF_c =   self.calculateEF(alpha, self.beta_c)
        self.EF_p =   self.calculateEF(alpha, self.beta_p)
        self.EF_tot = self.calculateEF(alpha, self.beta_s+self.beta_c+self.beta_p)
        return

    def calculateBeta(self, w,h):
        """
        calculates angle of fish-scale given height and width
        """
        beta = np.arctan(h/w)
        return beta

    def calculateEF(self, alpha, beta):
        """
        calculates enhancement factor given alpha(magnetic field line incident angle)
        and beta (fish-scale angle)
        """
        EF = np.sin(alpha+beta) / np.sin(alpha)
        return EF


    def addHeightsManually(self, aoi, aoiDesign):
        """
        function to add heights using simple magnetic field model
        can be used to increase fish-scale height when changing aoi

        aoi and aoiDesign are in radians

        aoiDesign is the max aoi the fishscales were designed to
        """
        
        self.h_s += self.g_s * np.abs(np.tan(aoi) - np.tan(aoiDesign))
        self.h_c += self.g_c * np.abs(np.tan(aoi) - np.tan(aoiDesign))
        self.h_p += self.g_p * np.abs(np.tan(aoi) - np.tan(aoiDesign))

        print(self.h_s)
        print(self.h_c)
        print(self.h_p)
        return



#build tile and calculate enhancement factors
tile = tileClass()
tile.setupTile(tileName, R_sp)
tile.setupHeight(h_s, h_c, h_p)
#tile.addHeightsManually(aoi, aoiDesign)
tile.calculateWidths()
tile.calculateBetas()
tile.calculateEFs(alpha)

#define angle explicitly
#EF_tot_explicit = tile.calculateEF(alpha, np.radians(1.22))


print("Slice Width: {:f} m".format(tile.w_s))
print("Carrier Width: {:f} m".format(tile.w_c))
print("Pedestal Width: {:f} m".format(tile.w_p))
print("\n")
print("Slice Beta: {:f} deg".format(np.degrees(tile.beta_s)))
print("Carrier Beta: {:f} deg".format(np.degrees(tile.beta_c)))
print("Pedestal Beta: {:f} deg".format(np.degrees(tile.beta_p)))
print("\n")
print("EF - Slice to Slice: {:f}".format(tile.EF_s))
print("EF - Carrier to Carrier: {:f}".format(tile.EF_c))
print("EF - Pedestal to Pedestal: {:f}".format(tile.EF_p))
print("EF - Total: {:f}".format(tile.EF_tot))





fig = go.Figure()
angles = np.radians(np.linspace(0.1, 3, 1000))
b = tile.beta_s

EF1 = np.sin(angles+b) / np.sin(angles)
EF2 = np.sin(angles+b) / np.sin(np.radians(aoiDesign))
L_tot = np.sqrt(tile.w_s**2 + tile.h_s**2)
EF = EF1*EF2

#fig.add_trace(go.Scatter(x=np.degrees(angles), y=EF1))
fig.add_trace(go.Scatter(x=np.degrees(angles), y=EF))

fig.update_xaxes(title="Angle of Incidence [degrees]")
fig.update_yaxes(title="Enhancement Factor * AOI Factor")
fig.update_layout(        font=dict(
            size=20,
        ))
fig.show()

minIdx = np.argmin(EF)
idxCurrent = np.argmin(np.abs(np.degrees(angles)-aoi))
print("Theoretical")
print(EF[minIdx])
print(np.degrees(angles[minIdx]))
print("Current")
print(EF[idxCurrent])
print(np.degrees(angles[idxCurrent]))