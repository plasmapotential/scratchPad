#enhancementFactor.py
#Description:   Analytically calculates enhancement factor for multiple mode number fish-scale
#               assumes fish-scale is poloidally uniform
#Engineer:      T Looby
#Date:          20230403

import numpy as np
#angle of incidence
aoi = 4.31 #degrees
Rctr = 1.76
alpha = np.radians(aoi)

def calculateBeta(w,h):
    """
    calculates angle of fish-scale given height and width
    """
    beta = np.arctan(h/w)
    return beta

def calculateEF(alpha, beta):
    """
    calculates enhancement factor given alpha(magnetic field line incident angle)
    and beta (fish-scale angle)
    """
    EF = np.sin(alpha+beta) / np.sin(alpha)
    return EF


#gaps between components
g_s = 500 #[um]
g_c = 2000 #[um]
g_p = 5000 #[um]

g_tot = 500*720 + 2000*144 + 5000*36

#only slice fish-scale
#w_s = 14580.0 # [um], width of slice at ctr
w_s = (2*np.pi*Rctr - g_tot*1e-6) / 720.0 * 1e6
h_s = 58.0 # [um], fish-scale height
beta_s = calculateBeta(w_s, h_s)
EF_s = calculateEF(alpha, beta_s)

#fish-scale + carrier
w_c = (2*np.pi*Rctr - 2000e-6*144 + 5000e-6*36) / 144.0* 1e6
h_c = 232.0 #[um]
beta_c = calculateBeta(w_c, h_c)
EF_c = calculateEF(alpha, beta_c)

#fish-scale + carrier + pedestal
w_p = (2*np.pi*Rctr - 5000e-6*36) / 36.0 * 1e6
h_p = 579.0 #[um]
beta_p = calculateBeta(w_p, h_p)
EF_p = calculateEF(alpha, beta_p)

#use angles above to define angle
#EF_tot = calculateEF(alpha, beta_s+beta_c+beta_p)
#define angle explicitly
EF_tot = calculateEF(alpha, np.radians(1.22))


print("Slice Width: {:f} um".format(w_s))
print("Carrier Width: {:f} um".format(w_c))
print("Pedestal Width: {:f} um".format(w_p))
print("\n")
print("Slice Beta: {:f} deg".format(np.degrees(beta_s)))
print("Carrier Beta: {:f} deg".format(np.degrees(beta_c)))
print("Pedestal Beta: {:f} deg".format(np.degrees(beta_p)))
print("\n")
print("EF - Slice to Slice: {:f}".format(EF_s))
print("EF - Carrier to Carrier: {:f}".format(EF_c))
print("EF - Pedestal to Pedestal: {:f}".format(EF_p))
print("EF - Total: {:f}".format(EF_tot))



