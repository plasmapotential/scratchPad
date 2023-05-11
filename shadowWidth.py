#shadowWidth.py
#Description:   Analytically calculate fish-scale shadow width provided AOI and height
#               assumes fish-scale is poloidally uniform
#Engineer:      T Looby
#Date:          20230403

import numpy as np
#angle of incidence
aoi = 5.0 #degrees
Rctr = 1.78

def calculateShadowFrac(aoi, h, g, w_x, h_x):
    """
    calculates fraction shadowed area
    """
    alpha = np.radians(aoi)
    beta = np.arctan(h_x/w_x)
    x = (h - g*np.tan(alpha)) / (np.tan(beta) + np.tan(alpha))
    l = x/np.cos(beta)
    frac_shadowed = l / np.sqrt(w_s**2+h_s**2)
#    EF = (w_s+g) / (np.sqrt(w_s**2+h_s**2) - l)
#    EF_matt = np.cos(beta) + np.sin(beta)/np.tan(alpha)
    return frac_shadowed

w_s = 14580.0 # [um], width of slice at ctr
#slice-slice
h_s = 118.0 # [um], fish-scale height
#h_s = 178.0
g = 500.0 # [um], gap between tiles
frac_s = calculateShadowFrac(aoi, h_s, g, w_s, h_s)


#carrier-carrier
h = 357.0 # [um], fish-scale height
#h = 602.0
g = 2000.0 # [um], gap between tiles
w_c = 2*np.pi*Rctr / 144
frac_c = calculateShadowFrac(aoi, h, g, w_c, h)



#plate-plate
h = 1079.0 # [um], fish-scale height
#h = 1579.0
g = 5000.0 # [um], gap between tiles
w_p = 2*np.pi*Rctr / 36
frac_p = calculateShadowFrac(aoi, h, g, w_p, h)

shadL = 20*w_s*frac_s
totL = 20*w_s

print("Shadow Frac at Ctr slice: {:f}".format(frac_s))
print("Shadow Frac at Ctr carrier: {:f}".format(frac_c))
print("Shadow Frac at Ctr plate: {:f}".format(frac_p))
print("Shadowed L: {:f}".format(shadL))
print("Total L: {:f}".format(totL))
print(totL / shadL)


#print("Shadow Width at Ctr: {:f}".format(l))
#print("Shadowed Fraction: {:f}%".format(frac_shadowed*100.0))
#print("Enhancement Factor: {:f}".format(EF))
#print("Matt's EF: {:f}".format(EF_matt))