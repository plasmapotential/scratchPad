#calculates the depth in a material that is above a maximum Temperature
import numpy as np
from scipy.special import erfinv, erfcinv

Tmax = 2000.0
T0 = 20.0
q = 1e9 #MW/m^2
kappa = 86.0 #W m^-1 K^-1
alpha = 23.0 * 1e-6 #m^2 s^-1 

#user specifies depth or time, calculator calculate the other
mode = 'depth'

x = 0.0004 #m
#t = 0.001 #s


if mode == 'depth':
    Tsurf = q/kappa * x + Tmax
    arg = (Tmax - T0) / (Tsurf - T0)
    t = x**2 / (2 * erfinv(arg))**2 / alpha
    print(t)
    print("\n\nDepth above {:0.1f} degC: {:0.3f} um".format(Tmax,x*1e6))
    print("Surface Temperature: {:0.1f}".format(Tsurf))
    print("Time until depth reaches Tmax: {:e}".format(t))
