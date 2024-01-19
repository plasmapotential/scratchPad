#calculates ReX fraction from hardness tests, then finds
#avrami exponent and prefactor

import numpy as np
from scipy.optimize import curve_fit
import plotly.graph_objects as go


#R = 8.314462 #J/K/mol
#Q = 300e3 #activation energy in J/mol
#
#Ea = Q / (9.648e4) #activation energy in eV/atom
kB = 8.617333e-5 #eV/K


Npulses = 1

T_ref = 1200 + 273.15

Ea = 4.0
n = 2.0

def avrami(t,k,n):
   return 1-(np.exp(-k*t**n))

def avrami2(t,k):
   return 1-(np.exp(-k*t**2.0))

def linear_func(x, m, c):
    return m * x + c

def JMAK(t, k0, Ea, n):
   return 1 - np.exp(-k0*np.exp(-Ea/(kB*T_ref)) * t**n )

def JMAK2(t, k0, Ea):
   n=2.0
   return 1 - np.exp(-k0*np.exp(-Ea/(kB*T_ref)) * t**n)

def JMAK3(t, k0):
   return 1 - np.exp(-k0*np.exp(-Ea/(kB*T_ref)) * t**n )


fig = go.Figure()

#=== 16mm @ 1200degC
#time
t = np.array([1.0,2,4,8,12])
#hardness from tests
HV2 = np.array([440.5, 427., 363.8, 353.5, 359.0])
#as received hardness [vickers]
HV2AR = HV2[0]+1.0
HV2max = 350 #per olivia from literature
ReX = (HV2AR-HV2) / (HV2AR - HV2max)
#plot hardness
#fig.add_trace(go.Scatter(x=t, y=HV2, name='HV2', mode='markers', marker={'size':15}))
#n = np.log(np.log( 1.0 / (1-ReX) )) / np.log(t)
#K  = np.log(1/1-ReX)
lnt = np.log(t)
Y = np.log(np.log(1/(1-ReX)))

#plot ReX
#log space
#fig.add_trace(go.Scatter(x=np.log(t), y=np.log(np.log( 1.0 / (1-ReX) )), name='logPlot',
#                         mode='markers', marker={'size':15}))
#normal space
fig.add_trace(go.Scatter(x=t, y=ReX, name='ReX', mode='markers', marker={'size':15}))

# fit all parameters in JMAK equation
#lower_bounds = [1.0, 3.0, 1.0]
#upper_bounds = [10000000000.0, 7.0, 4.0]
#params, covariance = curve_fit(JMAK, t, ReX, bounds=(lower_bounds, upper_bounds))
#k_opt, E_opt, n_opt = params
#print(f"Optimized parameters: k0 = {k_opt}, E_a = {E_opt}, n = {n_opt}")
#t2 = np.linspace(min(t), max(t), 100)
#y2 = JMAK(t2,k_opt,E_opt,n_opt)
#fig.add_trace(go.Scatter(x=t2, y=y2, name='k={:0.4f}, n={:0.4f}'.format(k_opt,n_opt)))

# fit all parameters in JMAK equation except n
#lower_bounds = [1.0, 3.0]
#upper_bounds = [1e10, 7.0]
#params, covariance = curve_fit(JMAK2, t, ReX, bounds=(lower_bounds, upper_bounds))
#k_opt, E_opt= params
#print(f"Optimized parameters: k0 = {k_opt}, E_a = {E_opt}, n = 2.0")
#t2 = np.linspace(min(t), max(t), 100)
#y2 = JMAK2(t2,k_opt,E_opt)
#fig.add_trace(go.Scatter(x=t2, y=y2, name='k={:0.4f}'.format(k_opt)))

# fit k0 in JMAK equation only
lower_bounds = [1.0]
upper_bounds = [1e14]
params, covariance = curve_fit(JMAK3, t, ReX, bounds=(lower_bounds, upper_bounds))
k_opt, = params
print("Optimized parameters: k0 = {:0.3f}, E_a = {:0.3f}, n = {:0.3f}".format(k_opt, Ea, n))
t2 = np.linspace(min(t), max(t), 100)
y2 = JMAK3(t2,k_opt)
fig.add_trace(go.Scatter(x=t2, y=y2, name='k0 = {:0.3e}, E_a = {:0.1f}, n = {:0.1f}'.format(k_opt, Ea, n)))


# Perform the curve fitting without constraining n
#params, covariance = curve_fit(avrami, t, ReX)
#k_opt, n_opt = params
#print(f"Optimized parameters: k = {k_opt}, n = {n_opt}")
#t2 = np.linspace(min(t), max(t), 100)
#y2 = avrami(t2,k_opt,n_opt)
#fig.add_trace(go.Scatter(x=t2, y=y2, name='k={:0.4f}, n={:0.4f}'.format(k_opt,n_opt)))


## Perform the curve fitting with n=fixed
#params, covariance = curve_fit(avrami2, t, ReX)
#k_opt = params[0]
#print(f"Optimized parameters: k = {k_opt}")
##fig.add_trace(go.Scatter(x=t, y=avrami2(t,k_opt), name="n=2"))
#t2 = np.linspace(min(t), max(t), 100)
#y2 = avrami2(t2,k_opt)
#fig.add_trace(go.Scatter(x=t2, y=y2, name='k={:0.4f}, n=2'.format(k_opt)))


#plot log space
#fig.add_trace(go.Scatter(x=np.log(t), y=Y, name='ReX', mode='markers', marker={'size':15}))
#fig.add_trace(go.Scatter(x=np.log(t2), y=np.log(np.log(1/(1-y2))), name='test'))


# Perform a linear fit
#t2 = np.linspace(min(t), max(t), 100)
#params, _ = curve_fit(linear_func, lnt, Y)
## Extract parameters
#n, lnK = params
#kref = np.exp(lnK)
#print("Optimized parameters: kref={:0.4f}, n = {:0.4f}".format(kref,n))
#y2 = avrami(t2,kref,n)
##log space
#fig.add_trace(go.Scatter(x=np.log(t2), y=np.log(np.log(1/(1-y2))), name="n={:0.4f}, kref={:0.4f}".format(n,kref)))
#time space
#fig.add_trace(go.Scatter(x=t2, y=avrami(t2,kref,n), name='k={:0.4f}, n={:0.4f}'.format(kref,n)))



fig.update_layout(
#title="Temperature Probe Time Evolution",
    font=dict(
        family="Arial",
        size=20,
        color="Black"
    ),
    margin=dict(
        l=10,
        r=10,
        b=10,
        t=10,
        pad=2
    ),
)

fig.update_yaxes(title='ReX Fraction')
fig.update_xaxes(title='Time [hours]')
fig.show()


