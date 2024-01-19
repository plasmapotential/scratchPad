#calculates ReX fraction given a temperature history and the avrami coefficients

import numpy as np
import plotly.graph_objects as go

f = '/home/tlooby/Downloads/peakGLADIS_2000C.csv'
data = np.genfromtxt(f, delimiter=',', skip_header=True)


R = 8.314462 #J/K/mol
Q = 300e3 #activation energy in J/mol

Ea = Q / (9.648e4) #activation energy in eV/atom
kB = 8.617333e-5 #eV/K

print(Ea)

Npulses = 1

T_ref = 1200 + 273.15


#test data
#hrs = 2
#t = np.linspace(0, hrs, 100)
#T_kelvin = np.zeros((len(t))) + T_ref

#real data from file
T_kelvin = data[:,1]+ 273.15
t = data[:,0]/3600.0

kArr = []
nArr = []
names = []

#from Olivia's avrami fit using scipy
kref = 0.0157
n = 3.4547
kArr.append(kref)
nArr.append(n)
names.append('avramiFit')

#using linear fit in logspace
kref=0.0257
n = 2.2115
kArr.append(kref)
nArr.append(n)
names.append('logspace')


#from Pantleon
kref = 0.0869
n = 2.0
kArr.append(kref)
nArr.append(n)
names.append('Pantleon')

def integrateT(t,T,Ea):
    integrand = np.exp(-Ea / (kB * T))
    return np.trapz(integrand, t)

#using boltzman constant
#c1 = np.exp(-Ea / (kB*T_ref))
#c2 = np.exp(Ea / (kB*T_ref))
#c3 = integrateT(t,T_kelvin,Ea)
##X_eV = 1 - np.exp( -kref * c1 * (c2 * c3)**n )
#X_eV = 1 - np.exp( -kref * (c2 * c3)**n )
#print("ReX fraction using eV: {:f}%".format(X_eV*100.0))
#print("Peak Temperature = {:0.4f} [degC]".format(max(T_kelvin-273.15)))

#using gas constant
#integrand = np.exp(-Q / (R * T_kelvin))
#c1 = np.exp(-Q / (R*T_ref))
#c2 = np.exp(Q / (R*T_ref))
#c3 = np.trapz(integrand, t)
#X_J = 1 - np.exp( -kref * c1 * (c2 * c3)**n )
#X_J = 1 - np.exp( -kref * (c2 * c3)**n )
#print("ReX fraction using J: {:f}%".format(X_J*100.0))



#plot ReX fraction as a function of Tpeak
#mults = np.linspace(0.25, 1, 100)
#fig = go.Figure()
##loop through k and n arrays
#for i in range(len(kArr)):
#    k = kArr[i]
#    n = nArr[i]
#    c1 = np.exp(-Ea / (kB*T_ref))
#    c2 = np.exp(Ea / (kB*T_ref))
#    ReX = []
#    Tpeak = []
#    for m in mults:
#        c3 = integrateT(t,T_kelvin*m,Ea)
#        ReX.append(1 - np.exp( -k * (c2 * c3)**n ))
#        Tpeak.append(max(T_kelvin*m) - 273.15)
#    fig.add_trace(go.Scatter(x=Tpeak, y=ReX, 
#                             name='k={:0.4f}, n={:0.4f}'.format(k,n),
#                             mode = 'lines+markers',
#                             marker_symbol=markers[i], marker_size=10
#                             ))


#plot ReX as a function of time for single pulse
fig = go.Figure()
markers = ['circle','square','cross','diamond','star','triangle-up']
c1 = np.exp(-Ea / (kB*T_ref))
c2 = np.exp(Ea / (kB*T_ref))
for i in range(len(kArr)):
    k = kArr[i]
    n = nArr[i]
    ReX = []
    for j in range(len(t)):
        c3 = integrateT(t[:j],T_kelvin[:j],Ea)
        ReX.append(1 - np.exp(-1.0*k * (c2 * c3)**n ))
        print('{:.16f}'.format((c2 * c3)))
    fig.add_trace(go.Scatter(x=t*3600.0, y=ReX, 
                         name='k={:0.4f}, n={:0.4f}'.format(k,n),
                         mode = 'lines+markers',
                         marker_symbol=markers[i], marker_size=10
                         ))


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

fig.update_yaxes(title='ReX Fraction', range=[0,1])
fig.update_xaxes(title='Time [s]')
fig.show()




