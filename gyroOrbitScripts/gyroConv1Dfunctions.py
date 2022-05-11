import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.integrate import trapz, quad
from scipy.stats import maxwell
from scipy.interpolate import interp1d
import scipy.integrate as integrate
#unit conversions
kg2eV = 5.609e35 #1kg = 5.609e35 eV
eV2K = 1.160e4 #1ev=1.160e4 K

#constants
m_eV = 931.49e6 #eV/c^2
#m_kg = m / kg2eV
kB = 8.617e-5 #ev/K
B = 5 #tesla
e = 1.602e-19 # C
T0_eV = 100 # eV
#T0_K = T0_eV * eV2K
c = 3e8 #m/s

#velocity space angle
gyroAng = np.radians(45)
print(gyroAng)

print("vAvg")
vAvg = np.sqrt(2*T0_eV/(m_eV/c**2))
print(vAvg)

vMax = 5*vAvg

v = np.linspace(0,vMax, 10000)
vPerp = v*np.cos(gyroAng)
vParallel = v*np.sin(gyroAng)
vPerpAvg = vAvg*np.cos(gyroAng)
vParallelAvg = vAvg*np.sin(gyroAng)

#f_v = v * (m_eV/c**2) / (T0_eV) * np.exp(-(m_eV/c**2 * v**2) / (2*T0_eV) )
#fofV = lambda x: x * (m_eV/c**2) / (T0_eV) * np.exp(-(m_eV/c**2 * x**2) / (2*T0_eV) )
fofV = lambda x: 4*np.pi* x**2 * ( (m_eV/c**2) / (2*np.pi*T0_eV) )**(3.0/2.0) * np.exp(-(m_eV/c**2 * x**2) / (2*T0_eV) )

f_v = fofV(v)

f_vPerp = vPerp * (m_eV/c**2) / (T0_eV) * np.exp(-(m_eV/c**2 * vPerp**2) / (2*T0_eV) )
f_vParallel = vParallel * (m_eV/c**2) / (T0_eV) * np.exp(-(m_eV/c**2 * vParallel**2) / (2*T0_eV) )

v_cdf = np.cumsum(f_v[1:])*np.diff(v)
v_cdf = np.insert(v_cdf, 0, 0)
#v_cdf = np.linspace(0,1,len(v))

inverseCDF = interp1d(v_cdf, v, kind='linear')
forwardCDF = interp1d(v, v_cdf, kind='linear')

cdfMax = v_cdf[-1]
cdfMin = v_cdf[0]
N_vSlice = 100
sliceWidth = cdfMax / (N_vSlice+1)
#cdfSlices = np.linspace(cdfMin,cdfMax,N_vSlice+2)[1:-1]
cdfSlices = np.linspace(0,1,N_vSlice+2)[1:-1]
CDFbounds = np.linspace(0,1,N_vSlice+1)[1:-1]
vSlices = inverseCDF(cdfSlices)
vBounds = inverseCDF(CDFbounds)
vBounds = np.insert(vBounds,0,0)
vBounds = np.append(vBounds,vMax)
print("vSlices")
print(vSlices)
print("cdfSlices")
print(cdfSlices)
print("bounds")
print(CDFbounds)
print(vBounds)

V_int = np.zeros((N_vSlice))
#velocity integrals
for j in range(N_vSlice):
    V_int[j] = integrate.quad(fofV, vBounds[j], vBounds[j+1])[0]
print(V_int)

#energy pdf (missing 1/2*mass but that gets divided out later anyways )
EofV = lambda x: x**2 * fofV(x)
E_v = EofV(v)
energySlices = EofV(vSlices)
energyIntegrals = np.zeros((N_vSlice))
#energy integrals
for j in range(N_vSlice):
    energyIntegrals[j] = integrate.quad(EofV, vBounds[j], vBounds[j+1])[0]

print("Energies")
print(energySlices)
print("Energy Ints")
print(energyIntegrals)

energyTotal = energyIntegrals.sum()
energyFracs = np.zeros((N_vSlice))
print("Energy Fracs")
for vSlice in range(N_vSlice):
    energyFracs[vSlice] = energyIntegrals[vSlice] / energyTotal
print(energyFracs)
print("Esum")
print(np.sum(energyFracs))

print("===Averages:")
print(vAvg)
print(vPerpAvg)
print(vParallelAvg)

print("===Integrals")
#Velocity stuff
#print(np.sum(f_v)*dv)
print("Integral of f(v):")
print(trapz(f_v,v))
print("Integral of slices")
sum = np.zeros((len(vSlices)+1))
#for i in range(len(vSlices)-1):
#    sum[i] = quad(fofV, vSlices[i], vSlices[i+1])[0]
#    print(quad(fofV, vSlices[i], vSlices[i+1])[0])
#sum[-1] = quad(fofV, vSlices[-1], v[-1])[0]
#print(quad(fofV, vSlices[-1], v[-1])[0])


for i in range(len(vSlices)):
    if i==0:
        sum[i] = quad(fofV, 0, vSlices[i])[0]
    else:
        sum[i] = quad(fofV, vSlices[i-1], vSlices[i])[0]
    print(sum[i])
sum[-1] = quad(fofV, vSlices[-1], vMax)[0]
print(sum[-1])
print("sum of vSlice integrals")
print(np.sum(sum))




print('\n')
print('\n')
print(vMax)

N_vP = 100
vP = np.linspace(0.0,np.pi/2,N_vP+2)#[1:-1]
print(vP)

vPerps = []
vPars = []
for vS in vSlices:
    for j in range(len(vP)):
        vPerps.append(vS*np.cos(vP[j]))
        vPars.append(vS*np.sin(vP[j]))

fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(go.Histogram(x=vPerps, opacity=0.5, name="vPerp"), secondary_y=False)
fig.add_trace(go.Histogram(x=vPars, opacity=0.5, name='v||'), secondary_y=False)
fig.add_trace(go.Scatter(x=v, y=f_v, name="vPDF", line=dict(width=4, dash='solid'),
                         mode='lines', marker_symbol='circle', marker_size=14),
                         secondary_y=True)

conv = np.convolve(f_v,t,'same')



fig.show()
