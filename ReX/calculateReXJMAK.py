
import numpy as np
import plotly.graph_objects as go

#GLADIS ANSYS data
#f = '/home/tlooby/Downloads/peakGLADIS_2000C.csv'
#HEAT PRD data
#f = '/home/tlooby/results/APS2023/sweepMEQ_T4_data/Tile094_TMax.csv'
#f = '/home/tlooby/HEAT/data/sparc_000001_sweepMEQ_T4_20231206_nominal_lq0.6_S0.6_fRadDiv70/Tile032_TMax.csv'
#f = '/home/tlooby/HEAT/data/sparc_000001_sweepMEQ_T4_20231206_nominal_lq0.6_S0.6_fRadDiv20/Tile032_TMax.csv'
#f = '/home/tlooby/HEAT/data/sparc_000001_sweepMEQ_T4_20231206_nominal_fRadDiv70_lq0.6_S0.6/TmaxData_PV.csv'
f = '/home/tlooby/HEAT/data/sparc_000001_sweepMEQ_T4_20231206_nominal_fRadDiv70_lq0.6_S0.6/Tpeak_v2.csv'
data = np.genfromtxt(f, delimiter=',', skip_header=True)


R = 8.314462 #J/K/mol
Q = 300e3 #activation energy in J/mol

Ea = Q / (9.648e4) #activation energy in eV/atom
kB = 8.617333e-5 #eV/K


#data for 16mm 12 hr test as an example
#time
tOven = np.array([1.0,2,4,8,12])
#hardness from tests
HV2 = np.array([440.5, 427., 363.8, 353.5, 359.0])
#as received hardness [vickers]
HV2AR = HV2[0]+1.0
HV2max = 350 #per olivia from literature
ReXOven = (HV2AR-HV2) / (HV2AR - HV2max)


T_ref = 1200 + 273.15
#test data
#hrs = 0.25
#t = np.linspace(0, hrs, 100)
#T_kelvin = np.zeros((len(t))) + T_ref


#real data from file
T_kelvin = data[:,1] #+ 273.15
t = data[:,0]/3600.0


kArr = []
nArr = []
EArr = []
names = []

#set everything but k0
k0 = 1553034867.0328593
Ea = 3.0
n = 2.0
kArr.append(k0)
nArr.append(n)
EArr.append(Ea)
names.append('Fit 1')

k0 = 4095139043853.176
Ea = 4.000
n = 2.000
kArr.append(k0)
nArr.append(n)
EArr.append(Ea)
names.append('Fit 2')

k0 = 4446601449.779298
Ea = 3.0
n = 1.0
kArr.append(k0)
nArr.append(n)
EArr.append(Ea)
names.append('Fit 3')

k0 = 500861865.1508129
Ea = 3.0
n = 3.0
kArr.append(k0)
nArr.append(n)
EArr.append(Ea)
names.append('Fit 4')

del k0
del Ea
del n

N = 1

def integrateT(t,T,Ea):
    integrand = np.exp(-Ea / (kB * T))
    return np.trapz(integrand, t)

def calculateTrapz(dt, T1, T2, E):
    var1 = np.exp(-E / (kB * T1))
    var2 = np.exp(-E / (kB * T2))
    return dt * (var1 + var2) / 2.0

#using boltzman constant
#c1 = np.exp(-Ea / (kB*T_ref))
#c2 = np.exp(Ea / (kB*T_ref))
#c3 = integrateT(t,T_kelvin,Ea)
#X_eV = 1 - np.exp( -k0 * c1 * (c2 * c3 * N)**n )
#print("ReX fraction using eV: {:f}%".format(X_eV*100.0))
#print("Peak Temperature = {:0.4f} [degC]".format(max(T_kelvin-273.15)))

markers = ['circle','square','cross','diamond','star','triangle-up']

#plot ReX as a function of time for single pulse
fig = go.Figure()
ReX = []
for i in range(len(kArr)):
    k = kArr[i]
    n = nArr[i]
    Ea = EArr[i]
    c1 = np.exp(-Ea / (kB*T_ref))
    c2 = np.exp(Ea / (kB*T_ref))
    ReX = []
    for j in range(len(t)):
        c3 = integrateT(t[:j],T_kelvin[:j],Ea)
        ReX.append(1 - np.exp(-k*c1 * (c2 * c3)**n ))
    fig.add_trace(go.Scatter(x=t*3600.0, y=ReX, 
                         name='k0 = {:0.3e}, E_a = {:0.1f}, n = {:0.1f}'.format(k, Ea, n),
                         mode = 'lines+markers',
                         marker_symbol=markers[i], marker_size=10
                         ))
    fig.update_xaxes(title='Time [s]')


#plot ReX as a function of time for single pulse using trapezoidal integration at
#each timestep (instead of a curve integration)
#fig = go.Figure()
#ReX = 0
#for i in range(len(kArr)):
#    k = kArr[i]
#    n = nArr[i]
#    Ea = EArr[i]
#    c1 = np.exp(-Ea / (kB*T_ref))
#    c2 = np.exp(Ea / (kB*T_ref))
#    ReX = 0
#    for j in range(len(t)):
#        if j>0:
#            dt = t[j] - t[j-1]
#            T2 = T_kelvin[j]
#            T1 = T_kelvin[j-1]
#            c3 = calculateTrapz(dt,T1,T2,Ea)       
#            ReX += 1 - np.exp(-k*c1 * (c2 * c3)**n )
#
#    print(ReX)



#overlay oven test data
#fig.add_trace(go.Scatter(x=tOven*3600.0, y=ReXOven, name='Oven ReX Data', mode = 'markers',
#                         marker_symbol=markers[i+1], marker_size=25))

#plot ReX as a function of pulse N
#fig = go.Figure()
#
#for i in range(len(kArr)):
#    k = kArr[i]
#    n = nArr[i]
#    Ea = EArr[i]
#    ReX = []
#    c1 = np.exp(-Ea / (kB*T_ref))
#    c2 = np.exp(Ea / (kB*T_ref))
#    c3 = integrateT(t,T_kelvin,Ea)
#    for j in range(N):
#        ReX.append(1 - np.exp(-k*c1 * (c2 * c3 * j)**n ))
#
#    fig.add_trace(go.Scatter(x=np.arange(N), y=ReX, 
#                         name='k0 = {:0.3e}, E_a = {:0.1f}, n = {:0.1f}'.format(k, Ea, n),
#                         mode = 'lines+markers',
#                         marker_symbol=markers[i], marker_size=10
#                         ))
#fig.update_xaxes(title='N Experiments')


#mults = np.linspace(0.5, 1.2, 100)
##plot ReX fraction as a function of Tpeak
#fig = go.Figure()
##loop through k and n arrays
#for i in range(len(kArr)):
#    k = kArr[i]
#    n = nArr[i]
#    Ea = EArr[i]
#    c1 = np.exp(-Ea / (kB*T_ref))
#    c2 = np.exp(Ea / (kB*T_ref))
#    ReX = []
#    Tpeak = []
#    for m in mults:
#        c3 = integrateT(t,T_kelvin*m,Ea)
#        ReX.append(1 - np.exp(-k*c1 * (c2 * c3)**n ))
#        Tpeak.append(max(T_kelvin*m) - 273.15)
#    fig.add_trace(go.Scatter(x=Tpeak, y=ReX, 
#                             name='k0 = {:0.3e}, E_a = {:0.1f}, n = {:0.1f}'.format(k, Ea, n),
#                             mode = 'lines+markers',
#                             marker_symbol=markers[i], marker_size=10
#                             ))
#fig.update_xaxes(title='Tpeak [degC]')







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
    legend=dict(
        x=0,  # x=0 positions the legend at the left
        y=1,  # y=1 positions the legend at the top
        xanchor='left',  # 'left' anchors the left part of the legend at x position
        yanchor='top'    # 'top' anchors the top part of the legend at y position
    )    
)
fig.update_yaxes(title='ReX Fraction')
fig.show()




