#plots Free Streaming Quantities vs HEAT quantities

import os
import numpy as np
import plotly.graph_objects as go
from scipy import special
from scipy.interpolate import interp1d
import scipy.integrate as integrate
from scipy.ndimage import uniform_filter1d
from scipy.ndimage import gaussian_filter1d


#constants
ionMassAMU = 2.515
T_eV = 1000.0 #eV

kg2eV = 5.609e35 #1kg = 5.609e35 eV/c^2
eV2K = 1.160e4 #1ev=1.160e4 K
AMU = 931.494e6 #ev/c^2
kB = 8.617e-5 #ev/K
e = 1.602e-19 # C
c = 299792458 #m/s
mass_eV = ionMassAMU * AMU
Z=1 #assuming isotopes of hydrogen here

vThermal = np.sqrt(2.0*T_eV/(mass_eV/c**2))
#vThermal = np.sqrt(T_eV/(mass_eV/c**2))
LPar = 4.649 #m

dt=1e-6
tMax = 50e-6
t = np.linspace(dt, tMax, int(tMax/dt))
tau = LPar / vThermal
x_a = tau / t


# === EDIT THESE ===
#plot filament draining
inFilMask = False
#plot normalized deposition per FSM
tgtNormMask =  True
#plot normalized deposition per HEAT
HEATmaskNorm = True
#plot deposition per FSM
tgtMask =  False
#plot deposition per HEAT
HEATmask = False
#HEAT paraview movieDir where the results live
rootPath = '/home/tom/HEAT/data/sparc_000001_filament_1ptFSM_NvS500/paraview/'
#rootPath = '/home/tom/HEAT/data/sparc_000001_filament_1ptFSM_NvS30/paraview/'
#where we save vector graphics
saveMask = False
outPath = '/home/tom/work/manuscripts/ELMs/figures/'
svgName = '30macros.svg'
#window type for filtering HEAT data, boxcar, gaussian, none
winType = 'none'
#window size
window=2
# ===================


fig = go.Figure()

if HEATmask==True:
    #this is used to compare HEAT output to FSM.  To do this, you need to trace a 
    #filament with Nr=Nb=Np=1 to a target, then calculate the connection length,
    #and update the parameters above
    prefixE = 'Edep_'
    prefixP = 'Particles_'
    namesE = sorted([f for f in os.listdir(rootPath) if (os.path.isfile(os.path.join(rootPath, f)) and prefixE in f)])
    namesP = sorted([f for f in os.listdir(rootPath) if (os.path.isfile(os.path.join(rootPath, f)) and prefixP in f)])
    Edep = np.zeros((len(namesE)))
    pDep = np.zeros((len(namesP)))
    for i,n in enumerate(namesE):
        E = np.genfromtxt(rootPath+n, delimiter=',', comments='#')
        Edep[i] = np.sum(E[:,3])
        if i==12:
            print("Location of Peak Flux")
            print(n)
            idx = np.argmax(E[:,3])
            print(E[idx,:-1])
            print('peak energy: {:f} [J]'.format(Edep[i]/dt/2))
    for i,n in enumerate(namesP):
        P = np.genfromtxt(rootPath+n, delimiter=',', comments='#')
        pDep[i] = np.sum(P[:,3])


    #boxcar filter
    if winType=='boxcar':
        Eavg = uniform_filter1d(Edep[1:], size=window)
        Pavg = uniform_filter1d(pDep[1:], size=window)
    #gaussian filter
    elif winType =='gaussian':
        Eavg = gaussian_filter1d(Edep[1:], window)
        Pavg = gaussian_filter1d(pDep[1:], window)    
    #no filter
    else:
        Eavg = Edep
        Pavg = pDep

#    fig.add_trace(go.Scatter(x=t, y=Eavg[1:]/dt, name="HEAT Deposition"))

    #particle flux
    fig.add_trace(go.Scatter(x=t, y=Pavg/dt/2, name="HEAT Particle Deposition",
                             line=dict(color='blue', width=2, dash='solid'),
                             mode='lines+markers', marker_symbol='cross', marker_size=10))
    #energy flux
    fig.add_trace(go.Scatter(x=t, y=Eavg/dt/2, name="HEAT Energy Deposition",
                             line=dict(color='red', width=2, dash='solid'),
                             mode='lines+markers', marker_symbol='x', marker_size=10))
    print('---')
    print("HEAT Particle Integral: {:0.3f}".format(np.sum(Pavg/dt/2)*dt))
    print("HEAT Energy Integral: {:0.3f}".format(np.sum(Eavg/dt/2)*dt))

    print("Max HEAT ptcl: {:f}".format(np.max(Pavg/dt/2)))
    print("Max HEAT energy: {:f}".format(np.max(Eavg/dt/2)))
    print("Timestep for max ptcl: {:f}".format(t[np.argmax(Pavg)]))
    print("Timestep for max energy: {:f}".format(t[np.argmax(Eavg)]))

if HEATmaskNorm==True:
    #this is used to compare HEAT output to FSM.  To do this, you need to trace a 
    #filament with Nr=Nb=Np=1 to a target, then calculate the connection length,
    #and update the parameters above
    prefixE = 'Edep_'
    prefixP = 'Particles_'
    namesE = sorted([f for f in os.listdir(rootPath) if (os.path.isfile(os.path.join(rootPath, f)) and prefixE in f)])
    namesP = sorted([f for f in os.listdir(rootPath) if (os.path.isfile(os.path.join(rootPath, f)) and prefixP in f)])
    Edep = np.zeros((len(namesE)))
    pDep = np.zeros((len(namesP)))
    for i,n in enumerate(namesE):
        E = np.genfromtxt(rootPath+n, delimiter=',', comments='#')
        Edep[i] = np.sum(E[:,3])
        #if i==20:
        #    print("Location of Peak Flux")
        #    print(n)
        #    idx = np.argmax(E[:,3])
        #    print(E[idx,:-1])
        #    print('peak energy: {:f} [J]'.format(Edep[i]))
    for i,n in enumerate(namesP):
        P = np.genfromtxt(rootPath+n, delimiter=',', comments='#')
        pDep[i] = np.sum(P[:,3])


    #boxcar filter
    if winType=='boxcar':
        Eavg = uniform_filter1d(Edep[1:], size=window)
        Pavg = uniform_filter1d(pDep[1:], size=window)
    #gaussian filter
    elif winType =='gaussian':
        Eavg = gaussian_filter1d(Edep[1:], window)
        Pavg = gaussian_filter1d(pDep[1:], window)    
    #no filter
    else:
        Eavg = Edep[1:]
        Pavg = pDep[1:]

#    fig.add_trace(go.Scatter(x=t, y=Eavg[1:]/dt, name="HEAT Deposition"))
    #particle flux
    fig.add_trace(go.Scatter(x=1/x_a, y=Pavg*tau/dt/2, name="HEAT Particle Deposition Norm",
                             line=dict(color='blue', width=2, dash='solid'),
                             mode='lines+markers', marker_symbol='cross', marker_size=10))
    #energy flux.  we normalize by 2 to account for +/- of parallel velocity distribution functino
    #  we normalize by dt to account for derivative.  we normalize by x_a to match normalization
    #  in Fundamenski Figure 5, which gives q in units of eps_0 / tau
    fig.add_trace(go.Scatter(x=1/x_a, y=Eavg*tau/dt/2/x_a, name="HEAT Energy Deposition Norm",
                             line=dict(color='red', width=2, dash='solid'),
                             mode='lines+markers', marker_symbol='x', marker_size=10))


    dx = np.diff(x_a)
    dx = np.abs(np.insert(dx, 0, 0))
    print('---')
    print("HEAT Particle Integral: {:0.3f}".format(np.sum(Pavg*tau/dt/2.0*dx)))
    print("HEAT Energy Integral: {:0.3f}".format(np.sum(Eavg*tau/dt/2.0/x_a*dx)))
    print("Max HEAT ptcl: {:f}".format(np.max(Pavg/dt/2*tau)))
    print("Max HEAT energy: {:f}".format(np.max(Eavg/dt/2*tau/x_a)))
    print("Timestep for max ptcl: {:f}".format(1/x_a[np.argmax(Pavg/dt/2*tau)]))
    print("Timestep for max energy: {:f}".format(1/x_a[np.argmax(Eavg/dt/2*tau/x_a)]))

#quantities remaining in filament as a function of time
if inFilMask==True:
    ptclsRemain = lambda t: special.erf(tau/t)
    energyRemain = lambda t: special.erf(tau/t) - 2 / (2*np.sqrt(np.pi)) * tau/t * np.exp(-(tau/t)**2)
    fig.add_trace(go.Scatter(x=t, y=ptclsRemain(t), name="Particles Remaining"))
    fig.add_trace(go.Scatter(x=t, y=energyRemain(t), name="Energy Remaining"))
    fig.update_xaxes(title="Time [s]")



#target deposited quantities
if tgtMask == True:
    multP = 1.0
    multE = x_a *3.0/2.0
    ptclTGT = lambda t: 1.0/np.sqrt(np.pi) * (tau/t)**2 * np.exp(-(tau/t)**2) / tau * multP
    energyTGT = lambda t: 1.0/np.sqrt(np.pi) * (2.0/3.0) * (1+(tau/t)**2) * (tau/t)**2 * np.exp(-(tau/t)**2) / tau * tau/t *3.0/2.0

    s = np.sum(energyTGT(t))

    fig.add_trace(go.Scatter(x=t, y=ptclTGT(t), name="FSM Particle Deposition",
                             line=dict(color='black', width=2, dash='solid')))
    fig.add_trace(go.Scatter(x=t, y=energyTGT(t), name="FSM Energy Deposition",
                             line=dict(color='black', width=2, dash='dot'),
                             mode='lines+markers', marker_symbol='circle', marker_size=10))
    fig.update_xaxes(title="Time [s]")
    
    #integrated ptcl / energy
    intP = integrate.quad(ptclTGT, t[0], t[-1])[0]
    intE = integrate.quad(energyTGT, t[0], t[-1])[0]
    print('---')
    print("Analytical Particle Integral: {:0.3f}".format(intP))
    print("Analytical Energy Integral: {:0.3f}".format(intE))
    print("Max FSM ptcl: {:f}".format(np.max(ptclTGT(t))))
    print("Max FSM ptcl: {:f}".format(np.max(energyTGT(t))))
    print("Timestep for max FSM ptcl: {:f}".format(t[np.argmax(ptclTGT(t))]))
    print("Timestep for max FSM energy: {:f}".format(t[np.argmax(energyTGT(t))]))    
   

#normalized target deposited quantities
if tgtNormMask == True:
    #ptclTGT = lambda t: 1.0/np.sqrt(np.pi) * (tau/t)**2 * np.exp(-(tau/t)**2) / tau
    #energyTGT = lambda t: 1.0/np.sqrt(np.pi) * (2.0/3.0) * (1+(tau/t)**2) * (tau/t)**2 * np.exp(-(tau/t)**2) / tau


    multP = tau
    multE = tau*3.0/2.0
    ptclTGT = lambda x: 1.0/np.sqrt(np.pi) * x**2 * np.exp(-x**2) / tau * multP
    energyTGT = lambda x: 1.0/np.sqrt(np.pi) * (2.0/3.0) * (1+x**2) * x**2 * np.exp(-x**2) / tau * multE

    #testing this...could Fundamenski be wrong or is HEAT wrong...?
    #energyTGT = lambda t: 1.0/np.sqrt(np.pi) * (2.0/3.0) * (1+(tau/t)**2) * (tau/t)**3 * np.exp(-(tau/t)**2) / tau
    
    s = np.sum(energyTGT(t))

    fig.add_trace(go.Scatter(x=1/x_a, y=ptclTGT(x_a), name="FSM Particle Deposition Norm",
                             line=dict(color='black', width=2, dash='solid')))

    #we multiply by tau and 3/2 to normalize the energy according to Fundamenski Figure 5,
    #where q is given in units of eps_0 / tau
    fig.add_trace(go.Scatter(x=1/x_a, y=energyTGT(x_a), name="FSM Energy Deposition Norm",
                             line=dict(color='black', width=2, dash='dot'),
                             mode='lines+markers', marker_symbol='circle', marker_size=10))
    
    fig.update_xaxes(title=r'$t / \tau_{||} $')

    #integrated ptcl / energy
    intP = integrate.quad(ptclTGT, 1/x_a[0], 1/x_a[-1])[0]
    intE = integrate.quad(energyTGT, 1/x_a[0], 1/x_a[-1])[0]
    print('---')
    print("Analytical Particle Integral: {:0.3f}".format(intP))
    print("Analytical Energy Integral: {:0.3f}".format(intE))
    print("Max FSM ptcl: {:f}".format(np.max(ptclTGT(x_a))))
    print("Max FSM ptcl: {:f}".format(np.max(energyTGT(x_a))))
    print("Timestep for max FSM ptcl: {:f}".format(1/x_a[np.argmax(ptclTGT(x_a))]))
    print("Timestep for max FSM energy: {:f}".format(1/x_a[np.argmax(energyTGT(x_a))]))    
   

#fig.update_layout(title="Window: {:d}".format(window))
fig.update_layout(legend=dict(
    yanchor="top",
    y=0.99,
    xanchor="right",
    x=0.99,
    ),
    font=dict(
        size=20,
    )
    )


#fig.update_xaxes(range=[0,3])
fig.update_yaxes(range=[0,1.0])

if saveMask == True:
    fig.write_image(outPath + svgName)
fig.show()

