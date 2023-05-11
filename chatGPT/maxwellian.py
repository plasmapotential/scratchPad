#plots a maxwellian in 1D

import numpy as np
import plotly.graph_objects as go
import scipy.integrate as integrate

#unit conversions
kg2eV = 5.609e35 #1kg = 5.609e35 eV/c^2
eV2K = 1.160e4 #1ev = 1.160e4 K
eV2J = 1.602e-19 #1eV = 1.602e-19 J
#constants
ionMassAMU=2.515
AMU = 931.494e6 #ev/c^2
kB = 8.617e-5 #ev/K
e = 1.602e-19 # C
c = 299792458 #m/s
diamag = -1 #diamagnetism = -1 for ions, 1 for electrons
mass_eV = ionMassAMU * AMU
Z=1 #assuming isotopes of hydrogen here



# Define the 3D Maxwellian energy distribution function
def maxwellian_E(v, m, T_eV):
    E = 0.5 * m/c**2 *v**2
    prefactor = 2.0/np.sqrt(np.pi) * (1/T_eV**(3.0/2.0)) * E
    return prefactor * np.exp(-E / (T_eV))

## Define the 1D Maxwellian energy distribution function
#def maxwellian_E1D(v, m, T_eV):
#    E = 0.5 * m/c**2 *v**2
#    #prefactor = m/c**2 * np.sqrt(E/T_eV)
#    prefactor = np.sqrt(E / (4*np.pi*T_eV))
#    return prefactor * np.exp(-E / (T_eV))


# Define the 1D Maxwellian speed distribution function
def maxwellian_E1D(v, m, T_eV):
    E = 0.5 * m/c**2 *v**2
    prefactor = (m/c**2 / (2 * np.pi * T_eV))**(1/2) * E
    return prefactor * np.exp(-m/c**2 * v**2 / (2 * T_eV))


# Define the 1D Maxwellian velocity distribution function
def maxwellian_V(v, m, T_eV):
    prefactor = (m/c**2 / (2 * np.pi * T_eV))**(1/2)
    return prefactor * np.exp(-m/c**2 * v**2 / (2 * T_eV))


# Define the plasma parameters (mass and temperature)
T_eV = 10000  # Plasma temperature in eV

vMax = 3 * np.sqrt(2*T_eV / (mass_eV/c**2))


# Generate a range of velocities
v = np.linspace(-vMax, vMax, 1000)  # Adjust the range and number of points as needed
vE = np.linspace(0, vMax, 10000)  # Adjust the range and number of points as needed

# Calculate the distribution function values
f_v = maxwellian_V(v, mass_eV, T_eV)
f_E = maxwellian_E(vE, mass_eV, T_eV)
f_E1D = maxwellian_E1D(vE, mass_eV, T_eV)

vTot = np.sum(f_v[1:]*np.diff(v))
print(vTot)
eTot = np.sum(f_E[1:]*np.diff(0.5 * mass_eV/c**2 * vE**2))
print(eTot)
e1DTot = np.sum(f_E1D[1:]*np.diff(0.5 * mass_eV/c**2 * vE**2))
print(e1DTot)




avgSpeed3D = np.sqrt(8*T_eV / (np.pi* mass_eV/c**2))
print("Theoretical Average 3D Speed: {:f}".format(avgSpeed3D))
print("Numerical Average 3D Speed: {:f}".format(vE[np.argmax(f_E)]))
avgSpeed1D = np.sqrt(2*T_eV / (np.pi* mass_eV/c**2))
print("Theoretical Average 1D Speed: {:f}".format(avgSpeed1D))
print("Numerical Average 1D Speed: {:f}".format(vE[np.argmax(f_E1D)]))

# Create the plot
fig = go.Figure()

fig.add_trace(go.Scatter(x=v, y=f_v, mode='lines', name='1D Maxwellian V||'))
fig.add_trace(go.Scatter(x=vE, y=f_E, mode='lines', name='3D Maxwellian Energy'))
fig.add_trace(go.Scatter(x=vE, y=f_E1D, mode='lines', name='1D Maxwellian Energy'))


fig.update_layout(
    title='1D Maxwellian Distribution',
    xaxis_title='Velocity (m/s)',
    yaxis_title='Probability Density',
    yaxis_exponentformat='e',
)

# Show the plot
fig.show()
