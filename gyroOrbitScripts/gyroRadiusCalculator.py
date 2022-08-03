import numpy as np

#deuterium
ionMassAMU = 3.0
#electrons
electronMassAMU = 1.0 / 1836
T_eV = 10000 #eV
B = 12 #T

kg2eV = 5.609e35 #1kg = 5.609e35 eV/c^2
eV2K = 1.160e4 #1ev=1.160e4 K
AMU = 931.494e6 #ev/c^2
kB = 8.617e-5 #ev/K
e = 1.602e-19 # C
c = 299792458 #m/s
mass_eV = ionMassAMU * AMU
Z=1 #assuming isotopes of hydrogen here
vThermal = np.sqrt(2*T_eV/(mass_eV/c**2))
omegaGyro = Z * e * B / (mass_eV / kg2eV)
fGyro = omegaGyro/(2*np.pi)

vPerp = vThermal

rGyro = vPerp / omegaGyro
#rGyro = vPerp / fGyro

print("===Using Gunn Assumptions (vPerp = vThermal):")
print("vThermal: {:f}".format(vThermal))
print("vPerp: {:f}".format(vPerp))
print("omegaGyro: {:f}".format(omegaGyro))
print("rGyro: {:f}".format(rGyro))


print("===Using vSpace Division (vPerp = v||):")
vPhase = np.radians(45)
vPerp = vThermal * vPhase
rGyro = vPerp / omegaGyro
print("vThermal: {:f}".format(vThermal))
print("vPerp: {:f}".format(vPerp))
print("omegaGyro: {:f}".format(omegaGyro))
print("rGyro: {:f}".format(rGyro))

print("===CUSTOM")
vPhase = np.radians(89) #rad
vSlice = vThermal #m/s
vPerp = vSlice * np.cos(vPhase)
rGyro = vPerp / omegaGyro
print("vPerp: {:f}".format(vPerp))
print("rGyro: {:f}".format(rGyro))

#print(": {:f}".format())
