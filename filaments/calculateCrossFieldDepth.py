#calculates how far across flux surfaces a filament can travel 
# between two adjacent limiters
#
#                    ^
#                    |
#                    | vR
#   ---------------->|
#         vPar
#                    
import numpy as np
import plotly.graph_objects as go

#angular distance between two limiter edges
a = np.radians(20.0) #degrees

#major radius at limiter edge
R = 2437.3 #[mm]

#magnetic field components
Bt = 9.4 #[T]
Bp = 2.48 #[T]
B = np.sqrt(Bt**2 + Bp**2)

#connection length between the two limiters
L = R * a * B / Bt

#user can use the following lines to use vThermal for v||, or can 
#comment them out and write their own v||
ionMassAMU = 2.515
T_eV = 1000.0 #eV
#constants
kg2eV = 5.609e35 #1kg = 5.609e35 eV/c^2
eV2K = 1.160e4 #1ev=1.160e4 K
AMU = 931.494e6 #ev/c^2
kB = 8.617e-5 #ev/K
e = 1.602e-19 # C
c = 299792458 #m/s
mass_eV = ionMassAMU * AMU
Z=1 #assuming isotopes of hydrogen here

#if you want to override, override this value
vPar = np.sqrt(T_eV/(mass_eV/c**2))

#radial filament velocity
vR = 500.0 #[m/s]

#now calculate the depth the filament penetrates across
#flux surfaces between two limiters
r = L * np.tan( np.arctan2(vR,vPar) )

print("RMS Parallel Velocity: {:f} [m/s]".format(vPar))
print("Distance across flux surfaces: {:f} [mm]".format(r))

scanVR = True
if scanVR == True:
    vR = np.linspace(0.0, 3000.0, 100) #[m/s]
    #now calculate the depth the filament penetrates across
    #flux surfaces between two limiters
    r = L * np.tan( np.arctan2(vR,vPar) )

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=vR, y=r,
                    mode='lines',
                    line={'width':5},
                    name='Cross Field Depth'),
                )
    
    #if you want a line at the max allowable depth
    fig.add_hline(y=6.0, line_width=3, line_dash="dash", line_color="red")

    fig.update_xaxes(title_text="Filament Radial Velocity [m/s]")
    fig.update_yaxes(title_text="Distance Across Flux Surfaces [mm]")

    fig.update_layout(
        font=dict(size=18),
        )

    fig.show()