import numpy as np
import plotly.graph_objects as go

T_mIn = 293 #K
T_mOut = 353 #K
T_sOut = 273+500 #K
deltaT = T_mOut - T_mIn
cp = 4181 #J/kg K for h20 @323K
Psum = 1e4
mu = 352e-6 #Pa s
k = 0.67 #W/(m K)
Pr = 2.2



#calculate flow rate
mDot = Psum / (cp * deltaT)
print("Required flow rate: {:f} kg / s".format(mDot))


#calculate L given D
D=0.002 #m


#mDot = 0.01
#D = 0.06
Re = 4 * mDot / (np.pi * D * mu)
print("Reynolds Number = {:f}".format(Re))

#Dittus-Boelter equation for Nusselt Number
Nu = 0.023 * Re**0.8 * Pr**0.4
print("Nusselt Number = {:f}".format(Nu))

#Heat transfer coefficient
h = Nu * k / D
print("Heat transfer coefficient: {:f} W/(m^2 K)".format(h))

qAvg = (T_sOut - T_mOut) * h
print("Constant Flux: {:f} MW/m^2".format(qAvg/1e6))






L = Psum / (np.pi * D * qAvg)
print("D = {:f} m".format(D))
print("L = {:f} m".format(L))
L_probe = 0.25
print("N tubes: {:f}".format(L / L_probe))

A_bulk = np.pi * ((0.04/2)**2 - (0.031/2)**2)
A_tube = np.pi * (D/2)**2
print(A_bulk / A_tube)


#total tubes that we could have:
print("======")
D_inner = 0.031 #m
P_in = np.pi * D_inner #inner perimeter
N = int(P_in / 0.003)
print("Max N of tubes = {:f}".format(N))
L_allTubes = N * L_probe
print("Total length of cooling channels: {:f} m".format(L_allTubes))
A_allTubes = np.pi * D * L_allTubes
print("Total area of all tubes: {:f}".format(A_allTubes))
qAllTubes = Psum / A_allTubes
print("Averaged flux across all tubes: {:f} MW/m^2".format(qAllTubes/1e6))

##plot D vs L
#d = np.linspace(0.001, 0.01, 100)
#l = np.linspace(1.0, 10, 100)
## Creating 2-D grid of features
#[X, Y] = np.meshgrid(d, l)
#Z = Psum / (np.pi * X * Y)
#fig = go.Figure(data =
#    go.Contour(z=Z, x=d, y=l, colorscale="Electric",
#                contours=dict(
#                    coloring ='heatmap',
#                    ),
#                ))
#fig.show()


# P_{sum} = q''_{conv} \cdot A_{pipe} = \dot{m} c_p (T_{m,i} - T_{m,o})
#\rightarrow \dot{m} = \frac{P_{sum}}{c_p \Delta T}
#\Delta T = \text{60K}
#\rightarrow \dot{m} = \text{3.986287 kg/s}
#Re_D = \frac{4 \dot{m}}{\pi \mu D} = 7209515
#Nu_{D}=0.023 \cdot \text{Re}^{0.8} \cdot \text{Pr}^{0.4}
#h = \frac{Nu_{D} \cdot k}{D} = 3236432 \text{ W/(m^2 K)}
#q_{surf} = h (T_{s,o} - T_{m,o}) = 16.2 \text{MW} / \text{m}^2
#P_{sum} = q_{surf} \cdot A = \pi D L \\ \rightarrow L = \frac{P_{sum}}{q_{surf}  \pi D}
#
