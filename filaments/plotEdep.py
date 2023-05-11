#plots deposited energy from a HEAT run CSV output
import numpy as np
import os
import plotly.graph_objects as go

rootPath = '/home/tom/HEAT/data/sparc_000001_filament_1ptFSM/paraview/'

prefix = 'Edep_'
names = sorted([f for f in os.listdir(rootPath) if (os.path.isfile(os.path.join(rootPath, f)) and prefix in f)])
t = np.linspace(0,100e-6, 101)

Edep = np.zeros((len(names)))
for i,n in enumerate(names):
    E = np.genfromtxt(rootPath+n, delimiter=',', comments='#')
    Edep[i] = np.max(E[:,3])



fig = go.Figure()
fig.add_trace(go.Scatter(x=t, y=Edep, name="HEAT Energy Deposition"))
fig.update_xaxes(title="Time [s]")
fig.show()

