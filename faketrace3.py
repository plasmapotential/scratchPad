import numpy as np
import plotly.graph_objects as go

f1 = '/home/tlooby/HEAT/data/sparc_000001_sweepMEQ_T4_20231206_nominal_fRadDiv70_lq0.6_S0.6/openFoam/heatFoam/T032/Tpeak.csv'
data1 = np.genfromtxt(f1, delimiter=',', skip_header=0)

f2 = '/home/tlooby/HEAT/data/sparc_000001_sweepMEQ_T4_20231206_nominal_fRadDiv70_lq0.6_S0.6/TmaxData_PV.csv'
data2 = np.genfromtxt(f2, delimiter=',', skip_header=1)
print(data2)


fig = go.Figure()
fig.add_trace(go.Scatter(x=data2[:,0], y=data1[1:], name='data1'))
fig.add_trace(go.Scatter(x=data2[:,0], y=data2[:,1], name='data2'))
fig.show()