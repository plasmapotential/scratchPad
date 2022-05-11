#plots maxT across all PFCs in an openFOAM directory
import plotly.graph_objects as go
import pandas as pd
import os
import numpy as np
#name of each PFC
#root = '/home/tom/results/sparc_1stRuns/sweep7/sparc_000001_sweep7/openFoam/heatFoam/'
root = '/home/tom/HEAT/data/sparc_000001_sweep7/openFoam/heatFoam/'
nombres = [f.name for f in os.scandir(root) if f.is_dir()]

data = []
maxTs = []
for i,name in enumerate(nombres):
    outfile = root+name+'/postProcessing/fieldMinMax1/0/fieldMinMax.dat'
    tmp = pd.read_csv(outfile, header=1, delimiter="\t")
    tmp.columns = tmp.columns.str.strip()
    tmp = tmp.sort_values('field')
    tmp['field'] = tmp['field'].str.strip()
    use = tmp['field']=='T'
    maxTs.append(max(tmp[use]['max'].values))
    data.append(tmp)


print(maxTs)
idxMax = np.argmax(maxTs)
print("Maximum T occurs on PFC: " + nombres[idxMax])
idxMax2 = np.argmax(maxTs[:idxMax]+maxTs[idxMax+1:])
print("2nd Maximum T occurs on PFC: " + nombres[idxMax2])

fig = go.Figure()
df = data[idxMax]
mask = df['field'] == 'T'
t = df[mask].sort_values('# Time')['# Time'].values
varMax = df[mask].sort_values('# Time')['max'].values
varMax = np.insert(varMax, 0, 300)
fig.add_trace(go.Scatter(x=t, y=varMax, name="Ion Optical", line=dict(color='rgb(17,119,51)', width=6, dash='dot'),
                         mode='lines', marker_symbol='cross', marker_size=14))


fig.update_layout(
title="Max T: "+nombres[idxMax],

margin=dict(
    l=100,
    r=100,
    b=100,
    t=100,
    pad=2
),
)

fig.update_layout(legend=dict(
    yanchor="middle",
    y=0.9,
    xanchor="left",
    x=0.1
    ),
    font=dict(
        family="Courier New, monospace",
        size=30,
    )

    )


fig.update_yaxes(title_text="<b>Maximum PFC Temperature [K]</b>")
fig.update_xaxes(title_text="<b>Time [s]</b>")




fig.show()
