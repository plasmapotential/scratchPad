import numpy as np
import plotly.graph_objects as go
x = ['B1', 'B2', 'B3', 'I1', 'I2', 'I3', 'I4']
xNum = np.arange(8)
y1 = np.array([np.nan, np.nan, np.nan, 10, 20, 30, 20, 10])
y2 = np.array([10, 50, 30, np.nan, np.nan, np.nan, np.nan, np.nan])

# calc the trendline
z1 = np.polyfit(xNum[3:], y1[3:], 1)
p1 = np.poly1d(z1)
z2 = np.polyfit(xNum[0:3], y2[0:3], 1)
p2 = np.poly1d(z2)
print(z1)

fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=y1, mode='lines+markers', name="I Data"))
fig.add_trace(go.Scatter(x=x, y=y2, mode='lines+markers', name="B Data"))
fig.add_trace(go.Scatter(x=x[3:], y=p1(xNum[3:]), mode='lines+markers', name="I Trend"))
fig.add_trace(go.Scatter(x=x[0:3], y=p2(xNum[0:3]), mode='lines+markers', name="B Trend"))

fig.update_layout(legend=dict(
    yanchor="middle",
    y=0.9,
    xanchor="right",
    x=0.80
    ),
    )
fig.update_layout(
#title="Temperature Probe Time Evolution",
xaxis_title="DAYS",
yaxis_title="SCORE",
font=dict(
    family="Arial",
    size=24,
    color="Black"
),
margin=dict(
    l=5,
    r=5,
    b=5,
    t=5,
    pad=2
),
)


fig.show()
