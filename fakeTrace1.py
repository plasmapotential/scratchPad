#makes a fake trace as a function of time.  good for Ip or RF like plots
import numpy as np
import plotly.graph_objects as go

# Parameters
ramp_up_end_value = 0  # End value of the ramp up
ramp_up_duration = 8   # Duration of the ramp up in seconds
step_duration = 5      # Duration of each step in seconds
step_count = 2         # Number of steps
ramp_down_duration = 7 # Duration of the ramp down in seconds

# Time points
t_ramp_up = np.linspace(0, ramp_up_duration, ramp_up_duration * 100)  # 100 points per second
t_ramp_down = np.linspace(ramp_up_duration + step_duration * step_count, ramp_up_duration + step_duration * step_count + ramp_down_duration, ramp_down_duration * 100)
t_stepwise = np.array([ramp_up_duration + 1.0, ramp_up_duration + 5.0, ramp_up_duration + 6.0, ramp_up_duration + 10.0])
trace_stepwise = np.array([1,1, 2, 2])


# Generate trace data
trace_ramp_up = (ramp_up_end_value / ramp_up_duration) * t_ramp_up
trace_ramp_down = np.linspace(trace_stepwise[-1], 0, ramp_down_duration * 100)

# Combine data
t = np.concatenate([t_ramp_up, t_stepwise, t_ramp_down])
trace = np.concatenate([trace_ramp_up, trace_stepwise, trace_ramp_down])

# Plot using Plotly
fig = go.Figure(data=go.Scatter(x=t, y=trace, name='Seeding Trajectory', mode='lines'))
fig.add_trace(go.Scatter(x=t_stepwise, y=trace_stepwise, name='Control Points', mode='markers',
                                 marker=dict(

                                            size=20,
                                            line=dict(
                                                color='black',
                                                width=2
                                                )
                                            )
                        )
            )
fig.update_layout(xaxis_title='[s]', yaxis_title='Seeding Value', font=dict(family="Arial",size=20,))
fig.update_layout(legend=dict(
    yanchor="top",
    y=0.99,
    xanchor="left",
    x=0.01
))
fig.show()