import pandas as pd
import plotly.graph_objects as go
import numpy as np

# Function to read CSV and plot
def plot_1d_function_paraviewOutput(csv_file, fig=None, offset=0.0, tMax = None):
    if fig==None:
        fig= go.Figure()
    # Read the data from the CSV file
    df = pd.read_csv(csv_file)

    # Extract X and Y values
    x_values = df['Time']
    if tMax is not None:
        x_values = np.linspace(0,tMax,len(x_values))
    y_values = df['max(temperature)'].values

    # Add the line plot
    fig.add_trace(go.Scatter(x=x_values + offset, y=y_values-273.0, mode='lines', 
                             marker_symbol='square',  marker_size=8, name='HEAT Output'))


    return fig

# Function to read CSV and plot
def plot_1d_function_DIAGOutput(file, fig=None, offset=0.0, tMax = None):
    if fig==None:
        fig= go.Figure()
    # Read the data from the CSV file
    data = np.genfromtxt(file, delimiter=' ')
    # Extract X and Y values
    x_values = data[:,0]
    y_values = data[:,1]

    if tMax is not None:
        x_values = np.linspace(0,tMax,len(x_values))
    # Add the line plot
    fig.add_trace(go.Scatter(x=x_values, y=y_values, mode='lines+markers', 
                             marker_symbol='cross',  marker_size=8, name='TC'))


    return fig



# Usage example
PVpath = '/home/tlooby/results/AUGvalidation/39231/HEAToutput/1D_temp_fRadDiv64_elmer_back.csv'  # Replace with your CSV file path
DIAGpath = '/home/tlooby/results/AUGvalidation/39231/diagnosticData/TC10.txt'

tMax = 5.7
offset = 2.28
fig = go.Figure()
fig = plot_1d_function_DIAGOutput(DIAGpath, fig, offset)
fig = plot_1d_function_paraviewOutput(PVpath, fig, offset, tMax)

# Add labels and titles
fig.update_layout(
    title='1D Function Plot',
    xaxis_title='idx',
    yaxis_title='T',
    font=dict(
        family="Arial",
        size=20,
        color="Black"
    ),
)
# Show plot
fig.show()

