#performs 2D regression analysis on CSV file for Tpeak as a function of
#Psol and lambda_q


import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scipy.interpolate as interp
import matplotlib.pyplot as plt

f = '/home/tom/HEATruns/SPARC/time2Recryst/lq_Psol_Tscan.csv'
df = pd.read_csv(f)


xRaw = np.unique(df['PSOL [MW]'].values)
yRaw = np.unique(df['lq [mm]'].values)
zRaw = df['Tpeak [K] '].values.reshape(len(yRaw), len(xRaw))

raw_data = np.vstack([df['PSOL [MW]'].values, df['lq [mm]'], df['Tpeak [K] '].values]).T

Tfunc = interp.RectBivariateSpline(xRaw, yRaw, zRaw.T, ky=2, kx=2)

x = np.linspace(min(xRaw),max(xRaw), 100)
y = np.linspace(min(yRaw),max(yRaw), 100)
z = Tfunc(x,y).T

X,Y = np.meshgrid(x,y)

raw_data = np.vstack([X.flatten(), Y.flatten(), z.flatten()]).T


#Normalize if norm_flag is set to 1
norm_flag = 0
if norm_flag==1:
   #Get mins and maxes column-wise
   maxval = []
   minval = []
   for i in range(raw_data.shape[1]-1):
      maxval.append(raw_data[:,i].max(axis=0))
      minval.append(raw_data[:,i].min(axis=0))
   #Normalize between 0 and 1
   for j in range(1,raw_data.shape[1]-1):
      for i in range(raw_data.shape[0]):
         if minval[j] == maxval[j]:
           raw_data[i,j] = 1
         else:
            raw_data[i,j] = (raw_data[i,j] - minval[j])/(maxval[j] - minval[j])


#===
#Regression algorithm
#===

#polynomial degree
degree = 2
#Number of Features
D = 2
#Number of Datapoints
N = len(raw_data)
#Features / inputs
X_mat = np.zeros((N, (D)*(degree)+1))
#Targets
targets = raw_data[:,2]
#Number of Training Sets (75%)
ntrain = int(len(targets)*0.75)



#Build X array based upon polynomial degree
colnum = 0
for i in range(D):
    for j in range(degree):
        X_mat[:,colnum] = raw_data[:,i]**(j+1)
        colnum += 1
#0 order term
X_mat[:,-1] = 1


#Training Data
X_tr = X_mat[0:ntrain]
target_tr = targets[0:ntrain]
#Test Data
X_te = X_mat[ntrain+1::]
N_test = float(N - ntrain)

#W = (X^T X)^-1 (X^T) (target)
W = np.matmul( np.matmul( np.linalg.inv( np.matmul(X_tr.T,X_tr) ), X_tr.T ) ,target_tr)

#predict T for given Psol and lq
Ptest = 7.0
lqTest = 1.5
Xtest1D = np.array([Ptest, lqTest])
Ntest = 1

#Build X test array based upon polynomial degree
Xtest = np.zeros((Ntest, (D)*(degree)+1))
colnum = 0
for i in range(D):
    for j in range(degree):
        Xtest[:,colnum] = Xtest1D[i]**(j+1)
        colnum += 1
#0 order term
Xtest[:,-1] = 1



Ttest = np.matmul(Xtest, W)

print("For inputs")
print(Xtest1D)
print("predicted T")
print(Ttest)
print("coefficients:")
print(W)




#=== for 1st order polynomials (lines) we can invert function
lq = 1.5
Tmax = 1627.0
if degree == 1:
    Pmax = (W[-1] - Tmax + W[1]*lq) / -W[0]
    print("Maximum P:")
    print(Pmax)





plotMask = False
if plotMask == True:
    fig = go.Figure()
    fig.add_trace(
            go.Contour(
                x=x,y=y,z=z,
                contours=dict(
                coloring ='heatmap',
                showlabels = True, # show labels on contours
                labelfont = dict( # label font properties
                    size = 12,
                    color = 'white',
                    )
                ),
                colorbar=dict(title='Peak Temperature [K]') ,
            )
        )
    fig.update(

        layout=go.Layout(
            xaxis=dict(title='$P_{SOL} [MW]$'),
            yaxis=dict(title='$\lambda_q [mm]$'),
            font=dict(
                size=18,
                )
            )
        )
    fig.show()


plotMask2 = False
if plotMask2 == True:
    symbols = ['x', 'star', 'diamond']
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xRaw, y=df['Tpeak [K] '].values[0:7], name='$\lambda_q=0.3mm$',
                    marker_symbol=symbols[0], marker_size=8))
    fig.add_trace(go.Scatter(x=xRaw, y=df['Tpeak [K] '].values[7:14], name='$\lambda_q=0.9mm$',
                    marker_symbol=symbols[1], marker_size=8))
    fig.add_trace(go.Scatter(x=xRaw, y=df['Tpeak [K] '].values[14:], name='$\lambda_q=1.5mm$',
                    marker_symbol=symbols[2], marker_size=8))
    fig.update(
        layout=go.Layout(
            xaxis=dict(title='$P_{SOL} [MW]$'),
            yaxis=dict(title='$\lambda_q [mm]$'),
            font=dict(
                size=18,
                )
            )
        )


    fig.show()
