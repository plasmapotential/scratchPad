#Convolution in 3D using tensorflow or scipy

import numpy as np
from scipy import signal
from scipy import misc
from scipy.spatial import Delaunay
import tensorflow as tf
import plotly.graph_objects as go
import plotly.express as px

#background grid
Y,X,Z = np.mgrid[0:100,0:100,0:100]
x = X.ravel()
y = Y.ravel()
z = Z.ravel()
#points = np.vstack([x,y,z]).T
#x = np.linspace(0,99,100)
#z = np.linspace(0,99,100)
#y = np.linspace(0,99,100)


#one box
#sq1 = np.zeros((1000000)).reshape(100,100,100)
#sq1[40:60,40:60,40:60] = 1

#two boxes
sq1 = np.zeros((1000000)).reshape(100,100,100)
sq1[:40,:40,:40] = 1
sq1[60:,:40,:40] = 1

#one box only surface
#sq1 = np.zeros((1000000)).reshape(100,100,100)
#sq1[40:60,40,40:60] = 1 #xz plane
#sq1[40:60,60,40:60] = 1 #xz plane
#sq1[40,40:60,40:60] = 1 #yz plane
#sq1[60,40:60,40:60] = 1 #yz plane
#sq1[40:60,40:60,40] = 1 #xy plane
#sq1[40:60,40:60,60] = 1 #xy plane


def create_spherical_mask(h, w, l, center=None, radius=None, minRad=None):
    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2), int(l/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], center[2], w-center[0], h-center[1], l-center[2])

    Y, X, Z = np.ogrid[:h, :w, :l]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2 + (Z - center[2])**2)

    #solid sphere
    if minRad == None:
        mask = dist_from_center <= radius
    #ring of width: radius - minRad
    else:
        mask = np.logical_and(dist_from_center <= radius, dist_from_center >= minRad)
    return mask


def create_cylinder_mask(h,w,l,r,minRad,x0,y0):
    Y, X, Z = np.mgrid[:h, :w, :l]
    dist_from_center = np.sqrt((X-x0)**2 + (Y-y0)**2)
    mask = np.logical_and(dist_from_center <= r, dist_from_center >= minRad)
    return mask





#kernel = create_spherical_mask(10,10,10,np.array([5,5,5]), 5, 2)
kernel = create_cylinder_mask(20,20,10,5.5,3,5,5)
#===TF convolution
tf1 = tf.constant(sq1, dtype=tf.float32)
tfK = tf.constant(kernel, dtype=tf.float32)
#reshape to 5D tensor for TF
tf1 = tf1[tf.newaxis, :, :, :, tf.newaxis]
tfK = tfK[:-1, :-1, :-1, tf.newaxis, tf.newaxis]
conv = tf.nn.conv3d(tf1, tfK, strides=[1, 1, 1, 1, 1], padding="SAME")

#plot the background
#fig = go.Figure()
#use = np.where(sq1.ravel() > 0)[0]
#fig=go.Figure(data=[go.Scatter3d(x=x[use], y=y[use], z=z[use], mode='markers',
#                    marker=dict(size=12,
#                                color=sq1.ravel()[use],                # set color to an array/list of desired values
#                                colorscale='Viridis',   # choose a colorscale
#                                opacity=0.0
#                                )
#                    )]
#            )


#plot the kernel
#fig = go.Figure()
#use = np.where(kernel.ravel() > 0)[0]
#Y,X,Z = np.mgrid[0:20,0:20,0:10]
#x = X.ravel()
#y = Y.ravel()
#z = Z.ravel()
#fig=go.Figure(data=[go.Scatter3d(x=x[use], y=y[use], z=z[use], mode='markers',
#                    marker=dict(size=12,
#                                color=kernel.ravel()[use],                # set color to an array/list of desired values
#                                colorscale='Viridis',   # choose a colorscale
#                                opacity=0.0
#                                )
#                    )]
#            )



#plot convolution
use = np.where(sq1.ravel() > 0)[0]
fig=go.Figure(data=[go.Scatter3d(x=x[use], y=y[use], z=z[use], mode='markers',
                    marker=dict(size=12,
                                color=conv.numpy()[0,:,:,:,0].ravel()[use],                # set color to an array/list of desired values
                                colorscale='Plasma',   # choose a colorscale
                                opacity=0.0
                                )
                    )]
            )



fig.show()
