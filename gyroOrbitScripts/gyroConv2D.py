#Convolution in 2D using tensorflow or scipy

import numpy as np
from scipy import signal
from scipy import misc
import tensorflow as tf
import plotly.graph_objects as go
import plotly.express as px


#square1 = np.array([[-1,-1],
#                    [-1,1],
#                    [1,1],
#                    [1,-1]])
#square2 = np.array([[-2,-2],
#                    [-2,2],
#                    [2,2],
#                    [2,-2]])
#
#grad = signal.convolve2d(square1, square2, boundary='symm', mode='same')
#fig = go.Figure()
#fig.add_trace(go.Scatter(x=square1[:,0], y=square1[:,1], name="SQ1", line=dict(color='royalblue', width=4, dash='solid'),
#                         mode='lines', marker_symbol='cross', marker_size=6))
#fig.show()

#square
#sq1 = np.zeros((10000), dtype=np.uint8).reshape(100,100)
#sq1[40:60,40:60] = 1

#two squares in corners
sq1 = np.zeros((10000), dtype=np.uint8).reshape(100,100)
sq1[50:,:40] = 3
sq1[50:,60:] = 3
sq1[60:,:40] = 2
sq1[60:,60:] = 2
sq1[80:,:40] = 1
sq1[80:,60:] = 1

#two square outline
outline = np.array([
                    [0,0],
                    [0,50],
                    [40,50],
                    [40,0],
                    [60,0],
                    [60,50],
                    [99,50],
                    [99,0]
                    ])


#square 2
sq2 = np.zeros((16), dtype=np.uint8).reshape(4,4)
sq2[1:2,1] = 1
sq2[1:2,2] = 1


def create_circular_mask(h, w, center=None, radius=None, minRad=None):
    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    #solid circle
    if minRad == None:
        mask = dist_from_center <= radius
    #ring of width: radius - minRad
    else:
        mask = np.logical_and(dist_from_center <= radius, dist_from_center >= minRad)
        #mask = np.logical_and(dist_from_center <= radius, X > center[0])
    return mask

kernel = create_circular_mask(20,20,np.array([10,10]), 0, 0)

#===using tensorflow
tf1 = tf.constant(sq1, dtype=tf.float32)
tf2 = tf.constant(sq2, dtype=tf.float32)
tfK = tf.constant(kernel, dtype=tf.float32)
#reshape to 4D tensor for TF
tf1 = tf1[tf.newaxis, :, :, tf.newaxis]
tf2 = tf2[:-1, :-1, tf.newaxis, tf.newaxis]
tfK = tfK[:-1, :-1, tf.newaxis, tf.newaxis]
#conv = tf.nn.conv2d(tf1, tfK, strides=[1, 1, 1, 1], padding=[[0, 0], [9,9], [10,8], [0, 0]])
conv = tf.nn.conv2d(tf1, tfK, strides=[1, 1, 1, 1], padding='SAME')
image = conv.numpy()[0,:,:,0]
image = np.array(list(reversed(image)))

#plot the background
#fig = px.imshow(sq1, color_continuous_scale='gray')

#plot the kernel
#fig = px.imshow(kernel, color_continuous_scale='gray')

#plot convolution
fig = px.imshow(image, color_continuous_scale='Plasma', origin='lower')
fig.add_trace(go.Scatter(x=outline[:,0], y=outline[:,1], mode='lines', line=dict(color='#16FF32', width=4)))
fig.show()

#plot outline overlaid on conv










#===using scipy
#conv = signal.convolve2d(sq1, sq2, boundary='symm', mode='same')
#fig = px.imshow(conv)
#fig.show()

#Example for 2D conv from scipy
#ascent = misc.ascent()
#scharr = np.array([[ -3-3j, 0-10j,  +3 -3j],
#                   [-10+0j, 0+ 0j, +10 +0j],
#                   [ -3+3j, 0+10j,  +3 +3j]]) # Gx + j*Gy
#grad = signal.convolve2d(ascent, scharr, boundary='symm', mode='same')
#fig = px.imshow(ascent)
#fig = px.imshow(np.absolute(grad))
#fig = px.imshow(np.angle(grad))
#fig.show()
