#aoiAlongLim.py
#Description:   plot separatrix + 1 lambda_q away from EQ
#Date:          20230505
#engineer:      T Looby
import sys
import os
import numpy as np
import scipy.interpolate as scinter
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import shutil

#set up python environment
#dev machine
EFITPath = '/home/tom/source'
HEATPath = '/home/tom/source/HEAT/github/source'
#appImage machine
#if you extract appImage you can point to these files directly in:
# <APPDIR>/usr/src/
#where AppDir is location of extracted files
#
#EFITPath = '/home/tom/source/HEAT/AppDir/usr/src'
#HEATPath = '/home/tom/source/HEAT/AppDir/usr/src'
sys.path.append(EFITPath)
sys.path.append(HEATPath)
import MHDClass
import GUIscripts.plotly2DEQ as pEQ


rootPath = '/home/tom/work/CFS/GEQDSKs/MEQ_20230501/tmp/'
name = 'sparc_1355.EQDSK_v3b_PsiOver2pi_negIp_negBt_negFpol'
try:
    f = rootPath+name
    MHD = MHDClass.setupForTerminalUse(gFile=f)
    ep = MHD.ep
except: #EFIT reader is very specific about shot names
    newf = rootPath+'g000001.00001'
    shutil.copyfile(f, newf)
    MHD = MHDClass.setupForTerminalUse(gFile=newf)


fig = pEQ.makePlotlyEQDiv(1, 1, 'sparc', MHD.ep)
fig = pEQ.highlightPsiFromSep(fig, MHD.ep, 0.0003)
fig.show()
