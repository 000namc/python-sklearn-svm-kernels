#!/usr/bin/env python

__version__ = "Time-stamp: <2018-12-06 11:55:22 ycopin>"
__author__ = "Yannick Copin <yannick.copin@laposte.net>"

"""
Example of use of TaylorDiagram. Illustration dataset courtesy of Michael
Rawlins.

Rawlins, M. A., R. S. Bradley, H. F. Diaz, 2012. Assessment of regional climate
model simulation estimates over the Northeast United States, Journal of
Geophysical Research (2012JGRD..11723112R).
"""

from taylor import TaylorDiagram
import numpy as NP
import matplotlib.pyplot as PLT
import pandas as pd
import sys
# Reference std
# stdrefs = dict(winter=48.491,
#               spring=44.927,
#               summer=37.664,
#               autumn=41.589)

problem = sys.argv[1]


# Sample std,rho: Be sure to check order and that correct numbers are placed!
data = pd.read_csv('result/' + problem + '.csv')

kernels = list(data.columns)
kernels.remove('dataset')
datas = list(data['dataset'])

samples = {}
for i in range(len(datas)):
    samples[datas[i]] = [[0,0,0] for x in range(len(kernels))]

for name in datas:
    for i, kernel in enumerate(kernels):
        
        temp = data.loc[data['dataset'] == name,:].copy()        
        samples[name][i][0] = float(temp.loc[:,kernel].values[0].split('/')[0])
        samples[name][i][1] = float(temp.loc[:,kernel].values[0].split('/')[1])
        samples[name][i][2] = kernel

        
stdrefs = {}
for data in datas:
    if problem == 'classification':
        stdrefs[data] = 1
    if problem == 'regression':
        stdrefs[data] = 3

# Colormap (see http://www.scipy.org/Cookbook/Matplotlib/Show_colormaps)
# colors = PLT.matplotlib.cm.Set1(NP.linspace(0,1,len(kernels)))
colors = ['#FF722B'
         ,'#FF624E'
         ,'#EB8C71'
         ,'#EBC5A7'
         ,'#FFB459'
         ,'#D032FF'
         ,'#9457FF'
         ,'#B878EB'
         ,'#E9AEEB'
         ,'#FF61CD'
         ,'#07E6FF'
         ,'#2BFFBD'
         ,'#369CFF'
         ,'#F1FF54'
         ,'#C3EB75'
         ,'#B4EBAB'
         ,'#FFDE35'
         ,'#FFCB59'
         ,'#EBCF7A'
         ,'#EBE5B0'
         ,'#FFFEBF']



# Here set placement of the points marking 95th and 99th significance
# levels. For more than 102 samples (degrees freedom > 100), critical
# correlation levels are 0.195 and 0.254 for 95th and 99th
# significance levels respectively. Set these by eyeball using the
# standard deviation x and y axis.

#x95 = [0.01, 0.68] # For Tair, this is for 95th level (r = 0.195)
#y95 = [0.0, 3.45]
#x99 = [0.01, 0.95] # For Tair, this is for 99th level (r = 0.254)
#y99 = [0.0, 3.45]

# 부체모양 직선을 위한 변수들 xx, yy
# 비율에 대한 입력을 집어넣음.
if problem == 'classification':
    xx = NP.arccos(NP.array([0.2, 0.4, 0.6, 0.8]))
if problem == 'regression':
    xx = NP.arccos(NP.array([0.2, 0.4, 0.6, 0.8]))

# xx = NP.arccos(NP.array([0.3 * (x+1) for x in range(9)]))
# yy = [0.0, 1.0]
# 부체모양 직선들의 길이
if problem == 'classification':
    yy = [0, 1]
if problem == 'regression':
    yy = [0, 3]

rects = {}
for i, data in enumerate(datas):
    rects[data] = 321 + i

fig = PLT.figure(figsize=(11,8))
fig.suptitle(problem, size='x-large')

for season in datas:

    dia = TaylorDiagram(stdrefs[season], fig=fig, rect=rects[season],
                        label='Reference',problem = problem)

    for i in range(len(xx)):
        dia.ax.plot([0,xx[i]],yy,color='#829FD9',alpha = 0.5,zorder = 1)
    
    # Add samples to Taylor diagram
    for i,(stddev,corrcoef,name) in enumerate(samples[season]):
        dia.add_sample(stddev, corrcoef,
                       #marker='$%d$' % (i+1),
                       marker='o', ms=5, ls='',
                       #mfc='k', mec='k', # B&W
                       mfc=colors[i], mec=colors[i], # Colors
                       label=name, zorder = 3)

    
    # Add RMS contours, and label them
    
    contours = dia.add_contours(levels=5, colors='0.5') # 5 levels
    dia.ax.clabel(contours, inline=1, fontsize=10, fmt='%.1f')
    
    # Tricky: ax is the polar ax (used for plots), _ax is the
    # container (used for layout)
    dia._ax.set_title(season.capitalize())

# Add a figure legend and title. For loc option, place x,y tuple inside [ ].
# Can also use special options here:
# http://matplotlib.sourceforge.net/users/legend_guide.html

fig.legend(dia.samplePoints,
           [ p.get_label() for p in dia.samplePoints ],
           numpoints=1, prop=dict(size='small'), loc='center')

fig.tight_layout()

PLT.savefig('figure/' + problem + '.png')
