
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

import matplotlib as mpl

mpl.rcParams.update({ #'figure.figsize': (6.0,4.0),                                                                                                                                                                                                                                                           
 'figure.facecolor': (1,1,1,0), # play nicely with white background in the Qt and notebook                                                                                                                                                                                                                    
 'figure.edgecolor': (1,1,1,0),
 'font.size': 16, # 12pt labels get cutoff on 6x4 logplots, so use 10pt.                                                                                                                                                                                                                                      
 'figure.dpi': 72, # 72 dpi matches SVG/qtconsole                                                                                                                                                                                                                                                             
  'figure.subplot.bottom' : .15, # 10pt still needs a little more room on the xlabel                                                                                                                                                                                                                          
  'axes.labelsize':22,
#    'text.usetex':True,
})

import numpy as np

x = np.linspace(0,1,100)
y = interp1d([0,0.3,0.7,1.0],[1.25,1,1,1.25])(x)

plt.figure(figsize=(6,3))
plt.plot(x,y, 'k-')
#plt.xlabel('Relative position along bivalent')
#plt.ylabel('RI density')
plt.ylim(0,1.5)
plt.xlim(0,1)
plt.xticks([0,1])
plt.yticks([0,1])
plt.savefig('../output/simulation_figures/f_plot.svg')


