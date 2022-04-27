
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

import matplotlib as mpl

mpl.rcParams.update({ 
 'figure.facecolor': (1,1,1,0), 
 'figure.edgecolor': (1,1,1,0),
 'font.size': 16, 
 'figure.dpi': 72, 
  'figure.subplot.bottom' : .15, 
  'axes.labelsize':22,
})

import numpy as np

x = np.linspace(0,1,100)
y = interp1d([0,0.3,0.7,1.0],[1.25,1,1,1.25])(x)

plt.figure(figsize=(6,3))
plt.plot(x,y, 'k-')
plt.ylim(0,1.5)
plt.xlim(0,1)
plt.xticks([0,1])
plt.yticks([0,1])
plt.savefig('f_plot.svg')


