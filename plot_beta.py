
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


mpl.rcParams.update({
    'figure.facecolor': 'none',
    'figure.edgecolor': 'none',
    'font.size': 20,
    'figure.dpi': 72,
    'figure.subplot.bottom' : .15,
    'axes.labelsize':28,
    'savefig.edgecolor': 'none',
    'savefig.facecolor': 'none',
    'svg.fonttype': 'none',

})



x = np.linspace(0, 2, 100)
y = 1/(1+x**12.5)
plt.figure(figsize=(4,3))
plt.plot(x,y)
plt.xlabel('$C/K$')
plt.ylabel('$\\beta/\\beta_C$')
plt.xticks([0,1,2])
plt.yticks([0,0.5,1])
plt.axvline(1, c='r')
plt.xlim(0,2)
plt.ylim(0,1)
plt.savefig('beta_plot.svg')
