

import matplotlib.pyplot as plt
import numpy as np

import  matplotlib as mpl

mpl.rcParams.update({ 
    'figure.facecolor': 'none', 
    'figure.edgecolor': 'none',       
    'font.size': 20, 
    'figure.dpi': 72, 
    'figure.subplot.bottom' : .15, 
    'axes.labelsize':28,
    'savefig.edgecolor': 'none',
    'savefig.facecolor': 'none',
    
})


u_array = []

with open('kymo.dat', 'r') as f:
    for i in range(5):
        next(f)
        next(f)
        u = []
        for j in range(100):
            l = next(f)
            u.append(np.array(list(map(float, l.strip().split(',')))))
        u_array.append(np.vstack(u))

u_array = np.hstack(u_array)
t_array = np.linspace(0, 10, 100)

plt.figure()
plt.plot(t_array, u_array)
plt.xlabel('Time (h)')
plt.ylabel('RI HEI10 (au)')
plt.xlim(0, 10)
plt.ylim(0)
plt.savefig('kymo.svg')
    
