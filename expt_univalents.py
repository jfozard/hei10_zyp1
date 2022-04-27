
"""
For experimental comparison with the univalent simulation plot - the unweighted means of the univalent counts from the 7 zyp1 null mutant lines in the Mercier paper are:

Cells with 0 univalents:88.3

Cells with 1 univalent: 10.7%

Cells with 2 univalents: 1%

"""

import numpy as np

import matplotlib.pyplot as plt

import matplotlib as mpl

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


data_prop_uv = np.array([ 88.3, 10.7, 1.0, 0, 0, 0])/100
data_num_uv = [0, 1, 2, 3, 4, 5]

plt.figure()
plt.bar(data_num_uv, data_prop_uv, color='r')
plt.ylabel('Relative frequency')
plt.xlabel('Number of univalents per cell')
plt.xticks([0,1,2,3,4,5])
plt.savefig('expt_univalents.svg')


plt.show()
