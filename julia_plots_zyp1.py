import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib as mpl
import csv
import random
from scipy.stats import poisson
import numpy.random as npr
import os

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

import statsmodels.api as sm

def lin_fit(x, y, min_x=None, of=None, r2=False):
    X = np.array(x)
    Y = np.array(y)
    X2 = sm.add_constant(X)
    est = sm.OLS(y, X2)
    est2 = est.fit()
    if of is None:
        print(est2.summary())
    else:
        print(est2.summary(), file=of)
    if min_x is None:
        X = np.linspace(np.min(X), np.max(X), 100)
    else:
        X = np.linspace(min_x, np.max(X), 100)
    X2 = sm.add_constant(X)
    yy = est2.predict(X2)
    if not r2:
        return X, yy
    else:
        return X, yy, est2.rsquared

LARGE_FS=32

import pickle

import statsmodels.api as sm

def pc(v, alpha=10):
    idx = (v>=alpha)
    return idx


def read_file(fn):
    all_data = []
    head = None
    try:
        with open(fn,'r') as f:
            
            try:
                nbuf = []
                head = next(f)
                head = head[head.index('(')+1:head.index(')')]
                head = head.replace(' ', '')
                while True:
                    nbuf = []
                    L_array = list(map(float, next(f).strip().split(',')))
                    x_arrays = []
                    u_arrays = []
                    for q in range(5):
                        x_arrays.append(list(map(float, next(f).strip().split(','))))
                        u_arrays.append(list(map(float, next(f).strip().split(','))))
                    
                    all_data.append((L_array, x_arrays, u_arrays))
            except StopIteration:
                pass
    except IOError:
        pass


    head_dict = dict([u.split('=') for u in head.split(',')])
    del head_dict['start']
    return head_dict, all_data


import os


def main(input_file, prefix):
    head, data = read_file(input_file)

    # Now make histograms
    n_co_chr = [ [] for q in range(5) ]
    co_rel_pos_chr = [ [] for q in range(5) ]
    co_int_chr = [ [] for q in range(5) ]

    spacing = []

    intensities_all = []

    foci_intensities = {i:[] for i in range(20) }
    
    for i in range(len(data)):
        L_array = data[i][0]
        x_arrays = data[i][1]
        u_arrays = data[i][2]
        cell_tot_int = 0
        n_co = 0
        for q in range(5):
            u = np.array(u_arrays[q])
            p = pc(u)            
            cell_tot_int += np.sum(u[p])
            n_co += len(u[p])

        cell_tot_int /= n_co

        for q in range(5):
            L = L_array[q]
            x = np.array(x_arrays[q])
            u = np.array(u_arrays[q])
            p = pc(u)

            n_co = np.sum(p)

            foci_intensities[n_co] += list(u[p]/cell_tot_int)
            
            n_co_chr[q].append(np.sum(p))
            co_rel_pos_chr[q].append(x[p]/L)
            co_int_chr[q].append(u[p]/cell_tot_int)
            if len(x[p])>1:
                spacing += list(np.diff(x[p])/L)
            
    for q in range(5):
        plt.figure()

        sum_n = len(n_co_chr[q])
        
        d = np.bincount(n_co_chr[q], minlength=11)[:11]
        plt.bar(np.arange(11), d/sum_n)
        plt.xticks(np.arange(11))
        mean_n = np.mean(n_co_chr[q])
        var_n = np.var(n_co_chr[q], ddof=1)
        plt.text(0.7, 0.9, f'$\mu={mean_n:.2f}$', transform=plt.gca().transAxes)
        plt.text(0.7, 0.7, f'$S^2={var_n:.2f}$', transform=plt.gca().transAxes)

        plt.plot(np.arange(11), poisson(mean_n).pmf(np.arange(11)), 'g-o')
        plt.xlabel('CO per CP')
        plt.ylabel('Relative frequency')
        plt.savefig(f'{prefix}-ch-{q}-number.svg')
        plt.close()
        plt.figure()
        pos = np.concatenate(co_rel_pos_chr[q])
        print(pos)
        plt.hist(pos, bins=np.linspace(0,1,10))
        plt.xlabel('Relative position along CP')
        plt.ylabel('Count')
        plt.xlim(0,1)
        plt.savefig(f'{prefix}-ch-{q}-relpos.svg')
        plt.close()
        plt.figure()
        pos = np.concatenate(co_rel_pos_chr[q])
        print(pos)
        plt.hist(pos, bins=np.linspace(0,1,10))
        plt.xlabel('Relative position along CP')
        plt.ylabel('Count')
        plt.xlim(0,1)
        plt.savefig(f'{prefix}-ch-{q}-relpos.svg')
        plt.close()
        intensities = np.concatenate(co_int_chr[q])
        intensities_all += list(intensities)
        
        plt.figure()
        pos = np.concatenate(co_rel_pos_chr[q])
        pos = np.minimum(pos, 1-pos)

        xx, yy, r2 = lin_fit(pos, intensities, r2=True)
        
        plt.plot(pos, intensities, 'bo', alpha=0.03)
        plt.plot(xx, yy, 'r-')
        plt.xlabel('Relative position along CP')
        plt.ylabel('Intensities')
        plt.title(f'r2={r2:.2f}')
        plt.xlim(0,1)
        plt.savefig(f'{prefix}-ch-{q}-pos-int.svg')
        plt.close()

    plt.figure()
    plt.hist(intensities_all, bins=np.linspace(0, 2, 10))
    plt.savefig(f'{prefix}-intensities.svg')
    plt.close()
        
    rel_pos = [x for y in co_rel_pos_chr for z in y for x in z ]

    plt.figure()
    plt.hist(rel_pos, bins=np.linspace(0,1,11), density=True)
    plt.xlabel('Relative focus position along CP')
    plt.ylabel('Frequency density')
    plt.xlim(0,1)
    plt.savefig(f'{prefix}-relpos.svg')
    plt.close()

    
    no_co_cell = [ sum(p) for p in zip(*n_co_chr) ]
    plt.figure()
    d = np.bincount(no_co_cell, minlength=31)[:31]
    plt.bar(np.arange(31), d/np.sum(d))
    plt.xticks([0,10,20,30])
    plt.xlim(0, 30)
    mean_n = np.mean(no_co_cell)
    var_n = np.var(no_co_cell, ddof=1)
    plt.text(0.7, 0.9, f'$\mu={mean_n:.2f}$', transform=plt.gca().transAxes)
    plt.text(0.7, 0.7, f'$S^2={var_n:.2f}$', transform=plt.gca().transAxes)
#    plt.plot(np.arange(31), len(no_co_cell)*poisson(mean_n).pmf(np.arange(31)), 'g-')
    plt.plot(np.arange(31), poisson(mean_n).pmf(np.arange(31)), 'g-o')
    plt.xlabel('Number of CO per cell')
    plt.ylabel('Relative frequency')
    plt.savefig(f'{prefix}-cell-number.svg')
    plt.close()

    cell_no_univalents = [ sum(np.array(p)==0) for p in zip(*n_co_chr) ]
    print('no uv', np.bincount(cell_no_univalents))

    plt.figure()
    d = np.bincount(cell_no_univalents, minlength=6)[:6]/len(cell_no_univalents)
    plt.bar(np.arange(6), d)
    plt.xlabel('Number of univalents per cell')
    plt.xticks(range(6))
    plt.ylabel('Relative frequency')
    plt.savefig(f'{prefix}-cell-no-univalents.svg')
    plt.close()

    eps = 0.1
    cell_co_1_4 = np.array([sum(np.array(p)[:-1]) for p in zip(*n_co_chr) ])
    cell_co_5 = np.array(n_co_chr[4])
    plt.figure()
    plt.scatter(cell_co_5+eps*npr.randn(len(cell_co_5)), cell_co_1_4+eps*npr.randn(len(cell_co_5)), alpha=0.5, s=0.2)
    plt.xlabel('No. of COs on Chr5')
    plt.ylabel('No. of COs on Chrs1-4')

    xx, yy, r2 = lin_fit(cell_co_5, cell_co_1_4, r2=True)
    plt.plot(xx, yy, 'r-')
    plt.text(0.6, 0.8, f'$R^2$ = {r2:.2f}', transform=plt.gca().transAxes)

    plt.savefig(f'{prefix}-covariance.svg')


    plt.figure()
    plt.hist(spacing, bins=np.linspace(0, 1, 11), density=True)
    plt.xlabel('Relative focus spacing along CP')
    plt.ylabel('Frequency density')
    plt.xlim(0, 1)
    plt.savefig(f'{prefix}-spacing.svg')

    plt.figure()
    plt.violinplot([v for v in foci_intensities.values() if len(v)>0], [i for i,v in foci_intensities.items() if len(v)>0], showmeans=True)

    data_n = [i  for i,v in foci_intensities.items() for j in v] 
    data_v = [u  for v in foci_intensities.values() for u in v]

    print('LIN FIT'+prefix)
    xx, yy = lin_fit(data_n, data_v)
    plt.plot(xx, yy, 'r-')

    plt.xticks(np.arange(1, np.max([i for i,v in foci_intensities.items() if len(v)>0]) +1 ) )
    plt.xlabel('Number of foci per CP')
    plt.ylabel('Focus relative intensity')

    plt.savefig(f'{prefix}-violin_intensities.svg')
    plt.savefig(f'{prefix}-violin_intensities.png')

    
for prefix in ['poisson-0.0']:
    
    main('out-'+prefix+'.dat', prefix)

