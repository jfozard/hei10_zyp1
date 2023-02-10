
import sys
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.stats import poisson
import numpy.random as npr

from collections import defaultdict

from pathlib import Path

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
    'svg.fonttype': 'none',

})

def rel_freq_hist(ax, m, bins=np.linspace(0,1,11)):
    if (m<bins[0]).any() or (m>bins[-1]).any():
        print('DOMAIN', np.max(m), bins[-1], np.sum(m>bins[-1]))
    
    h, edges = np.histogram(m, bins)
    h = h/np.sum(h)
    ax.bar(edges[:-1], h, edges[1:]-edges[:-1], align='edge')


import statsmodels.api as sm

def lin_fit(x, y, min_x=None, max_x=None,of=None, r2=False):
    X = np.array(x)
    Y = np.array(y)
    X2 = sm.add_constant(X)
    est = sm.OLS(y, X2)
    est2 = est.fit()
    if of is None:
        print(est2.summary())
    else:
        print(est2.summary(), file=of)
    lx = np.min(X) if min_x is None else min_x
    ux = np.max(X) if max_x is None else max_x
    X = np.linspace(lx, ux, 100)
    X2 = sm.add_constant(X)
    yy = est2.predict(X2)
    if not r2:
        return X, yy
    else:
        return X, yy, est2.rsquared

simulation_output_path = '../output/simulation_output/'
image_output_path = '../output/simulation_figures/'

Path(image_output_path).mkdir(exist_ok=True, parents=True)

R = 0.6
T = 1.0

def pc(u, H):
    return np.where(u>H)[0]


class CellData(object):
    def __init__(self):
        self.paths = {}

class PathData(object):
    def __init__(self):
        pass

def read_file(fn):
    all_data = []
    head = None
    try:
        with open(fn,'r') as f:
            i = 0
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
                    for q in range(len(L_array)):
                        l = next(f)
                        if l.strip():
                            x_arrays.append(list(map(float, l.strip().split(','))))
                            u_arrays.append(list(map(float, next(f).strip().split(','))))
                        else:
                            x_arrays.append([])
                            next(f)
                            u_arrays.append([])
                            
                    all_data.append((L_array, x_arrays, u_arrays))
            except StopIteration:
                pass
    except IOError:
        pass


    head_dict = dict([u.split('=') for u in head.split(',')])
    del head_dict['start']
    return head_dict, all_data

    
def process_file(fn):

    head_dist, all_data = read_file(fn)

    cd_all = []
    pd_all = []
    
    for L_cell, x_cell, u_cell in all_data:

        cd = CellData()
        
        Q = len(L_cell)
        u_cell_int = []
        u_cell_max = 0

        # First find the focus intensity threshold for this cell
        
        for i in range(Q):
            u = u_cell[i]
            L = L_cell[i]
            if len(u):
                u_cell_int += list(u) 
                u_cell_max = max(np.max(u), u_cell_max)
        u_cell_int = np.array(u_cell_int)
        T = R*np.mean(u_cell_int[u_cell_int>1])

        # Now find the sum of the focus intensities (and the sum of all RI intensities)

        u_foci_tot = 0.0
        u_cell_tot = 0.0
        n_foci = 0
        
        for i in range(Q):
            u = np.array(u_cell[i])
            L = L_cell[i]
            if len(u):
                foci = pc(u, T)
                u_foci_tot += np.sum(u[foci])
                u_cell_tot += np.sum(u)
                n_foci += len(foci)

        cd.paths = []
        for i in range(Q):

            u = np.array(u_cell[i])
            L = L_cell[i]
            x = np.array(x_cell[i])

            pd = PathData()
            pd.SC_length = L

            foci = pc(u, T)

            if len(foci)==len(u):
                print('All foci', i, u)
            
            pd.foci_pos = x[foci] / L
            pd.abs_intensies = u[foci]
            pd.foci_intensities = u[foci] / u_foci_tot
            pd.norm_intensities = (u[foci] / u_foci_tot) * n_foci
            cd.paths.append(pd)
            pd_all.append(pd)
        cd_all.append(cd)

    return {'path_data':pd_all, 'cell_data': cd_all }            

def process(input_file, prefix, max_n=30, max_n_sc=10, max_dist=60, of=sys.stdout):

    
    data = process_file(input_file)

    pd_all = data['path_data']
    cd_all = data['cell_data']

    # Find range of Co number per SC/ per Cell

    pos = [] # Relative positions of all foci

    for pd in pd_all:
        pos += list(pd.foci_pos)
        

    # Now make histograms
    n_co_chr = [ [] for q in range(5) ]
    co_rel_pos_chr = [ [] for q in range(5) ]
    co_int_chr = [ [] for q in range(5) ]

    for cd in cd_all:
        for q in range(5):
            pd = cd.paths[q]
            n_co_chr[q].append(len(pd.foci_pos))
            co_rel_pos_chr[q].append(pd.foci_pos)
            co_int_chr[q].append(pd.foci_intensities)

    with open(f'{prefix}-chr-co-counts.csv', 'w') as f:
        f.write(','.join(f'path_{j+1}' for j in range(5))+'\n')
        for i in range(len(n_co_chr[0])):
            f.write(','.join(str(n_co_chr[j][i]) for j in range(5))+'\n')
            
            
    for q in range(5):

        plt.figure()

        sum_n = len(n_co_chr[q])
        
        d = np.bincount(n_co_chr[q], minlength=max_n_sc+1)[:max_n_sc+1]
        plt.bar(np.arange(max_n_sc+1), d/sum_n)

        if np.max(n_co_chr)>max_n_sc:
            print('RANGE', q, prefix, np.max(n_co_chr), max_n_sc, np.sum(np.array(n_co_chr)>max_n_sc))

        
        plt.xticks(np.arange(max_n_sc+1))
        mean_n = np.mean(n_co_chr[q])
        var_n = np.var(n_co_chr[q], ddof=1)
        plt.text(0.7, 0.9, f'$\mu={mean_n:.2f}$', transform=plt.gca().transAxes)
        plt.text(0.7, 0.7, f'$S^2={var_n:.2f}$', transform=plt.gca().transAxes)

        plt.plot(np.arange(max_n_sc+1), poisson(mean_n).pmf(np.arange(max_n_sc+1)), 'g-o')
        plt.xlabel('COs per chromosome pair')
        plt.ylabel('Relative frequency')
        plt.savefig(f'{prefix}-ch-{q}-number.svg')
        plt.close()

        fig, ax = plt.subplots()
        pos = np.concatenate(co_rel_pos_chr[q])
        print(pos)
        
        rel_freq_hist(ax, pos, bins=np.linspace(0,1,10))
        plt.xlabel('Relative position along chromosome pair')
        plt.ylabel('Count')
        plt.xlim(0,1)
        plt.savefig(f'{prefix}-ch-{q}-relpos.svg')
        plt.close()


        
    pos = [] # Relative positions of all foci

    for pd in pd_all:
        pos += list(pd.foci_pos)

    fig, ax = plt.subplots()
    rel_freq_hist(ax, pos, bins=np.linspace(0,1,11))
    plt.xlabel('Relative CO position along\nchromosome pair', fontsize=22)
    plt.ylabel('Relative frequency')
    plt.xlim(0,1)
    plt.savefig(f'{prefix}-relpos.svg')
    plt.close()

    print(prefix, ' N CO_pos ', len(pos))

    
    cell_tot_co = []
    for cd in cd_all:
        n_foci = 0
        for pd in cd.paths:
            n_foci += len(pd.foci_pos)
        cell_tot_co.append(n_foci)

    plt.figure()
    d = np.bincount(cell_tot_co, minlength=max_n+1)[:max_n+1]
    plt.bar(np.arange(max_n+1), d/np.sum(d))
    plt.xticks(np.arange(0, max_n+1, 10))
    plt.xlim(0, max_n)
    if np.max(cell_tot_co)>max_n:
        print('DOMAIN', cell_tot_co, max_n)
        
    mean_n = np.mean(cell_tot_co)
    var_n = np.var(cell_tot_co, ddof=1)
    plt.text(0.7, 0.9, f'$\mu={mean_n:.2f}$', transform=plt.gca().transAxes)
    plt.text(0.7, 0.7, f'$S^2={var_n:.2f}$', transform=plt.gca().transAxes)
    plt.plot(np.arange(max_n+1), poisson(mean_n).pmf(np.arange(max_n+1)), 'g-o')

    print('sum check', np.sum(d/np.sum(d)), np.sum(poisson(mean_n).pmf(np.arange(max_n+1))))
    
    plt.xlabel('Number of COs per cell')
    plt.ylabel('Relative frequency')
    plt.savefig(f'{prefix}-cell-number.svg')
    plt.close()

    with open(f'{prefix}-cell-co-counts.csv', 'w') as f:
        f.write('count\n')
        for i in cell_tot_co:
            f.write(str(i)+'\n')
        

    
    cell_no_univalents = []
    for cd in cd_all:
        n = 0
        for pd in cd.paths:
            if len(pd.foci_pos)==0:
                n+=1
        cell_no_univalents.append(n)

    print(prefix+' no uv', np.bincount(cell_no_univalents), file=of)

    
    plt.figure()
    d = np.bincount(cell_no_univalents, minlength=6)[:6]/len(cell_no_univalents)
    plt.bar(np.arange(6), d)
    plt.xlabel('Number of univalents per cell')
    plt.xticks(range(6))
    plt.ylabel('Relative frequency')
    plt.savefig(f'{prefix}-cell-no-univalents.svg')
    plt.close()



    average_classII = 2

    npr.seed(112233)
    
    cell_no_univalents_with_classII = []
    for cd in cd_all:
        n = 0
        cell_total_SC = sum(pd.SC_length for pd in cd.paths)
        
        for pd in cd.paths:
            nf = len(pd.foci_pos)
            extra_n = poisson.rvs((pd.SC_length/cell_total_SC)*average_classII)
            if (nf + extra_n) ==0:
                n+=1
        cell_no_univalents_with_classII.append(n)

    print(prefix+f' no uv with {average_classII} classII length bias', np.bincount(cell_no_univalents_with_classII), file=of)


    plt.figure()
    d = np.bincount(cell_no_univalents_with_classII, minlength=6)[:6]/len(cell_no_univalents_with_classII)
    plt.bar(np.arange(6), d)
    plt.xlabel('Number of univalents per cell')
    plt.xticks(range(6))
    plt.ylabel('Relative frequency')
    plt.savefig(f'{prefix}-cell-no-univalents-with-class-II-length-bias.svg')
    plt.close()

    

    # Number of univalents accounting for 2 (2.2?) additional Class II COs per cell (poisson total number)

    average_classII = 2

    npr.seed(112233)
    
    cell_no_univalents_with_classII = []
    for cd in cd_all:
        n = 0
        cell_total_SC = sum(pd.SC_length for pd in cd.paths)
        
        for pd in cd.paths:
            nf = len(pd.foci_pos)
            extra_n = poisson.rvs((pd.SC_length/cell_total_SC)*average_classII)
            if (nf + extra_n) ==0:
                n+=1
        cell_no_univalents_with_classII.append(n)

    print(prefix+f' no uv with {average_classII} classII length bias', np.bincount(cell_no_univalents_with_classII), file=of)


    plt.figure()
    d = np.bincount(cell_no_univalents_with_classII, minlength=6)[:6]/len(cell_no_univalents_with_classII)
    plt.bar(np.arange(6), d)
    plt.xlabel('Number of univalents per cell')
    plt.xticks(range(6))
    plt.ylabel('Relative frequency')
    plt.savefig(f'{prefix}-cell-no-univalents-with-class-II-length-bias.svg')
    plt.close()

    npr.seed(112233)
    
    cell_no_univalents_with_classII = []
    for cd in cd_all:
        n = 0
        for pd in cd.paths:
            nf = len(pd.foci_pos)
            extra_n = poisson.rvs(0.2*average_classII)
            if (nf + extra_n) ==0:
                n+=1
        cell_no_univalents_with_classII.append(n)

    print(prefix+f' no uv with {average_classII} classII no length bias', np.bincount(cell_no_univalents_with_classII), file=of)

    plt.figure()
    d = np.bincount(cell_no_univalents_with_classII, minlength=6)[:6]/len(cell_no_univalents_with_classII)
    plt.bar(np.arange(6), d)
    plt.xlabel('Number of univalents per cell')
    plt.xticks(range(6))
    plt.ylabel('Relative frequency')
    plt.savefig(f'{prefix}-cell-no-univalents-with-class-II-no-length-bias.svg')
    plt.close()


    cell_no_univalents_with_classII = []
    for cd in cd_all:
        n = 0
        extra_co = npr.randint(0, 5, 2)
        for i, pd in enumerate(cd.paths):
            nf = len(pd.foci_pos)
            extra_n = np.sum(extra_co==i)
            if (nf + extra_n) ==0:
                n+=1
        cell_no_univalents_with_classII.append(n)

    print(prefix+f' no uv with fixed 2 classII no length bias', np.bincount(cell_no_univalents_with_classII), file=of)
    
    plt.figure()
    d = np.bincount(cell_no_univalents_with_classII, minlength=6)[:6]/len(cell_no_univalents_with_classII)
    plt.bar(np.arange(6), d)
    plt.xlabel('Number of univalents per cell')
    plt.xticks(range(6))
    plt.ylabel('Relative frequency')
    plt.savefig(f'{prefix}-cell-no-univalents-with-fixed-2-class-II-no-length-bias.svg')
    plt.close()


    

    spacing = []
    for pd in pd_all:
        fp = pd.foci_pos
        if len(fp)>1:
            spacing += list(np.diff(fp))


#    plt.figure()
    fig, ax = plt.subplots()
    rel_freq_hist(ax, spacing, bins=np.linspace(0, 1, 11))
    plt.xlabel('Relative CO spacing along\nchromosome pair', fontsize=22)
    plt.ylabel('Relative frequency')
    plt.xlim(0, 1)
    plt.savefig(f'{prefix}-spacing.svg')

    print(prefix, ' N focus_spacing ', len(spacing))


    spacing_abs = []
    for pd in pd_all:
        fp = pd.foci_pos
        L = pd.SC_length
        if len(fp)>1:
            spacing_abs += list(np.diff(fp)*L)

    fig, ax = plt.subplots()
    rel_freq_hist(ax, spacing_abs, bins=np.linspace(0, max_dist, 11))
    plt.xlabel('CO spacing along chromosome pair ($\mu$m)', fontsize=22)
    plt.ylabel('Relative frequency')
    plt.xlim(0, max_dist)
    plt.savefig(f'{prefix}-spacing-abs.svg')

    # What is this a plot of?

    foci_intensities = defaultdict(list)
    for pd in pd_all:
        nf = len(pd.foci_pos)
        if nf:
            foci_intensities[nf] += list(pd.norm_intensities)
    
    plt.figure()
    plt.violinplot([v for v in foci_intensities.values() if len(v)>0], [i for i,v in foci_intensities.items() if len(v)>0], showmeans=True)


    data_n = [i  for i,v in foci_intensities.items() for j in v] 
    data_v = [u  for v in foci_intensities.values() for u in v]


    print('LIN FIT intensities'+prefix, file=of)
    xx, yy = lin_fit(data_n, data_v, of=of)
    plt.plot(xx, yy, 'r-')


    plt.xticks(np.arange(1, np.max([i for i,v in foci_intensities.items() if len(v)>0]) +1 ) )
    plt.xlabel('Number of COs per chromosome pair', fontsize=24)
    plt.ylabel('CO relative intensity')

    plt.savefig(f'{prefix}-violin_intensities.svg')
    plt.savefig(f'{prefix}-violin_intensities.png')

simulation_output_path = '../output/simulation_output/'
simulation_figures_path = '../output/simulation_figures/'

def main():
    with open(simulation_figures_path+'zyp1_analysis.txt', 'w') as f:

        for s in ["regtot-nuc-"]:
          for prefix in ['poisson-0.0', 'poisson-0.2', 'no-ends-poisson-0.0']:
            prefix = s+prefix
            print(prefix, file=f)
            process(simulation_output_path+prefix+'.dat', simulation_figures_path+prefix, of=f, max_n_sc=10)

        for prefix in ['ox-poisson-0.0', 'ox-poisson-0.2']:
            prefix = s + prefix
            print(prefix, file=f)
            process(simulation_output_path+prefix+'.dat', simulation_figures_path+prefix, max_n=80, max_n_sc=40, of=f)


if __name__=="__main__":
    main()
