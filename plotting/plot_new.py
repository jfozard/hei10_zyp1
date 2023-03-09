
import sys
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.stats import poisson
import numpy.random as npr

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


def rel_freq_hist(ax, m, bins=np.linspace(0,1,11), stacked=False):
    if (m<bins[0]).any() or (m>bins[-1]).any():
        print('DOMAIN', np.max(m), bins[-1], np.sum(m>bins[-1]))
    
    if stacked==False:
        h, edges = np.histogram(m, bins)
        h = h /np.sum(h)
        ax.bar(edges[:-1], h, edges[1:]-edges[:-1], align='edge')
    else:
        b = np.zeros_like(bins)[:-1]
        n_all = sum([len(d) for d in m])
        for d in m:
            h, edges = np.histogram(d, bins)
            h = h/n_all
            ax.bar(edges[:-1], h, edges[1:]-edges[:-1], align='edge', bottom = b)
            b += h

        
            
    

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
        
        for i in range(Q):
            u = np.array(u_cell[i])
            L = L_cell[i]
            if len(u):
                foci = pc(u, T)
                u_foci_tot += np.sum(u[foci])
                u_cell_tot += np.sum(u)

        cd.paths = []
        for i in range(Q):

            u = np.array(u_cell[i])
            L = L_cell[i]
            x = np.array(x_cell[i])

            pd = PathData()
            pd.SC_length = L

            foci = pc(u, T)
            pd.foci_pos = x[foci] / L
            pd.abs_intensies = u[foci]
            pd.foci_intensities = u[foci] / u_foci_tot

            cd.paths.append(pd)
            pd_all.append(pd)
        cd_all.append(cd)

    return {'path_data':pd_all, 'cell_data': cd_all } 



def main(sim_file, prefix, max_n=3, max_n_cell=20, max_d = 60, n_d=13, downsample=False, obj_name = 'SC', obj_name_2 = 'SC', do_single=True, of=sys.stdout):



    data = process_file(simulation_output_path+sim_file)

    

    pd_all = data['path_data']
    cd_all = data['cell_data']

    # Find range of Co number per SC/ per Cell
#    max_n_cell = 0
    max_n_sc = 0

    
    for pd in pd_all:
        max_n_sc = max(max_n_sc, len(pd.foci_pos))

    cell_tot_co = []
    for cd in cd_all:
        n_foci = 0
        for pd in cd.paths:
            n_foci += len(pd.foci_pos)
        cell_tot_co.append(n_foci)
        max_n_cell = max(max_n_cell, n_foci)

    pos = [] # Relative positions of all foci

    for pd in pd_all:
        pos += list(pd.foci_pos)
        
    resampled_d_pos = [] # Resampled (relative) difference in position, multiplied by n foci

    for pd in pd_all:
        nf = len(pd.foci_pos)
        r_pos = list(sorted(random.sample(pos, nf)))
        if nf>1:
            resampled_d_pos += list(np.diff(r_pos)*nf)

    
    n, c = np.unique(cell_tot_co, return_counts=True)

    if any(n>max_n_cell):
        raise Exception
    
    fig, ax = plt.subplots()
    ax.bar(n,c/np.sum(c))
    ax.set_xlim(0,max_n_cell)
    ax.set_xlabel('Cell CO number')
    plt.ylabel('Relative frequency')
    mean_n = np.mean(cell_tot_co)
    var_n = np.var(cell_tot_co, ddof=1)
    plt.text(0.05, 0.9, f'$\mu={mean_n:.2f}$', transform=plt.gca().transAxes)
    plt.text(0.05, 0.7, f'$S^2={var_n:.2f}$', transform=plt.gca().transAxes)

    print(prefix, 'cell CO number', n, c, 'N=', len(cell_tot_co), file=of)
    

    print('cell_tot_co', len(cell_tot_co))
        
    
    plt.plot(np.arange(max_n_cell+1), poisson(mean_n).pmf(np.arange(max_n_cell+1)), 'g-o')

    plt.savefig(image_output_path+prefix+'cell_co_num.svg')
    plt.close()


    # List of CO per segment / per SC
    n_co = [len(pd.foci_pos) for pd in pd_all ]
        
    fig, ax = plt.subplots()
    n, c = np.unique(n_co, return_counts=True)
    ax.bar(n,c/np.sum(c))
    ax.set_xlim(0-0.5,max_n_sc+0.5)
    ax.set_xticks(2*np.arange((max_n_sc+1)//2+1))
    ax.set_xlabel(f'{obj_name} CO number')
    ax.set_ylabel('Relative frequency')
    plt.ylabel('Relative frequency')
    mean_n = np.mean(n_co)
    var_n = np.var(n_co, ddof=1)
    plt.text(0.7, 0.9, f'$\mu={mean_n:.2f}$', transform=plt.gca().transAxes)
    plt.text(0.7, 0.7, f'$S^2={var_n:.2f}$', transform=plt.gca().transAxes)

    print(prefix, f'{obj_name} number', n, c, 'N=', len(n_co), file=of)

    if any(n>max_n_sc):
        raise Exception

    
    plt.savefig(image_output_path+prefix+'segment_co_num.svg')
    plt.close()

    print('segment CO count', prefix, n, c)

    # Spacing between COs (microns)

    d_pos = []
    for pd in pd_all:
        fp = pd.foci_pos
        if len(fp)>1:
            d_pos += list(np.diff(fp)*pd.SC_length)
    
            
    fig, ax = plt.subplots()
    rel_freq_hist(ax, d_pos, bins=np.linspace(0,max_d,n_d))
    ax.set_xlim(0, max_d)
    ax.set_xlabel('CO spacing ($\\mu$m)')
    ax.set_ylabel('Relative frequency')
    plt.savefig(image_output_path+prefix+'co_spacing.svg')
    plt.close()

    print(prefix, f'CO spacing', 'N=', len(d_pos), file=of)


    # Relative positions of COs

    fig, ax = plt.subplots()
    rel_freq_hist(ax, pos, bins=np.linspace(0,1,11))
    ax.set_xlim(0,1)
    ax.set_xlabel(f'Relative CO position along {obj_name_2}', fontsize=24)
    ax.set_ylabel('Relative frequency')
    plt.savefig(image_output_path+prefix+'rel_pos.svg')
    plt.close()

    print(prefix, f'{obj_name} CO number', 'N=', len(pos), file=of)

        
    for j in range(1,4):
        fig, ax = plt.subplots()
        m = [ [] for _ in range(j) ]
        for pd in pd_all:
            c = pd.foci_pos
            if len(c)==j:
                for k in range(j):
                    m[k].append(c[k])

        rel_freq_hist(ax, m, bins=np.linspace(0,1,11), stacked=True)
        ax.set_xlim(0,1)
        ax.set_xlabel(f'Relative CO position along {obj_name_2}', fontsize=24)
        ax.set_ylabel('Relative frequency') 
        plt.savefig(image_output_path+prefix+f'rel_pos_{j}.svg')
        plt.close()
        

    max_n = 3
#    data_stacked = [ [ p for p in int_by_n[d]] for d in range(1, max_n+1)]
    data_stacked = [ [] for d in range(1, max_n+1) ]
    for pd in pd_all:
        nf = len(pd.foci_pos)
        if nf>0:
            data_stacked[min(nf-1, max_n-1)] += list(pd.foci_intensities)
                
    fig, ax = plt.subplots()
    intensity_bins = 40
    ax.hist(data_stacked, bins=intensity_bins, stacked=True, label=['{}'.format(i) for i in range(1, max_n)] + [str(max_n)+'+'])
    ax.legend()
    ax.set_xlabel('Relative CO focus intensity')
    ax.set_ylabel('Number of COs')
    plt.savefig(image_output_path+prefix+'rel_int.svg')
    plt.close()


    max_L = 40

    violin_data = defaultdict(list)
    for pd in pd_all:
        nf = len(pd.foci_pos)
        violin_data[min(nf, max_n)].append(pd.SC_length)

    violin_pos = list(violin_data.keys())
    
    violin_labels = [ str(i) if i < max_n else f'{max_n}+' for i in violin_pos]

    fig, ax = plt.subplots()        
    ax.violinplot(list(violin_data.values()), positions=list(violin_pos), showmeans=True)
    ax.set_xticks(violin_pos)
    ax.set_xticklabels(violin_labels)
    ax.set_xlim(-0.5, max_n+0.5)
    ax.set_xlabel(f'{obj_name} CO number')
    ax.set_ylabel(f'{obj_name} length ($\\mu$m)')
    ax.set_ylim(0, max_L)
    plt.savefig(image_output_path+prefix+'violin_L_n.svg')
    plt.close()

    violin_data = defaultdict(list)
    for pd in pd_all:
        nf = len(pd.foci_pos)
        if nf>0:
            violin_data[min(nf, max_n)] += list(pd.foci_intensities)

    violin_pos = list(violin_data.keys())
    
    violin_labels = [ str(i) if i < max_n else f'{max_n}+' for i in violin_pos]
                

    fig, ax = plt.subplots()
    ax.violinplot(list(violin_data.values()), positions=list(violin_pos), showmeans=True)
    ax.set_xticks(violin_pos)
    ax.set_xticklabels(violin_labels)
    ax.set_xlim(-0.5, max_n+0.5)
    ax.set_xlabel(f'{obj_name} CO number')
    ax.set_ylabel(f'Focus intensity')
    plt.savefig(image_output_path+prefix+'violin_int_n.svg')
    plt.close()


    if do_single:
    
        single_L = []
        single_int = []
        for pd in pd_all:
            nf = len(pd.foci_pos)
            if nf==1:
                single_L.append(pd.SC_length)
                single_int.append(pd.foci_intensities[0])

        single_L = np.array(single_L)
        single_int = np.array(single_int)


        fig, ax = plt.subplots()        
        ax.scatter(single_L, single_int, s=1, alpha=0.5)

        print('Lin fit intensity vs length all', file=of)
        xx, yy, r2 = lin_fit(single_L, single_int, r2=True, min_x=0, max_x=30, of=of)
        ax.plot(xx, yy, 'r-')
        ax.set_xlim(0, 30)
        ax.set_ylim(0, 0.3)
        if (single_L>30).any() or (single_L<0).any() or (single_int>0.3).any() or (single_int<0).any():
            print('DOMAIN L ', np.max(single_int), np.max(single_L), np.sum(single_L>30), np.sum(single_int>0.3), len(single_L))
#            raise Exception
        
        ax.text(0.4, 0.9, f'$N={len(single_L)}$  $R^2={r2:.2f}$', transform=ax.transAxes)
        ax.set_ylabel('Relative intensity')
        ax.set_xlabel(f'{obj_name} length ($\\mu$m)')

        plt.savefig(image_output_path+f'single_int_vs_L.svg')
        plt.close()


        if downsample:
            downsample_n = 297
            fig, ax = plt.subplots()        

            x = np.array(single_L)
            y = np.array(single_int)
            npr.seed(1234)
            idx = npr.choice(np.arange(len(x)), downsample_n, replace=False)

            x = x[idx]
            y = y[idx]

            ax.scatter(x, y)#, s=1, alpha=0.5)

            print('Lin fit intensity vs length sample', file=of)

            xx, yy, r2 = lin_fit(x, y, r2=True, min_x=0, max_x=30, of=of)
            ax.plot(xx, yy, 'k-')
            ax.set_xlim(0, 30)
            ax.set_ylim(0, 0.3)
            ax.text(0.4, 0.9, f'$N={len(x)}$  $R^2={r2:.2f}$', transform=ax.transAxes)
            ax.set_ylabel('Relative intensity')
            ax.set_xlabel(f'{obj_name} length ($\\mu$m)')

            plt.savefig(image_output_path+prefix+'single_int_vs_L_downsample.svg')
            plt.close()

main('survey_wt_ox_0.0_1.0.dat', 'survey_wt_ox_0.0_1.0_', max_n_cell=35, do_single=False, n_d=14, max_d=65)
main('survey_interpolate_ends_ox_0.1_0.9.dat', 'interpolate_0.1_ox_', max_n_cell=35, do_single=False, n_d=14, max_d=65)
main('survey_wt_0.0_1.0.dat', 'survey_wt_0.0_1_', of=open(image_output_path+'wt_simulation_summary.txt', 'w'), do_single=False, n_d=14, max_d=65, max_n_cell=15)
main('survey_interpolate_ends_0.1_0.9.dat', 'interpolate_0.1_', of=open(image_output_path+'interpolate_simulation_summary.txt', 'w'), do_single=False, n_d=14, max_d=65, max_n_cell=15)
main('pch_0.2_0.15_0.1_regtot.dat', 'pch_', max_n_cell=20, max_d=24, n_d=13, downsample=True, obj_name='Segment', obj_name_2='segment', of=open(image_output_path+'pch_simulation_summary.txt', 'w'))

main('test_5_4.5_0.0.dat', 'test_5_4.5_0.0_', max_n_cell=35, do_single=False)
main('test_5_4.5_0.2.dat', 'test_5_4.5_0.2_', max_n_cell=35, do_single=False)
main('test_5_1_0.0.dat', 'test_5_1_0.0_', do_single=False)
main('test_5_1_0.2.dat', 'test_5_1_0.2_', do_single=False)



