
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

#H = 10 #0.6*50

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
            pd.all_pos = x
            pd.all_int = u
            pd.foci_pos = x[foci] / L
            pd.abs_intensies = u[foci]
            pd.foci_intensities = u[foci] / u_foci_tot

            cd.paths.append(pd)
            pd_all.append(pd)
        cd_all.append(cd)

    return {'path_data':pd_all, 'cell_data': cd_all } 



def main(sim_file, prefix, max_n=3, max_n_cell=20, max_d = 60, downsample=False, obj_name = 'SC', obj_name_2 = 'SC', do_single=True, of=sys.stdout):



    data = process_file(simulation_output_path+sim_file)


    pd_all = data['path_data']
    cd_all = data['cell_data']

    L_all_CO = []
    m_all_CO = []
    L_all_no_CO = []
    m_all_no_CO = []


    for pd in pd_all:
        nf = len(pd.foci_pos)
        if nf:
            L = pd.SC_length
            m = np.max(pd.all_int)
            L_all_CO.append(L)
            m_all_CO.append(m)
        else:
            L = pd.SC_length
            if len(pd.all_int):
                m = np.max(pd.all_int)
            else:
                m = 0.0
            L_all_no_CO.append(L)
            m_all_no_CO.append(m)


    L_all_no_CO = np.array(L_all_no_CO)
    m_all_no_CO = np.array(m_all_no_CO)

    
    plt.figure()
    plt.scatter(L_all_CO, m_all_CO)
    plt.scatter(L_all_no_CO, m_all_no_CO)

    
    
    plt.figure()
    plt.scatter(L_all_no_CO, m_all_no_CO, alpha=0.5)
    plt.figure()
    plt.hist(L_all_no_CO)
    plt.figure()
    plt.title('No RI')
    plt.hist(L_all_no_CO[m_all_no_CO==0.0])
    plt.figure()
    plt.title('With RI')
    plt.hist(L_all_no_CO[(0.0<m_all_no_CO)])

    
    
    plt.figure()
    plt.bar([0, 1, 2], [len(L_all_no_CO[m_all_no_CO==0]), len(L_all_no_CO[m_all_no_CO>0]), len(L_all_CO)])

    
    len_data = [len(L_all_no_CO[m_all_no_CO==0]), len(L_all_no_CO[m_all_no_CO>0]), len(L_all_CO)]
    print(len_data)

    len_data_2 = len(L_all_no_CO[(0.0<m_all_no_CO) & (m_all_no_CO<1.0)])

    
    print(f'{100*(len_data[0]+len_data[1])/sum(len_data)}% with no CO.')
    print(f'Of these {100*len_data[0]/(len_data[0]+len_data[1])}% ({100*(len_data[0])/sum(len_data)}% of total) have no RI.')
    print(f'Of the remaining {100*(len_data[1])/sum(len_data)}% of total, {100*len_data_2/len_data[1]} ( {100*len_data_2/sum(len_data)} of total) have a very dim RI.')

    print(f'Maximum length of simulated univalent is {np.max(L_all_no_CO)}')

    L_all_CO = np.array(L_all_CO)
    L_all_no_CO = np.array(L_all_no_CO)
    
    x_all = []
    p_all = []
    for x in range(30):
        
        N1 = sum(L_all_no_CO>=x)
        N2 = sum(L_all_CO>=x)
        x_all.append(x)
        p_all.append(N2/(N1+N2))
    print(list(zip(x_all, p_all)))
    plt.figure()
    plt.plot(x_all, p_all)
    plt.axhline(0.95)
    plt.axhline(0.99)

    print(f'For segments longer than 10 um, proportion of non-univalents is {p_all[11]}')
    
    plt.show()

    
main('pch_0.2_0.15_0.1_regtot.dat', 'pch_', max_d=20, downsample=True, obj_name='Segment', obj_name_2='segment')

#main('pch_1.0_0.27_0.2_regtot.dat', 'pch_alpha_fixed_', max_d=20, downsample=True, obj_name='Segment', obj_name_2='segment')
#main('pch_0.2_0.1_0.1_regtot.dat', 'pch_', max_d=20, downsample=True, obj_name='Segment', obj_name_2='segment')
#main('pch_0.2_0.16_0.0_regtot.dat', 'pch_nonuc_', max_d=20, downsample=True, obj_name='Segment', obj_name_2='segment')



