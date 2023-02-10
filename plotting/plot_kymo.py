

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
    'svg.fonttype' : 'none',
})

def read_file(fn):

    sim_all = []
    
    f = open(fn, 'r')
    next(f)
    try:
        while True:
            h = next(f)
            L_cell = np.array(list(map(float, h.split(','))))
            h = next(f)
            t = np.array(list(map(float, h.split(','))))
            x_cell = []
            u_t_cell = []
            Q = len(L_cell)
            Nt = len(t)
            
            for i in range(Q):
                h = next(f)
                x = np.array(list(map(float, h.split(','))))
                x_cell.append(x)
                
                u_t = []
                for j in range(Nt):
                    h = next(f)
                    u = np.array(list(map(float, h.split(','))))
                    u_t.append(u)
                u_t_cell.append(np.array(u_t))

            h = next(f)
            p_t = np.array(list(map(float, h.split(','))))
            sim = { 'L_cell':L_cell, 'x_cell': x_cell, 't':t, 'u_t_cell':u_t_cell,  'p_t':p_t }
            sim_all.append(sim)
    except StopIteration:
        pass


    return sim_all

def kymo(t, L, x, u):
     
    n = 256
    m = 100

    canvas = np.zeros((n, m))


    n_dt, N = u.shape
    print(u.shape)

    T = t[-1]
    
    t_plot = (0.5+np.arange(n))/n*T

    # Interpolate simulation data onto a fixed grid
    for i in range(N):
        v = u[:, i]
        f = interp1d(t, v)
        vv = f(t_plot)
        idx = int((x[i]/L)*m)
        canvas[:, idx] = np.maximum(canvas[:,idx], vv) # Maximum intensity if two traces overlap


    canvas = np.maximum(canvas, 0.05)

    return canvas


def process(data_path, output_fn):

    data = read_file(data_path)[0]

    t_array = data['t']/60/60

    print(data['u_t_cell'])

    u_t_array = np.hstack(data['u_t_cell'])
    
    plt.figure()
    plt.plot(t_array, u_t_array)
    plt.xlabel('Time (h)')
    plt.ylabel('RI HEI10 amounts (a.u.)')
#    plt.xlim(0, 10)
    plt.ylim(0)
    plt.savefig(output_fn+'_kymo.svg')

    u = u_t_array[-1,:]
    threshold = np.mean(u[u>1])*0.6
    plt.axhline(threshold)
    plt.savefig(output_fn+'_kymo_threshold.svg')
    
    
simulation_output_path = '../output/simulation_output/'
simulation_figures_path = '../output/simulation_figures/'

s = "regtot-nuc-"
for prefix in ['poisson-0.0', 'poisson-0.2', 'ox-poisson-0.0', 'ox-poisson-0.2']:
    prefix = s + prefix
    print(prefix)
    process(simulation_output_path+prefix+'.kymo', simulation_figures_path+prefix)

