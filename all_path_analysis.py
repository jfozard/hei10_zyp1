
import numpy as np
import scipy.linalg as la

from scipy.spatial import cKDTree
from measure_all import measure_chrom2 as measure

from tifffile import imread

from scipy.stats import poisson


import matplotlib.pyplot as plt

import os

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

# Generate a 5 voxel radius sphere.
def make_sphere(R=5):
    x, y, z = np.ogrid[-R:R, -R:R, -R:R]
    sphere = x**2 + y**2 + (0.0909/0.0397 * z)**2 < R**2
    return sphere

# Fix mask near edges of stack
def make_mask_s(p, melem, measure_stack):
    mask = melem
    
    R = melem.shape[0] // 2
    r, c, z = p

    m_data = np.zeros(melem.shape)
    s = measure_stack.shape
    o_1, o_2, o_3 = max(R-r, 0), max(R-c, 0), max(R-z,0)
    e_1, e_2, e_3 = min(R-r+s[0], 2*R), min(R-c+s[1], 2*R), min(R-z+s[2], 2*R)
    m_data[o_1:e_1,o_2:e_2,o_3:e_3] = measure_stack[max(r-R,0):min(r+R,s[0]),max(c-R,0):min(c+R,s[1]),max(z-R,0):min(z+R, s[2])]
    return mask, m_data

def measure_from_mask(mask, measure_stack):
    return np.sum(mask * measure_stack)

def max_from_mask(mask, measure_stack):
    return np.max(mask * measure_stack)

# Find mean / max intensity of stack in a sphere around a point
def measure_at_point(p, measure_stack, op='mean'):
    melem = make_sphere()
    if op=='mean':
        mask, m_data = make_mask_s(p, melem, measure_stack)
        melem_size = np.sum(melem)
        return float(measure_from_mask(mask, m_data) / melem_size)
    else:
        mask, m_data = make_mask_s(p, melem, measure_stack)
        return float(max_from_mask(mask, m_data))

PAIRS = True

import pandas as pd

# Change these paths to point to downloaded data
A = pd.read_csv('/media/foz/moredata2/ZYP1/ZYP1-1/Focus_coordinates.csv')
B = pd.read_csv('/media/foz/moredata2/ZYP1/ZYP1-1/Copy of ZYP1_SNT_Foci_Measurements_18.2.22.csv')

base_path = '/media/foz/moredata2/ZYP1/ZYP1-1/'

img_scale = np.array([0.0909, 0.0397, 0.0397]) # Voxel size for this datset
dist_threshold = 0.3 # Threshold for comparing mean of minimum distances
dist_threshold_group = 0.5


pos_all = []
pos_n = { i:[] for i in range(10) }

foci_intensities = { i:[] for i in range(10) }
pos_all_nf = []
spacing = []

for k in B.iloc[::10].index: # Loop over all cells
    s = B.loc[k]
    path = base_path + s.Plant + ' image '+s.Image[5:].strip()+'/'
    A_plant = A.loc[ (A['Plant']==s.Plant) & (A['Image']==int(s.Image[5:])) ]   # Extract foci for this plant

    pts_fp_raw = np.array(A_plant[['CoreZ', 'CoreY', 'CoreX']]) # Focipicker foci coordinates in voxels

    stack = imread(path + 'HEI10.tif') # Read HEI10 stack

    paths = {i: imread(path + f'Path{i}.tif') for i in range(1,11 if PAIRS else 6) }   # Read path stacks (from snt)

    traced_paths = {}
    path_starts = {}
    mean_hei10 = {}

    def dist(x,y): # Euclidean distance function
        return la.norm(np.array(x)-np.array(y))

    for i in range(1,11,2):

        # Use Chris' data to find correct end of paths - this ordering is unimportant for the final results

        b = B.iloc[k+i-1]
        c = B.iloc[k+i]

        j = i+1
        _, p, m = measure(paths[i], stack)

        if not pd.isna(b['Focus 1 ID']):
            idx = int(b['Focus 1 ID']) - 1
            first_chris_focus = pts_fp_raw[idx]
            if dist(p[0]*img_scale, first_chris_focus*img_scale) > dist(p[-1]*img_scale, first_chris_focus*img_scale):
                p = p[::-1]
                m = m[::-1]
        elif not pd.isna(c['Focus 1 ID']):
            idx = int(c['Focus 1 ID']) - 1
            first_chris_focus = pts_fp_raw[idx]
            if dist(p[0]*img_scale, first_chris_focus*img_scale) > dist(p[-1]*img_scale, first_chris_focus*img_scale):
                p = p[::-1]
                m = m[::-1]      

        p = p*img_scale[np.newaxis,:] # Convert from voxels to microns
        traced_paths[i] = p
        path_starts[i] = p[0]
        mean_hei10[i] = m

        _, p, m = measure(paths[j], stack) # Extract path for second of chromosome pair
        p = p*img_scale[np.newaxis,:]
        if dist(p[0], path_starts[i]) > dist(p[-1], path_starts[i]):
            p = p[::-1]
            m = m[::-1]
        traced_paths[j] = p
        path_starts[j] = p[0]
        mean_hei10[j] = m

    pts_fp = pts_fp_raw[:,:3]*img_scale[np.newaxis,:] # Focipicker focus coordinates in microns

    bg = np.median(stack)
    
    measured_intensities = []
    for i in range(len(pts_fp_raw)):
        measured_intensities.append(measure_at_point(pts_fp_raw[i].astype(int), stack)-bg) # Measure mean intensity in 5 voxel sphere,
                                                                                           # subtract stack median as estimate of bg

    measured_intensities = np.array(measured_intensities)

    # Place the foci onto the paths


    path_trees = { i: cKDTree(traced_paths[i]) for i in traced_paths }

    point_traces = {i:[] for i in range(1,11)}
    point_idx = {i:[] for i in range(1,11)}


    for j, p in enumerate(pts_fp):
            distances, pt_idx = {}, {}
            for i in range(1,11): # Find closest point on each trace
                distances[i], pt_idx[i] = path_trees[i].query(p[:3])
            group_distances = []
            for i in range(1,6): # For each CP find mean of two closests distances
                group_distances.append(0.5*sum((distances[2*i-1], distances[2*i])))

            group = np.argmin(group_distances)+1
            group_dist = group_distances[group-1]

            if group_dist < dist_threshold_group:  # Mean distance to CP sufficiently small
                if distances[2*group-1] <= distances[2*group]:
                    u = 2*group-1
                    o = 2*group
                else:
                    u = 2*group
                    o = 2*group-1

                pp = (pt_idx[u]/(len(traced_paths[u]-1)))
                ppo = (pt_idx[o]/(len(traced_paths[u])-1))
                point_traces[u].append(pp)
                point_traces[o].append(ppo)
                point_idx[u].append(j)
                point_idx[o].append(j)
            else:  
                u = np.argmin(list(distances.values()))+1

                min_dist = distances[u]

                print(' matching to path', j, u, min_dist, dist_threshold)
                print(distances)
                if min_dist < dist_threshold:   # Otherwise is closest CP sufficiently near?
                    o = u+1 if (u%2==1) else u-1
                    pp = (pt_idx[u]/(len(traced_paths[u]-1)))
                    ppo = (pt_idx[o]/(len(traced_paths[o])-1))
                    point_traces[u].append(pp)
                    point_traces[o].append(ppo)
                    point_idx[u].append(j)
                    point_idx[o].append(j)



    # Normalize foci intensities by sum of all assigned foci within the cell
    cell_tot_int = 0.0
    
    n_co = 0
    for i in range(1,11, 2): 
        cell_tot_int += sum(measured_intensities[j] for j in point_idx[i] )
        n_co += len(point_idx[i])

    cell_tot_int /= n_co

        
    for i in range(1,11, 2): 

        b = B.iloc[k+i-1]

        order = np.argsort(point_traces[i])
        pp = np.array(point_idx[i])[order]
        qq = np.array(point_traces[i])[order]
        
        
        if len(point_traces[i])>0:
            n = len(qq)
            pos_n[n] += list(qq)
            pos_n[n] += list(1-qq)
            pos_all += list(qq)
            pos_all += list(1-qq)

            pos_all_nf += list(np.minimum(qq, 1-qq))

            foci_intensities[n] += [ measured_intensities[j]/cell_tot_int for j in pp ]
            
            if n>1:
                spacing += list(np.diff(qq))


from scipy.stats import kstest

import statsmodels.api as sm


# Linear regression using python libraries
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


# Output file for testing
with open('pos_all.txt', 'w') as f:
    f.write(str(pos_all_nf))



plt.figure()
plt.hist(pos_all, bins=np.linspace(0,1,10), color='r', density=True)
plt.xlabel('Relative focus position along CP')
plt.ylabel('Frequency density')
plt.text(0.65, 0.9, f'N={len(pos_all)//2}', transform=plt.gca().transAxes)
plt.xlim(0, 1)
plt.savefig('pos_all.svg')


plt.figure()
plt.hist(spacing, bins=np.linspace(0, 1, 11), color='r', density=True)
plt.xlim(0, 1)
plt.text(0.7, 0.9, f'N={len(spacing)}', transform=plt.gca().transAxes)

plt.xlabel('Relative focus spacing along CP')
plt.ylabel('Frequency density')

plt.savefig('spacing.svg')


plt.figure()

plt.violinplot([v for v in foci_intensities.values() if len(v)>0], [i for i,v in foci_intensities.items() if len(v)>0], showmeans=True)
for i,v in foci_intensities.items():
    if len(v)>0:
        plt.scatter([i]*len(v), v)


data_n = [i  for i,v in foci_intensities.items() for j in v]
data_v = [u  for v in foci_intensities.values() for u in v]

with open('zyp1_intensity.dat', 'w') as f:
    f.write('num_co,intensity\n')
    for n,v in zip(data_n, data_v):
        f.write(f'{n},{v}\n')
        


xx, yy = lin_fit(data_n, data_v)
plt.plot(xx, yy, 'r-')
plt.xticks(np.arange(1, 10))

plt.xlabel('Number of foci per CP')
plt.ylabel('Focus relative intensity')

plt.savefig('intensity_violin.png')
plt.savefig('intensity_violin.svg')


    
