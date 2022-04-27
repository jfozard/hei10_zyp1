
import numpy as np
import scipy.linalg as la
from skimage.feature import peak_local_max, blob_dog
from skimage.filters import gaussian

from scipy.spatial import cKDTree
#
from measure_all import measure_chrom2 as measure

from tifffile import imread

from scipy.stats import poisson

import napari
from vispy.color import Colormap

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




def make_sphere(R=5):
    x, y, z = np.ogrid[-R:R, -R:R, -R:R]
    sphere = x**2 + y**2 + (0.0909/0.0397 * z)**2 < R**2
    return sphere

def make_mask_s(p, melem, measure_stack):
    mask = melem
    
    #mask = np.zeros(measure_stack.shape, dtype=np.uint8)
    R = melem.shape[0] // 2
    r, c, z = p
    #mask[r-R:r+R,c-R:c+R,z-R:z+R] = mask[r-R:r+R,c-R:c+R,z-R:z+R] | melem

    m_data = np.zeros(melem.shape)
    s = measure_stack.shape
    o_1, o_2, o_3 = max(R-r, 0), max(R-c, 0), max(R-z,0)
    e_1, e_2, e_3 = min(R-r+s[0], 2*R), min(R-c+s[1], 2*R), min(R-z+s[2], 2*R)
#    print(o_1,o_2,o_3,e_1,e_2,e_3)
    m_data[o_1:e_1,o_2:e_2,o_3:e_3] = measure_stack[max(r-R,0):min(r+R,s[0]),max(c-R,0):min(c+R,s[1]),max(z-R,0):min(z+R, s[2])]
    return mask, m_data

def measure_from_mask(mask, measure_stack):
    return np.sum(mask * measure_stack)

def max_from_mask(mask, measure_stack):
    return np.max(mask * measure_stack)


def measure_at_point(p, measure_stack, op='mean'):
    melem = make_sphere()
    if op=='mean':
        mask, m_data = make_mask_s(p, melem, measure_stack)
        melem_size = np.sum(melem)
        return float(measure_from_mask(mask, m_data) / melem_size)
    else:
        mask, m_data = make_mask_s(p, melem, measure_stack)
#        melem_size = np.sum(melem)
        return float(max_from_mask(mask, m_data))




PAIRS = True

import pandas as pd

A = pd.read_csv('/media/foz/moredata2/ZYP1/ZYP1-1/Focus_coordinates.csv')

B = pd.read_csv('/media/foz/moredata2/ZYP1/ZYP1-1/Copy of ZYP1_SNT_Foci_Measurements_18.2.22.csv')

print(B.iloc[0])

base_path = '/media/foz/moredata2/ZYP1/ZYP1-1/'

img_scale = np.array([0.0909, 0.0397, 0.0397])
dist_threshold = 0.3
dist_threshold_group = 0.5
unit_scale = 1


def no_last_true(x):
    if len(x)==0:
        return 0
    idx = len(x) - 1 - np.argmax(x[::-1])
    if x[idx]:
        return idx+1
    else:
        return 0

def get_n_foci(A):
    N_foci = []
    for i in range(0, len(A.index), 2):
        n1 = no_last_true(~pd.isna(A.iloc[i,16:16+9]))
        n2 = no_last_true(~pd.isna(A.iloc[i+1,16:16+9]))
        n = max(n1, n2)
        N_foci += [n,n]
    return N_foci

pos_all = []
pos_n = { i:[] for i in range(10) }
intensities = []
intensities_n = { i:[] for i in range(10) }
foci_intensities = { i:[] for i in range(10) }
catchment_all = []
catchment_intensities = []
pos_all_s = []
pos_all_nf = []
intensities_s = []
spacing = []
no_co_cell = []
no_co_cell_chris = []


for k in B.iloc[::10].index:
    s = B.loc[k]
    path = base_path + s.Plant + ' image '+s.Image[5:].strip()+'/'
    A_plant = A.loc[ (A['Plant']==s.Plant) & (A['Image']==int(s.Image[5:])) ]

    print('A_plant index', A_plant.index)
    print(A_plant.iloc[0])

    N_foci = get_n_foci(B.iloc[k:k+10])

    pts_fp_raw = np.array(A_plant[['CoreZ', 'CoreY', 'CoreX']])


    stack = imread(path + 'HEI10.tif')

    paths = {i: imread(path + f'Path{i}.tif') for i in range(1,11 if PAIRS else 6) }   

    traced_paths = {}
    path_starts = {}
    mean_hei10 = {}

    def dist(x,y):
        return la.norm(np.array(x)-np.array(y))


    for i in range(1,11,2):

        # Use Chris' data to find correct end of paths.

        b = B.iloc[k+i-1]
        c = B.iloc[k+i]


        j = i+1
        _, p, m = measure(paths[i], stack)

        if not pd.isna(b['Focus 1 ID']):
            idx = int(b['Focus 1 ID']) - 1
            first_chris_focus = pts_fp_raw[idx]
            print('first_chris_focus', i, first_chris_focus, p[0], p[-1])
            if dist(p[0]*img_scale, first_chris_focus*img_scale) > dist(p[-1]*img_scale, first_chris_focus*img_scale):
                p = p[::-1]
                m = m[::-1]
                print('flip')
        elif not pd.isna(c['Focus 1 ID']):
            idx = int(c['Focus 1 ID']) - 1
            first_chris_focus = pts_fp_raw[idx]
            print('other', i, first_chris_focus, p[0], p[-1])
            if dist(p[0]*img_scale, first_chris_focus*img_scale) > dist(p[-1]*img_scale, first_chris_focus*img_scale):
                p = p[::-1]
                m = m[::-1]
                print('flip')
        else:
            print('No first focus?', b['Plant'], b['Image'], b['Path'])        

        p = p*img_scale[np.newaxis,:]
        traced_paths[i] = p
        path_starts[i] = p[0]
        mean_hei10[i] = m
        _, p, m = measure(paths[j], stack)
        p = p*img_scale[np.newaxis,:]
        if dist(p[0], path_starts[i]) > dist(p[-1], path_starts[i]):
            p = p[::-1]
            m = m[::-1]
        traced_paths[j] = p
        path_starts[j] = p[0]
        mean_hei10[j] = m

    
#    pts_orig = blob_dog(stack, min_sigma=2, max_sigma=12, threshold=0.09)

#    pts_raw = np.array(pts_orig)

    pts = pts_fp_raw[:,:3]*img_scale[np.newaxis,:]

    pts_fp = pts_fp_raw[:,:3]*img_scale[np.newaxis,:]

    bg = np.median(stack)
    
    measured_intensities = []
    for i in range(len(pts_fp_raw)):
        measured_intensities.append(measure_at_point(pts_fp_raw[i].astype(int), stack)-bg)

    measured_intensities = np.array(measured_intensities)
#    measured_intensities /= np.sum(measured_intensities)

    print('measured_intensities', len(measured_intensities), measured_intensities)
    
    # Match the foci between fp and blob_dog
    # Ignore potential matches further apart than min_dist

    pts_fp_tree = cKDTree(pts_fp)

    match_dist=0.2

    map_blob_fp = {}
    for j, p in enumerate(pts):
        d, idx = pts_fp_tree.query(p)
        print(j, ' -> ', idx, d)
        if d<match_dist:
            map_blob_fp[j] = idx
    print('map_blob_fp', map_blob_fp)



    # Place the foci onto the paths

    nearest_path = []
    nearest_dist = []
    path_pos = []
    path_pos_other = []
    distance_other = []


    path_trees = { i: cKDTree(traced_paths[i]) for i in traced_paths }

    point_traces = {i:[] for i in range(1,11)}
    point_idx = {i:[] for i in range(1,11)}

    ASSIGN_OPT=2


    if ASSIGN_OPT==1:
        for j, p in enumerate(pts):
            distances, pt_idx = {}, {}
            for i in range(1,11):
                distances[i], pt_idx[i] = path_trees[i].query(p[:3])
            

            u = np.argmin(list(distances.values()))+1


            min_dist = distances[u]

            print(' matching to path', j, u, min_dist, dist_threshold*unit_scale)
            print(distances)
            if min_dist < dist_threshold*unit_scale:
                o = u+1 if (u%2==1) else u-1

                nearest_path.append(u)  
                nearest_dist.append(distances[u]/unit_scale)
                path_pos.append((pt_idx[u]/(len(traced_paths[u]-1))))
                path_pos_other.append((pt_idx[o]/(len(traced_paths[o])-1)))
                point_traces[u].append(path_pos[-1])
                point_traces[o].append(path_pos_other[-1])
                point_idx[u].append(j)
                point_idx[o].append(j)

                distance_other.append(distances[o]/unit_scale)

    elif ASSIGN_OPT==2 or ASSIGN_OPT==3: # Assign to CP accoriding to mean or max. Need to be pairs
        for j, p in enumerate(pts):
            distances, pt_idx = {}, {}
            for i in range(1,11):
                distances[i], pt_idx[i] = path_trees[i].query(p[:3])
            group_distances = []
            for i in range(1,6):
                if ASSIGN_OPT == 2:
                    group_distances.append(0.5*sum((distances[2*i-1], distances[2*i])))

                else: # ASSIGN_OPT==3
                    group_distances.append(max(distances[2*i-1], distances[2*i]))

            group = np.argmin(group_distances)+1
            group_dist = group_distances[group-1]
            print(j, 'group_dist', group_dist, dist_threshold*unit_scale)
            print(group_distances)
            if group_dist < dist_threshold_group*unit_scale:
                if distances[2*group-1] <= distances[2*group]:
                    u = 2*group-1
                    o = 2*group
                else:
                    u = 2*group
                    o = 2*group-1

                nearest_path.append(u)  
                nearest_dist.append(distances[u]/unit_scale)
                path_pos.append((pt_idx[u]/(len(traced_paths[u]-1))))
                path_pos_other.append((pt_idx[o]/(len(traced_paths[u])-1)))
                point_traces[u].append(path_pos[-1])
                point_traces[o].append(path_pos_other[-1])
                distance_other.append(distances[o]/unit_scale)
                point_idx[u].append(j)
                point_idx[o].append(j)
            else:
                u = np.argmin(list(distances.values()))+1


                min_dist = distances[u]

                print(' matching to path', j, u, min_dist, dist_threshold*unit_scale)
                print(distances)
                if min_dist < dist_threshold*unit_scale:
                    o = u+1 if (u%2==1) else u-1

                    nearest_path.append(u)  
                    nearest_dist.append(distances[u]/unit_scale)
                    path_pos.append((pt_idx[u]/(len(traced_paths[u]-1))))
                    path_pos_other.append((pt_idx[o]/(len(traced_paths[o])-1)))
                    point_traces[u].append(path_pos[-1])
                    point_traces[o].append(path_pos_other[-1])
                    point_idx[u].append(j)
                    point_idx[o].append(j)

                    distance_other.append(distances[o]/unit_scale)
                else:
                    nearest_path.append(0)  
                    nearest_dist.append(0)
                    path_pos.append(0)
                    path_pos_other.append(0)
                    distance_other.append(0)           


    cell_tot_int = 0.0

    
    n_co = 0
    for i in range(1,11, 2): 
        cell_tot_int += sum(measured_intensities[map_blob_fp[j]] for j in point_idx[i] )
        n_co += len(point_idx[i])

    cell_tot_int /= n_co

    no_co_cell.append(n_co)
    print('num co', n_co, cell_tot_int, np.sum(measured_intensities))
        
    print(path)
    for i in range(1,11, 2): 

        b = B.iloc[k+i-1]

        order = np.argsort(point_traces[i])
        pp = np.array(point_idx[i])[order]
        qq = np.array(point_traces[i])[order]

        inc = lambda x: x+1 if x is not None else None
        print(i, point_traces[i], point_idx[i], pp, [inc(map_blob_fp.get(j)) for j in pp])
        print(b[16:16+N_foci[i-1]])

        
        
        if len(point_traces[i])>0:
            n = len(qq)
            pos_n[n] += list(qq)
            pos_n[n] += list(1-qq)
            pos_all += list(qq)
            pos_all += list(1-qq)

            pos_all_nf += list(np.minimum(qq, 1-qq))

           
            intensities += [ measured_intensities[map_blob_fp[j]]/cell_tot_int for j in pp ]*2
            intensities_n[n] += [ measured_intensities[map_blob_fp[j]]/cell_tot_int for j in pp ]*2
            foci_intensities[n] += [ measured_intensities[map_blob_fp[j]]/cell_tot_int for j in pp ]

            print('int map', n, [ measured_intensities[map_blob_fp[j]]/cell_tot_int for j in pp ] )
            
            if n>1:
                spacing += list(np.diff(qq))
            if n==4:
                pos_all_s += list(np.minimum(qq, 1-qq))
                intensities_s += [ measured_intensities[map_blob_fp[j]]/cell_tot_int for j in pp ]
                m = 0.5*(qq[1:]+qq[:-1])
                catchment = np.diff([0] + list(m) + [1])
                catchment_all += list(catchment)
                tot_intensities = np.array([ measured_intensities[map_blob_fp[j]]/cell_tot_int for j in pp ])
                catchment_intensities += list(tot_intensities/np.sum(tot_intensities))


from scipy.stats import kstest

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


with open('pos_all.txt', 'w') as f:
    f.write(str(pos_all_nf))


plt.figure()
plt.hist(pos_n[3])
v = kstest(pos_n[3], pos_all)
plt.title(f' 3 {v}')
plt.savefig('pos3.png')
plt.figure()
plt.hist(pos_n[4])
v = kstest(pos_n[4], pos_all)
plt.title(f' 4 {v}')
plt.savefig('pos4.png')
plt.figure()
plt.hist(pos_all, bins=np.linspace(0,1,10), color='r', density=True)
plt.xlabel('Relative focus position along CP')
plt.ylabel('Frequency density')
plt.text(0.65, 0.9, f'N={len(pos_all)//2}', transform=plt.gca().transAxes)
plt.xlim(0, 1)
plt.savefig('pos_all.svg')
plt.savefig('pos_all.png')

plt.figure()
plt.hist(pos_all_nf, bins=np.linspace(0,0.5,6))
plt.savefig('pos_all_nf.png')

plt.figure()
plt.hist(spacing, bins=np.linspace(0, 1, 11), color='r', density=True)
plt.xlim(0, 1)
plt.text(0.7, 0.9, f'N={len(spacing)}', transform=plt.gca().transAxes)

plt.xlabel('Relative focus spacing along CP')
plt.ylabel('Frequency density')

plt.savefig('spacing.png')
plt.savefig('spacing.svg')

plt.figure()
X, yy = lin_fit(pos_all_s, intensities_s)

plt.scatter(pos_all_s, intensities_s)
plt.plot(X, yy, 'r-')
plt.savefig('pos_intensity_4.png')
plt.figure()
plt.scatter(pos_n[4], intensities_n[4])
plt.savefig('pos_intensity4.png')

plt.figure()
plt.hist(intensities, bins=np.linspace(0, 2, 10))
plt.savefig('intensities.png')

plt.figure()
plt.scatter(catchment_all, catchment_intensities)
X, yy = lin_fit(catchment_all, catchment_intensities)

plt.savefig('catchment_4.png')

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
plt.savefig(f'data-cell-number.svg')
plt.close()


#plt.show()
    
