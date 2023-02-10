
import numpy as np

from imageio import volread

def skel_to_points(skel):
    coords_list = list(zip(*np.where(skel)))
    return coords_list

from scipy.spatial import cKDTree

def extract_path(coords_list, p_start=None):
    ordered_points = []
    if p_start is None:
        p_start = coords_list[0]
    




def find_an_end(coords_list, p_start=None):
    available_points = set(coords_list)
    if p_start is None:
        p_start = coords_list[0]
    p = p_start

    def closest_point(p):
        v = np.array(list(available_points)) - p
        sq_dists = np.sum(v * v, axis=1)
        closest_index = np.argmin(sq_dists)
        sq_d = sq_dists[closest_index]
        return list(available_points)[closest_index], sq_d

    available_points.remove(p)

    ordered_points = []
    while len(available_points):
        ordered_points.append(p)
        p_next, sq_d = closest_point(p)
        # TODO - concealing messy stuff here
        if(sq_d) > 10:
            return p
        p = p_next

        available_points.remove(p)


def closest_point(available_points, p):
    v = np.array(list(available_points)) - p
    sq_dists = np.sum(v * v, axis=1)
    closest_index = np.argmin(sq_dists)
    sq_d = sq_dists[closest_index]
    return list(available_points)[closest_index], sq_d


def order_points(coords_list, p_start=None):
    available_points = set(coords_list)
    if p_start is None:
        p_start = coords_list[0]
    p = p_start

    available_points.remove(p)

    ordered_points = []
    while len(available_points):
        ordered_points.append(p)
        p_next, sq_d = closest_point(available_points, p)
        if sq_d > 12:
            return ordered_points
        p = p_next
        available_points.remove(p)
    ordered_points.append(p)

    return ordered_points


def make_full_mask_s(ordered_points, measure_stack):
    R = 5
    x, y, z = np.ogrid[-R:R, -R:R, -R:R]
    sphere = x**2 + y**2 + ((2.3 * z)**2) < R**2

    s = measure_stack.shape


    mask = np.zeros(measure_stack.shape, dtype=np.bool)

    for p in ordered_points:
        r, c, z = p
        o_1, o_2, o_3 = max(R-r, 0), max(R-c, 0), max(R-z,0)
        e_1, e_2, e_3 = min(R-r+s[0], 2*R), min(R-c+s[1], 2*R), min(R-z+s[2], 2*R)
        mask[r-R:r+R,c-R:c+R,z-R:z+R] = mask[r-R:r+R,c-R:c+R,z-R:z+R] | sphere[o_1:e_1, o_2:e_2, o_3:e_3]

    return mask

def measure_from_mask(mask, measure_stack):
    return np.sum(mask * measure_stack)

def max_from_mask(mask, measure_stack):
    return np.max(mask * measure_stack)

def measure_at_point(p, melem, measure_stack, op='mean'):
    if op=='mean':
        mask = make_mask_s(p, melem, measure_stack)
        melem_size = np.sum(melem)
        return float(measure_from_mask(mask, measure_stack) / melem_size)
    else:
        mask = make_mask_s(p, melem, measure_stack)
        melem_size = np.sum(melem)
        return float(max_from_mask(mask, measure_stack))



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

def measure_at_point(p, melem, measure_stack, op='mean'):
    if op=='mean':
        mask, m_data = make_mask_s(p, melem, measure_stack)
        melem_size = np.sum(melem)
        return float(measure_from_mask(mask, m_data) / melem_size)
    else:
        mask, m_data = make_mask_s(p, melem, measure_stack)
#        melem_size = np.sum(melem)
        return float(max_from_mask(mask, m_data))



def make_sphere(R=5):
    x, y, z = np.ogrid[-R:R, -R:R, -R:R]
    sphere = x**2 + y**2 + (2.3 * z)**2 < R**2
    return sphere

def measure_all_with_sphere(points_list, measure_stack, op='mean'):
    melem = make_sphere()
    measure_func = lambda p: measure_at_point(p, melem, measure_stack, op)
    return list(map(measure_func, points_list))



def measure_chrom2(path, hei10):
    coords_list = skel_to_points(path)
#    print(coords_list)
    p_end = find_an_end(coords_list)
#    print('found_end', p_end)
    ordered_points = order_points(coords_list, p_end)
    measurements = measure_all_with_sphere(ordered_points, hei10, op='mean')
    return coords_list, ordered_points, measurements

