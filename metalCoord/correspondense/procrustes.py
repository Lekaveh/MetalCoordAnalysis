import itertools
import numpy as np
from scipy.linalg import helmert
import tensorflow as tf
from sklearn.cluster import DBSCAN


def is_it_plane(xyz):
    R = np.zeros(shape=3); Rp = np.zeros(shape=3); Rpp = np.zeros(shape=3)
    xyz_l = np.zeros_like(xyz)
    l = len(xyz[:,0])
    if l <=3 :
        return(0.0,0.0)
    for i in range(3):
        R[i] = np.sum(xyz[0:l,i])
    R = R/l
    for i in range(3):
        xyz_l[0:l,i] = xyz[0:l,i]-R[i]
    inds = np.arange(l)
    for i in range(3):
        Rp[i] = np.sum(xyz_l[inds,i] * np.sin(2*np.pi*inds/l))
        Rpp[i] = np.sum(xyz_l[inds,i] * np.cos(2*np.pi*inds/l))
    cc = (2/l)**0.5
    Rp = cc*Rp; Rpp = cc*Rpp
    nnorm = np.cross(Rp,Rpp)
    nnorm = nnorm/np.linalg.norm(nnorm)
    z = np.zeros(shape=l)
    for i in inds:
        z[i] = np.dot(xyz_l[i,0:3],nnorm)
    return(np.sqrt(np.sum(z*z)/l),np.max(np.abs(z)))


def find_rings(coords):
    rings = []
    others = []
    centred_coords = coords[1:] - coords[0]
    normed_coords = centred_coords/np.max(np.linalg.norm(centred_coords, axis=1))
    clustering =  DBSCAN(eps=1, min_samples=4).fit(normed_coords)
    clusters = np.unique(clustering.labels_)

    for cluster in clusters:
        indices = (np.where(clustering.labels_ == cluster)[0] + 1).tolist()
        if cluster == -1:
            others.extend(indices)
            continue
       
        cluster_coords = normed_coords[clustering.labels_ == cluster]
        t1, t2 = is_it_plane(cluster_coords)
        if t1 <= 0.6 and t2 <= 0.6:
            rings.append(indices)
        else:
            others.append(indices)
    return sorted(rings, key = lambda x: len(x), reverse=True), others

def have_same_ring_length(rings1, rings2):
    if len(rings1) != len(rings2):
        return False
    
    for r1, r2 in zip(rings1, rings2):
        if len(r1) != len(r2):
            return False
    
    return True

core_vectors_num = 4
def norm(x, hm):
    return tf.sqrt(tf.linalg.trace(tf.transpose(x, perm=[0, 2, 1])@tf.transpose(hm, perm=[0, 2, 1])@hm@x))


def preshape(x):
    hm = helmert(x.shape[1])
    hm = tf.broadcast_to(tf.convert_to_tensor(
        hm, dtype='float32'), (1, hm.shape[0], hm.shape[1]))
    return hm@x/tf.reshape(norm(x, hm), (-1, 1, 1))


def distance(x1, x2):
    z2 = preshape(x2)
    z1 = preshape(x1)
    s, u, v = tf.linalg.svd(tf.transpose(
        z1, perm=[0, 2, 1])@z2@tf.transpose(z2, perm=[0, 2, 1])@z1)
    return tf.sqrt(tf.abs(1 - tf.reduce_sum(tf.sqrt(s), axis=1)**2))

def procrustes_fit(A, B):
    
    A_t = tf.transpose(A, perm=[0, 2, 1])
    s, v, w = tf.linalg.svd(A_t@B)
    wt = tf.transpose(w, perm=[0, 2, 1])
    R = v@wt
    c = tf.linalg.trace(tf.transpose(
        R, perm=[0, 2, 1])@A_t@B)/tf.linalg.trace(A_t@A)
    c = tf.reshape(c, (-1, 1, 1))
    approx = c*A@R
    return (distance(approx, B), approx, c, R)

def get_combinations(groups=None):

    c_list = []
    ranges = [0] + np.cumsum(groups).tolist()
    for  i in range(len(ranges) - 1):
        c_list.append(np.fromiter(itertools.chain.from_iterable(itertools.permutations(range(ranges[i], ranges[i + 1]))), dtype=np.int32).reshape(-1, ranges[i + 1] - ranges[i]))

    combinations = np.vstack([np.concatenate(x) for x in itertools.product(*c_list)])  
    k = len(combinations)
    return combinations,k


def create_group(rings, others):
    if len(others):
        return [[0]] + rings + [others]
    return [[0]] + rings
    



def fit_group(coords, ideal_coords, groups=None):
    n1 = coords.shape[0]
    n2 = ideal_coords.shape[1]
    
    if groups is None:
        correspondense = [list(range(n1)), list(range(n1))]
        lengths = [n1]
    else:
        correspondense = [np.hstack(groups[0]).tolist(), np.hstack(groups[1]).tolist()]
        lengths = [len(gr) for gr in  groups[0]]

    y = tf.broadcast_to(tf.convert_to_tensor(ideal_coords[correspondense[1]], dtype='float32'), (1, n1, n2))
    
    combinations, k = get_combinations(groups=lengths)

    t_combinations = combinations.reshape(-1, combinations.shape[0], combinations.shape[1], 1)

    s_y = tf.gather_nd(y, indices=t_combinations, batch_dims=1)
    s_y = tf.reshape(s_y, (k*1, n1, coords.shape[1]))
    s_x = tf.broadcast_to(tf.convert_to_tensor(coords[correspondense[0]], dtype='float32'), (1, n1, n2))   

    distances, approxs, c, R = procrustes_fit(s_y, s_x)
    distances = distances.numpy()
    min_distance = np.min(distances)
    mask = distances <= min_distance + 0.1
    
    distances = distances[mask]
    R = tf.boolean_mask(R, mask)
    c = tf.boolean_mask(c, mask)
    indices = tf.boolean_mask(combinations, mask).numpy()
    indices = indices.reshape(len(indices), n1)

    base_index = np.full((len(indices), n1), np.arange(n1))
    back_index = np.arange(n1)[np.argsort(correspondense[0])]

    rotated = tf.broadcast_to(tf.convert_to_tensor(ideal_coords, dtype='float32'), (1, n1, n2))@R
    approxs = (c*rotated).numpy()

    x_index = base_index[:, correspondense[1]]
    indices = np.vstack([x[i] for x, i in zip(x_index, indices)])[:,back_index]
    return distances, approxs, c.numpy() ,R.numpy(), indices,rotated.numpy()

def  find_in_group(groups, index):
    for i, group in enumerate(groups):
        if index in group:
            return i
    return -1
    
def fit(coords, ideal_coords, groups=None, all = False):


    rings1, others1 = find_rings(coords)
    rings2, others2 = find_rings(ideal_coords)

    if len(coords) > 9 and len(rings1) > 0 and have_same_ring_length(rings1, rings2):
        ring_lengths = [len(ring) for ring in rings2]
        current_l = -1
        ring_groups = []
        for i, l in enumerate(ring_lengths):
            if current_l == l:
                ring_groups[-1].append(i)
            else:
                ring_groups.append([i])
            current_l = l
        
        ring_group_permutations = [] 
        for ring_group in ring_groups:
            for perm_group in itertools.permutations(ring_group):
                ring_group_permutations.append(create_group([rings2[i] for i in perm_group], others2))
        ring1_group = create_group(rings1, others1)

        results = []
        for ring2_group in ring_group_permutations:
            current_groups = [ring1_group, ring2_group]
            results.append(fit_group(coords, ideal_coords, current_groups))
        
        distances, approxs, c, R, indices, rotated = [np.concatenate([r[i] for r in results], axis=0) for i in range(6)]


        
        if groups is not None:
            filtered_indices = []
            for i, index in enumerate(indices):
                if len([0 for j, id in enumerate(index) if find_in_group(groups[0], j) != find_in_group(groups[1], id)]) == 0:
                    filtered_indices.append(i)
        
            if len(filtered_indices):
                distances, approxs, c, R, indices, rotated =  distances[filtered_indices], approxs[filtered_indices], c[filtered_indices], R[filtered_indices], indices[filtered_indices], rotated[filtered_indices]
            else:
                distances, approxs, c, R, indices, rotated  = np.ones(1) , np.expand_dims(np.zeros_like(approxs[0]), axis=0),  np.expand_dims(np.zeros_like(c[0]), axis=0),  np.expand_dims(np.zeros_like(R[0]), axis=0),  np.expand_dims(np.zeros_like(indices[0]), axis=0),  np.expand_dims(np.zeros_like(rotated[0]), axis=0)
    else:
        distances, approxs, c, R, indices, rotated = fit_group(coords, ideal_coords, groups=groups)
    
    min_arg = np.argmin(distances)
    if all:
        return (distances, indices, distances[min_arg].squeeze(), rotated)
    
    return (distances[min_arg].squeeze(), approxs[min_arg][indices[min_arg]].squeeze(),  c[min_arg].ravel()[0], R[min_arg].squeeze(), indices[min_arg].ravel())