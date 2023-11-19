import numpy as np
from scipy.linalg import helmert
import itertools
import tensorflow as tf

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

def get_combinations(n1, groups=None):

    c_list = []
    ranges = [0] + np.cumsum(groups).tolist()
    for  i in range(len(ranges) - 1):
        c_list.append(np.fromiter(itertools.chain.from_iterable(itertools.permutations(range(ranges[i], ranges[i + 1]))), dtype=np.int32).reshape(-1, ranges[i + 1] - ranges[i]))

    combinations = np.vstack([np.concatenate(x) for x in itertools.product(*c_list)])  
    k = len(combinations)
    return combinations,k


def fit(coords, ideal_coords, groups=None, all = False):

    
    n1 = coords.shape[0]
    n2 = ideal_coords.shape[1]
    if groups is None:
        correspondense = [list(range(n1)), list(range(n1))]
        lengths = [n1]
    else:
        correspondense = [np.hstack(groups[0]).tolist(), np.hstack(groups[1]).tolist()]
        lengths = [len(gr) for gr in  groups[0]]

    x = tf.broadcast_to(tf.convert_to_tensor(coords[correspondense[0]], dtype='float32'), (1, n1, n2))    
    y = tf.broadcast_to(tf.convert_to_tensor(ideal_coords[correspondense[1]], dtype='float32'), (1, n1, n2))
    
    combinations, k = get_combinations(n1, groups=lengths)

    t_combinations = combinations.reshape(-1, combinations.shape[0], combinations.shape[1], 1)

    s_y = tf.gather_nd(y, indices=t_combinations, batch_dims=1)
    s_y = tf.reshape(s_y, (k*1, n1, coords.shape[1]))
    s_x = tf.broadcast_to(tf.convert_to_tensor(coords, dtype='float32'), (1, n1, coords.shape[1]))

    distances, approxs, c, R = procrustes_fit(s_y, s_x)
    distances = distances.numpy()
    min_distance = np.min(distances)
    mask = distances <= min_distance + 0.1
    
    distances = distances[mask]
    min_arg = np.argmin(distances)
    R = tf.boolean_mask(R, mask)
    c = tf.boolean_mask(c, mask)
    indices = tf.boolean_mask(combinations, mask).numpy()
    indices = indices.reshape(len(indices), n1)

    base_index = np.full((len(indices), n1), np.arange(n1))
    back_index = np.arange(n1)[np.argsort(correspondense[0])]

    approxs = (c*tf.broadcast_to(tf.convert_to_tensor(ideal_coords, dtype='float32'), (1, n1, n2))@R).numpy()

    x_index = base_index[:, correspondense[1]]
    indices = np.vstack([x[i] for x, i in zip(x_index, indices)])[:,back_index] 
  
    if all:
        return (distances, indices, distances[min_arg].squeeze())
    
    return (distances[min_arg].squeeze(), approxs[min_arg][indices[min_arg]].squeeze(),  c[min_arg].numpy().ravel()[0], R[min_arg].numpy().squeeze(), indices[min_arg].ravel())
