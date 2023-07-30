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

def fit(coords, ideal_coords, n=1, core_vectors_num = 4):
    n1 = ideal_coords.shape[0]
    n2 = ideal_coords.shape[1]
    x = tf.broadcast_to(tf.convert_to_tensor(coords, dtype='float32'), (1, n1, n2))

    y = tf.broadcast_to(tf.convert_to_tensor(ideal_coords, dtype='float32'), (n, n1, n2))
    
    indices = np.apply_along_axis(np.random.permutation, 1,  np.broadcast_to(np.arange(1, n1), (n, n1 - 1)))
    indices = np.insert(indices, 0, 0, axis=1)
    indices = indices.reshape(-1, indices.shape[1], 1)

    perm_y = tf.gather_nd(y, indices=indices, batch_dims=1)

    combinations = np.fromiter(itertools.chain.from_iterable(itertools.combinations(range(1, n1), core_vectors_num - 1)), dtype=np.int32)
    
    combinations = combinations.reshape(-1, core_vectors_num - 1)
    combinations = np.apply_along_axis(np.random.permutation, 1,  np.repeat(combinations, np.math.factorial(core_vectors_num - 1), axis = 0))
    k = len(combinations)
    combinations = np.insert(combinations, 0, 0, axis=1)
    combinations = np.broadcast_to(combinations, (n, combinations.shape[0], combinations.shape[1]))
    
    s_y = tf.gather_nd(perm_y, indices=combinations.reshape(-1, combinations.shape[1], combinations.shape[2], 1), batch_dims=1)
    s_y = tf.reshape(s_y, (k*n, core_vectors_num, 3))
    s_x = tf.broadcast_to(tf.convert_to_tensor(coords[:core_vectors_num], dtype='float32'), (1, core_vectors_num, coords.shape[1]))

    distances, approxs, c, R = procrustes_fit(s_y, s_x)
    distances = distances.numpy()
    min_distance = np.min(distances)
    indices = np.argsort(distances)
    mask = distances <= min_distance + 0.1

    
    R = tf.boolean_mask(R, mask)
    c = tf.boolean_mask(c, mask)


    approxs = c*tf.broadcast_to(tf.convert_to_tensor(
        ideal_coords, dtype='float32'), (1, n1, n2))@R
    
    approxs_numpy = approxs.numpy()
    
    mask = np.ones(coords.shape[0], bool) 
    indices = np.zeros((approxs_numpy.shape[0], coords.shape[0]), dtype=np.int32)
    for k, approx in enumerate(approxs_numpy):
        mask[...] = 1
        mask[0] = 0
        for i in np.arange(1, approx.shape[0]):
            min_arg = np.argmax(np.exp(-np.sum((approx[i] - coords)**2,  axis=1).T)*mask)
            indices[k, min_arg] = i
            mask[min_arg] = 0
   
    indices = indices.reshape(-1, indices.shape[1], 1)
    perm_y = tf.gather_nd(approxs, indices=indices, batch_dims=1)

    distances = distance(x, perm_y).numpy()

    min_arg = np.argmin(distances) 
    return (distances[min_arg].squeeze(), approxs_numpy[min_arg][indices[min_arg].ravel()].squeeze(),  c[min_arg].numpy().ravel()[0], R[min_arg].numpy().squeeze(), indices[min_arg].ravel())
