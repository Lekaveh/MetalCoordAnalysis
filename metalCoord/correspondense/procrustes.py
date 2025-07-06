import itertools
import numpy as np
from scipy.linalg import helmert
from sklearn.cluster import DBSCAN


def is_it_plane(xyz):
    """
    Determines if a set of 3D points lies approximately on a plane.

    Parameters:
    xyz (numpy.ndarray): A 2D array of shape (n, 3) representing the coordinates of n points in 3D space.

    Returns:
    tuple: A tuple containing two float values:
        - The root mean square deviation of the points from the plane.
        - The maximum absolute deviation of the points from the plane.

    Notes:
    - If the number of points is less than or equal to 3, the function returns (0.0, 0.0).
    - The function uses a Procrustes analysis approach to determine the plane.
    """
    r = np.zeros(shape=3)
    rp = np.zeros(shape=3)
    rpp = np.zeros(shape=3)
    xyz_l = np.zeros_like(xyz)
    l = len(xyz[:, 0])
    if l <= 3:
        return (0.0, 0.0)
    for i in range(3):
        r[i] = np.sum(xyz[0:l, i])
    r = r/l
    for i in range(3):
        xyz_l[0:l, i] = xyz[0:l, i]-r[i]
    inds = np.arange(l)
    for i in range(3):
        rp[i] = np.sum(xyz_l[inds, i] * np.sin(2*np.pi*inds/l))
        rpp[i] = np.sum(xyz_l[inds, i] * np.cos(2*np.pi*inds/l))
    cc = (2/l)**0.5
    rp = cc*rp
    rpp = cc*rpp
    nnorm = np.cross(rp, rpp)
    nnorm = nnorm/np.linalg.norm(nnorm)
    z = np.zeros(shape=l)
    for i in inds:
        z[i] = np.dot(xyz_l[i, 0:3], nnorm)
    return (np.sqrt(np.sum(z*z)/l), np.max(np.abs(z)))


def plane_from_points(p1, p2, p3):
    """Find the plane equation coefficients from three points."""
    # Create vectors from points
    v1 = p2 - p1
    v2 = p3 - p1

    # Normal vector to the plane
    n = np.cross(v1, v2)
    n = n / np.linalg.norm(n)

    # Plane equation coefficients
    a, b, c = n
    d = np.dot(n, p1)

    return a, b, c, d, n


def find_intersection(ap, p1, p2, p3):
    """Finds the intersection point of the perpendicular from ap to the plane."""
    a, b, c, d, n = plane_from_points(p1, p2, p3)
    x1, y1, z1 = ap
    # Parameter t calculation
    t = (d - a * x1 - b * y1 - c * z1) / (a**2 + b**2 + c**2)

    # Intersection point calculation
    x = x1 + a * t
    y = y1 + b * t
    z = z1 + c * t

    return np.array([x, y, z]), n


def angle(center, point1, point2, n):
    """
    Calculate the angle between two vectors in a right-handed coordinate system.
    Parameters:
    center (array-like): The center point from which the vectors originate.
    point1 (array-like): The endpoint of the first vector.
    point2 (array-like): The endpoint of the second vector.
    n (array-like): The normal vector to the plane in which the angle is measured.
    Returns:
    float: The angle between the two vectors in degrees.
    """
    a = point1 - center
    b = point2 - center
    a = np.asarray(a)/np.linalg.norm(a)
    b = np.asarray(b)/np.linalg.norm(b)

    right_handed_angle = np.arctan2(np.dot(np.cross(a, b), n), np.dot(a, b))
    return np.rad2deg(right_handed_angle)


def sort_ring(coords, ring):
    """
    Sorts the indices of a ring based on their angular position around a center point.

    Args:
        coords (list of tuples): A list of coordinate tuples representing points in space.
        ring (list of int): A list of indices representing the points that form the ring.

    Returns:
        list of int: The indices of the ring sorted by their angular position around the center point.
    """
   
    center, n = find_intersection(
        coords[0], coords[ring[0]], coords[ring[1]], coords[ring[2]])
    
    return sorted(ring, key=lambda x: angle(center, coords[ring[0]], coords[x], n))


def sort_rings(coords, rings):
    """
    Sorts a list of rings based on their coordinates.

    Args:
        coords (list of tuple): A list of tuples representing the coordinates.
        rings (list of list): A list of rings, where each ring is a list of indices corresponding to the coordinates.

    Returns:
        list of list: A list of sorted rings, where each ring is sorted based on the coordinates.
    """
    return [sort_ring(coords, ring) for ring in rings]


def find_rings(coords):
    """
    Identifies ring structures and other clusters in a set of coordinates.
    Parameters:
    coords (numpy.ndarray): A 2D array of coordinates where the first coordinate is the reference point.
    Returns:
    tuple: A tuple containing:
        - list: A sorted list of ring structures, each represented by a list of indices.
        - list: A list of indices that do not form ring structures.
    """
    rings = []
    others = []
    centred_coords = coords[1:] - coords[0]
    normed_coords = centred_coords / \
        np.max(np.linalg.norm(centred_coords, axis=1))
    clustering = DBSCAN(eps=1, min_samples=3).fit(normed_coords)
    clusters = np.unique(clustering.labels_)

    for cluster in clusters:
        indices = (np.where(clustering.labels_ == cluster)[0] + 1).tolist()
        if cluster == -1:
            others.append(indices)
            continue

        cluster_coords = normed_coords[clustering.labels_ == cluster]
        t1, t2 = is_it_plane(cluster_coords)
        if t1 <= 0.6 and t2 <= 0.6:
            rings.append(indices)
        else:
            others.append(indices)
    return sort_rings(coords, sorted(rings, key=len, reverse=True)), others


def have_same_ring_length(rings1, rings2):
    """
    Check if two lists of rings have the same lengths.
    This function compares two lists of rings (where each ring is represented 
    as a list of elements) and determines if they have the same number of rings 
    and if each corresponding ring has the same length.
    Parameters:
    rings1 (list of list): The first list of rings to compare.
    rings2 (list of list): The second list of rings to compare.
    Returns:
    bool: True if both lists have the same number of rings and each corresponding 
          ring has the same length, False otherwise.
    """
    if len(rings1) != len(rings2):
        return False

    for r1, r2 in zip(rings1, rings2):
        if len(r1) != len(r2):
            return False

    return True


def flatten_list(l):
    """
    Flattens a list of lists into a single list.

    Args:
        l (list of lists): A list where each element is a list.

    Returns:
        list: A single list containing all the elements of the sublists.
    """
    return [item for sublist in l for item in sublist]


def ring_permutations(ring):
    """
    Generate all cyclic permutations of a given ring.

    A ring is a sequence where the start and end are connected, forming a loop.
    This function returns a list of all possible cyclic permutations of the input ring.

    Parameters:
    ring (list): A list representing the ring to be permuted.

    Returns:
    list of lists: A list containing all cyclic permutations of the input ring.

    Example:
    >>> ring_permutations([1, 2, 3])
    [[1, 2, 3], [2, 3, 1], [3, 1, 2]]
    """
    n = len(ring)
    indices = list(range(n))
    return [[ring[(idx + i) % n] for idx in indices] for i in range(n)]


def norm(x, hm):
    """
    Computes the norm of a tensor using the Procrustes method.

    Args:
        x (np.ndarray): An array of shape (batch_size, n, m).
        hm (np.ndarray): An array of shape (batch_size, n, m).

    Returns:
        np.ndarray: The computed norm as an array.
    """
    return np.sqrt(np.trace(x.transpose(0, 2, 1) @ hm.transpose(0, 2, 1) @ hm @ x, axis1=1, axis2=2))


def preshape(x):
    """
    Transforms the input matrix `x` using the Helmert matrix and normalizes it.

    Args:
        x (np.ndarray): An array of shape (n_samples, n_features) representing the input data.

    Returns:
        np.ndarray: The transformed and normalized array.
    """
    hm = helmert(x.shape[1])
    hm = np.broadcast_to(hm.astype('float32'), (1, hm.shape[0], hm.shape[1]))
    return hm @ x / np.reshape(norm(x, hm), (-1, 1, 1))


def distance(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """
    Calculates the distance between two sets of coordinates using the Procrustes analysis.

    Args:
        x1: The first set of coordinates.
        x2: The second set of coordinates.

    Returns:
        Distances between the two sets of coordinates.
    """
    z2 = preshape(x2)
    z1 = preshape(x1)
    s = np.linalg.svd(z1.transpose(0, 2, 1) @ z2 @
                      z2.transpose(0, 2, 1) @ z1, compute_uv=False)
    return np.sqrt(np.abs(1 - np.sum(np.sqrt(s), axis=1)**2))

def frobenius_norm(A: np.ndarray) -> np.ndarray:
    """
    Computes the Frobenius norm of a matrix.

    Parameters:
    A (np.ndarray): The input matrix.

    Returns:
    np.ndarray: The Frobenius norm of the matrix.
    """
    
    return np.sqrt(np.sum(A**2, axis=(1, 2)))

def frobenius_distance(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Computes the Frobenius distance between two matrices A and B.

    Parameters:
    A (np.ndarray): The first matrix.
    B (np.ndarray): The second matrix.

    Returns:
    np.ndarray: The Frobenius distance between the two matrices.
    """
    return frobenius_norm(preshape(A)  - preshape(B) )

def procrustes_fit(A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes the Procrustes fit between two matrices A and B.

    Parameters:
    A (np.ndarray): The first matrix.
    B (np.ndarray): The second matrix.

    Returns:
    tuple: A tuple containing the following elements:
        - distance (np.ndarray): The distance between the approximated matrix and B.
        - approx (np.ndarray): The approximated matrix.
        - c (np.ndarray): The scaling factor.
        - R (np.ndarray): The rotation matrix.
    """
    A_t = A.transpose(0, 2, 1)
    v, _, wt = np.linalg.svd(A_t @ B)
    R = v @ wt

    c = np.trace(R.transpose(0, 2, 1) @ A_t @ B, axis1=1,
                 axis2=2) / np.trace(A_t @ A, axis1=1, axis2=2)
    c = np.reshape(c, (-1, 1, 1))

    approx = c * A @ R
    return (frobenius_distance(approx,  B), approx, c, R)


def get_combinations(groups=None, rings=None):
    """
    Generate all possible combinations of permutations for given groups and rings.

    Parameters:
    groups (list of int): A list where each element represents the size of a group.
    rings (list of int, optional): A list where each element is a flag (0 or 1) indicating 
                                   whether the corresponding group should be treated as a ring. 
                                   If None, all groups are treated as non-rings.
    Returns:
    tuple: A tuple containing:
        - combinations (np.ndarray): A 2D array where each row is a unique combination of permutations.
        - k (int): The total number of combinations.
    """
    c_list = []
    ranges = [0] + np.cumsum(groups).tolist()
    if rings is None:
        rings = [0] * len(groups)

    for i in range(len(ranges) - 1):
        if rings[i]:
            perms = ring_permutations(range(ranges[i], ranges[i + 1]))
            perms_rev = ring_permutations(
                list(range(ranges[i], ranges[i + 1]))[::-1])
            c_list.append(np.array(perms + perms_rev))
        else:
            perms = itertools.permutations(range(ranges[i], ranges[i + 1]))
            c_list.append(np.array(list(perms), dtype=np.int32))

    combinations = np.array([np.concatenate(x)
                            for x in itertools.product(*c_list)])
    k = len(combinations)
    return combinations, k

class Procustes:
    """
    Procustes class for performing Procrustes analysis on coordinate data.
    Methods
    -------
    __init__(self, coords, ideal_coords)
        Initializes the Procustes object with the given coordinates and ideal coordinates.
    fit(self, coords, ideal_coords, groups=None, all=False, center=True)
        Fits the Procrustes model to the given coordinates and ideal coordinates.
    _dummy(self, all=False)
        Returns dummy results when no valid results are found.
    _fit(self, combinations)
        Performs the Procrustes fitting on the given combinations of indices.
    _get_next_combinations(self)
        Generates the next set of combinations for Procrustes fitting.
    _get_combinations(self, current_index)
        Generates all possible combinations of indices for the next step of Procrustes fitting.
    is_finished(self)
        Checks if the Procrustes fitting process is finished.
    """

    def _init(self, coords, ideal_coords):
        """
        Initialize the Procrustes analysis object.

        Args:
            coords (np.ndarray): Coordinate array to be fitted
            ideal_coords (np.ndarray): Ideal coordinate array to fit to
        """
        self._coords = coords
        self._ideal_coords = ideal_coords
        self._n = len(coords)
        self._n_processed = 1
        self._step = 6
        self._index = list(range(self._n))
        self._current_indices = [[0]]
        self._dim1 = coords.shape[0]
        self._dim2 = coords.shape[1]

        # Replace TensorFlow broadcasting with NumPy
        self._y = np.broadcast_to(
            ideal_coords.astype('float32'),
            (1, self._dim1, self._dim2)
        )
        self._x = np.broadcast_to(
            ideal_coords.astype('float32'),
            (1, self._dim1, self._dim2)
        )
        self._result = []

    def fit(self, coords, ideal_coords, groups=None, all=False, center=True):
        """
        Fits the given coordinates to the ideal coordinates using the Procrustes analysis.
        Parameters:
        coords (numpy.ndarray): The coordinates to be fitted.
        ideal_coords (numpy.ndarray): The ideal coordinates to fit to.
        groups (list, optional): A list of groups for filtering indices. Defaults to None.
        all (bool, optional): If True, returns all distances and indices. Defaults to False.
        center (bool, optional): If True, centers the coordinates before fitting. Defaults to True.
        Returns:
        tuple: If `all` is False, returns a tuple containing:
            - float: The minimum distance.
            - numpy.ndarray: The approximated coordinates corresponding to the minimum distance.
            - float: The translation component.
            - numpy.ndarray: The rotation matrix.
            - numpy.ndarray: The indices of the best fit.
        If `all` is True, returns a tuple containing:
            - numpy.ndarray: All distances.
            - numpy.ndarray: All indices.
            - float: The minimum distance.
            - numpy.ndarray: All rotated coordinates.
        """
        if center:
            self._init(coords - coords[0], ideal_coords - ideal_coords[0])
        else:
            self._init(coords, ideal_coords)

        while not self.is_finished():
            self._get_next_combinations()

        if self._results:
            distances, approxs, c, R, indices, rotated = [np.concatenate(
                [r[i] for r in self._results], axis=0) for i in range(6)]

            if groups is not None:
                filtered_indices = []
                for i, index in enumerate(indices):
                    if len([0 for j, id in enumerate(index) if find_in_group(groups[0], j) != find_in_group(groups[1], id)]) == 0:
                        filtered_indices.append(i)

                if filtered_indices:
                    distances, approxs, c, R, indices, rotated = distances[filtered_indices], approxs[filtered_indices], c[
                        filtered_indices], R[filtered_indices], indices[filtered_indices], rotated[filtered_indices]
                else:
                    return self._dummy(all)

            if all:
                return (distances, indices, distances[np.argmin(distances)].squeeze() if len(distances) else 1, rotated)

            min_arg = np.argmin(distances)
            return (distances[min_arg].squeeze(), approxs[min_arg][indices[min_arg]].squeeze(),  c[min_arg].ravel()[0], R[min_arg].squeeze(), indices[min_arg].ravel())

        return self._dummy(all)

    def _dummy(self, all=False):
        if all:
            return ([], [], 1, [])
        return (1, None, None, None, [])

    def _fit(self, combinations):
        """
        Internal fitting method that performs Procrustes analysis on given combinations.

        Args:
            combinations (np.ndarray): Array of index combinations to try

        Returns:
            tuple: Contains distances, approximations, scaling factors, rotation matrices,
                indices, and rotated coordinates
        """
        k = combinations.shape[0]
        n1 = combinations.shape[1]

        # Initialize s_y array
        s_y = np.zeros((k, n1, self._dim2), dtype='float32')

        # Apply permutations directly
        for i in range(k):
            s_y[i] = self._y[0, combinations[i]]

        # Create broadcast arrays
        s_x = np.broadcast_to(
            self._coords[:n1].astype('float32'),
            (1, n1, self._dim2)
        )

        # Compute Procrustes fit
        distances, approxs, c, R = procrustes_fit(s_y, s_x)

        # Apply masking based on distance threshold
        min_dist = min(np.nanmin(distances) + 0.1, 0.2)
        mask = distances <= min_dist

        distances = distances[mask]
        R = R[mask]
        c = c[mask]
        indices = combinations[mask]
        indices = indices.reshape(len(indices), n1)

        # Compute rotated coordinates
        rotated = np.broadcast_to(
            self._ideal_coords[:n1].astype('float32'),
            (1, n1, self._dim2)
        ) @ R

        approxs = c * rotated

        return distances, approxs, c, R, indices, rotated

    def _get_next_combinations(self):
        self._results = []
        next_step = min(self._step, self._n - self._n_processed)
        for current_index in self._current_indices:
            combinations = self._get_combinations(current_index)
            result = self._fit(combinations)
            if result[0].size:
                self._results.append(result)

        if self._results:
            self._current_indices = np.concatenate(
                [r[4].tolist() for r in self._results], axis=0).tolist()
            self._n_processed += next_step
        else:
            self._current_indices = []
            self._n_processed = self._n

    def _get_combinations(self, current_index):
        result = []
        step = min(self._step, self._n - self._n_processed)
        candidates = np.setdiff1d(self._index, current_index)

        for next_indices in np.fromiter(itertools.chain.from_iterable(itertools.combinations(candidates, step)), dtype=np.float32).reshape(-1, step):
            permutations = np.fromiter(itertools.chain.from_iterable(
                itertools.permutations(next_indices)), dtype=int).reshape(-1, step)
            result.extend([current_index + permutation.tolist()
                          for permutation in permutations])
        return np.array(result)

    def is_finished(self):
        """
        Check if the processing is finished.

        Returns:
            bool: True if the number of processed items equals the total number of items, False otherwise.
        """
        return self._n_processed == self._n
    
fitter =  Procustes()

def fit_group(coords, ideal_coords, groups=None, rings=None):
    """
    Fits a group of coordinates to ideal coordinates using Procrustes analysis.

    Parameters:
    coords (np.ndarray): The coordinates to be fitted, shape (n1, m).
    ideal_coords (np.ndarray): The ideal coordinates to fit to, shape (n2, m).
    groups (list of lists, optional): Groups of indices for correspondence. Default is None.
    rings (list, optional): Additional parameter for get_combinations function. Default is None.

    Returns:
    tuple: A tuple containing:
        - distances (np.ndarray): The distances after fitting.
        - approxs (np.ndarray): The approximated coordinates after fitting.
        - c (np.ndarray): The scaling factors.
        - R (np.ndarray): The rotation matrices.
        - indices (np.ndarray): The indices of the best fit.
        - rotated (np.ndarray): The rotated ideal coordinates.
    """
    n1 = coords.shape[0]
    n2 = ideal_coords.shape[1]

    if groups is None:
        correspondense = [list(range(n1)), list(range(n1))]
        lengths = [n1]
    else:
        correspondense = [
            np.hstack(groups[0]).tolist(), np.hstack(groups[1]).tolist()]
     
        lengths = [len(gr) for gr in groups[0]]
    
    y = np.broadcast_to(ideal_coords[correspondense[1]].astype('float32'),
                        (1, n1, n2))

    combinations, k = get_combinations(groups=lengths, rings=rings)

    # Create s_y by applying the permutations
    s_y = np.zeros((k, n1, coords.shape[1]), dtype='float32')
    for i, comb in enumerate(combinations):
        s_y[i] = y[0, comb]

    s_x = np.broadcast_to(coords[correspondense[0]].astype('float32'),
                          (1, n1, n2))

    distances, approxs, c, R = procrustes_fit(s_y, s_x)

    min_distance = np.nanmin(distances)
    mask = distances <= min_distance + 0.2
    distances = distances[mask]

    R = R[mask]
    c = c[mask]
    indices = combinations[mask]
    indices = indices.reshape(len(indices), n1)

    back_index = np.argsort(correspondense[0])

    rotated = np.broadcast_to(ideal_coords.astype('float32'),
                              (1, n1, n2)) @ R
    approxs = (c * rotated)

    x_index = np.full((len(indices), n1), correspondense[1])

    indices = np.vstack([x[i] for x, i in zip(x_index, indices)])[
        :, back_index]
    return distances, approxs, c, R, indices, rotated

def create_group(rings, others):
    """
    Creates a group by combining a list of rings and a list of other elements.

    Parameters:
    rings (list): A list of lists, where each sublist represents a ring.
    others (list): A list of other elements to be included in the group.

    Returns:
    list: A combined list containing a single-element list [0], followed by the elements of `rings`, 
          and optionally followed by `others` if it is not empty.
    """
    if len(others):
        return [[0]] + rings + others
    return [[0]] + rings

def find_in_group(groups, index):
    """
    Find the index of the group that contains the specified index.

    Args:
        groups (list of list of int): A list of groups, where each group is a list of indices.
        index (int): The index to search for within the groups.

    Returns:
        int: The index of the group that contains the specified index, or -1 if the index is not found in any group.
    """
    for i, group in enumerate(groups):
        if index in group:
            return i
    return -1


def fit(coords: np.ndarray, ideal_coords: np.ndarray, groups: tuple = None, all: bool = False, center: bool = True):
    """
    Fits the given coordinates to the ideal coordinates using Procrustes analysis.
    Parameters:
    coords (np.ndarray): The coordinates to be fitted.
    ideal_coords (np.ndarray): The ideal coordinates to fit to.
    groups (tuple, optional): A tuple of groups to be considered during fitting. Defaults to None.
    all (bool, optional): If True, returns all fitting results. Defaults to False.
    center (bool, optional): If True, centers the coordinates before fitting. Defaults to True.
    Returns:
    tuple: If `all` is False, returns a tuple containing:
        - float: The minimum distance after fitting.
        - np.ndarray: The approximated coordinates after fitting.
        - float: The scaling factor.
        - np.ndarray: The rotation matrix.
        - np.ndarray: The indices of the fitted coordinates.
        If `all` is True, returns a tuple containing:
        - np.ndarray: All distances after fitting.
        - np.ndarray: All indices of the fitted coordinates.
        - float: The minimum distance after fitting.
        - np.ndarray: The rotated coordinates.
    """
    
    if groups is None:
        groups = [[[0], list(range(1, len(coords)))], [[0], list(range(1, len(ideal_coords)))]]
                  
    if center:
        coords = coords - coords[0]
        ideal_coords = ideal_coords - ideal_coords[0]

    rings1, others1 = find_rings(coords)
    rings2, others2 = find_rings(ideal_coords)
     
    if len(coords) >= 8 and len(rings1) > 0 and have_same_ring_length(rings1, rings2):
            
        # Group rings by length more efficiently
        ring_length_dict = {}
        for i, ring in enumerate(rings2):
            length = len(ring)
            if length not in ring_length_dict:
                ring_length_dict[length] = []
            ring_length_dict[length].append(i)
        
        # Handle permutations more efficiently
       
        ring_dict_permutations = {}
        for key, group in ring_length_dict.items():
            ring_dict_permutations[key] = [list(perm) for perm in itertools.permutations(group)]

      
        # Generate final permutations more memory-efficiently
        ring_group_permutations = []
       
        if len(ring_dict_permutations) == 1:
  
            ring_combinations = list(ring_dict_permutations.values())[0]
            
            for combination in ring_combinations:
                group = create_group(list([rings2[i] for i in combination]), others2)
                ring_group_permutations.append(group)
        else:
            
            ring_combinations = itertools.product(*ring_dict_permutations.values())
            for combination in ring_combinations:
                group = create_group([rings2[i] for gr in list(combination) for i in gr] , others2)
                ring_group_permutations.append(group)
        
        ring1_group = create_group(rings1, others1)
        results = []
        for ring2_group in ring_group_permutations:
            current_groups = [ring1_group, ring2_group]
            results.append(fit_group(coords, ideal_coords, current_groups, [
                            0] + [1 for i, x in enumerate(rings1)] + [0]))

        distances, approxs, c, r, indices, rotated = [
            np.concatenate([r[i] for r in results], axis=0) for i in range(6)]

        if groups is not None:
            filtered_indices = []
            for i, index in enumerate(indices):
                if len([0 for j, id in enumerate(index) if find_in_group(groups[0], j) != find_in_group(groups[1], id)]) == 0:
                    filtered_indices.append(i)

            if filtered_indices:
                distances, approxs, c, r, indices, rotated = distances[filtered_indices], approxs[filtered_indices], c[
                    filtered_indices], r[filtered_indices], indices[filtered_indices], rotated[filtered_indices]
            else:
                distances, approxs, c, r, indices, rotated = np.ones(1), np.expand_dims(np.zeros_like(approxs[0]), axis=0),  np.expand_dims(np.zeros_like(
                    c[0]), axis=0),  np.expand_dims(np.zeros_like(r[0]), axis=0),  np.expand_dims(np.zeros_like(indices[0]), axis=0),  np.expand_dims(np.zeros_like(rotated[0]), axis=0)
    
    # elif len(coords) >= 10:
    #     return  fitter.fit(coords, ideal_coords, groups, all, center)
    else:
        distances, approxs, c, r, indices, rotated = fit_group(
            coords, ideal_coords, groups)


    min_arg = np.argmin(distances)
    if all:
        return (distances, indices, distances[min_arg].squeeze(), rotated)
    
    return (distances[min_arg].squeeze(), approxs[min_arg][indices[min_arg]].squeeze(),  c[min_arg].ravel()[0], r[min_arg].squeeze(), indices[min_arg].ravel())



