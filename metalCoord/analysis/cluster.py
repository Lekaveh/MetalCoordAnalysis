import numpy as np
from scipy import stats
import sklearn.mixture


def _inlist(x, lst, x_tol=1e-5):
    """
    Check if a value is present in a list within a tolerance.

    Args:
        x (float): The value to check.
        lst (list): The list of values to search in.
        x_tol (float, optional): The tolerance for comparing values. Defaults to 1e-5.

    Returns:
        bool: True if the value is present in the list within the tolerance, False otherwise.
    """
    for v in lst:
        if np.abs(x - v) < x_tol:
            return True
    return False


def _remove_dup(duplicate, x_tol=1e-5):
    """
    Remove duplicates from a list and sort its values.

    Args:
        duplicate (list): The list with duplicates.
        x_tol (float, optional): The tolerance for comparing values. Defaults to 1e-5.

    Returns:
        list: The list with duplicates removed and sorted values.
    """
    final_list = []
    for num in duplicate:
        if not _inlist(num, final_list, x_tol):
            final_list.append(num)
    return final_list


def _filter_bymax(values, f):
    """
    Filter a list based on a function that calculates a score for each value.

    Args:
        values (list): The list to filter.
        f (function): The scoring function.

    Returns:
        list: The filtered list.
    """
    mx = max([f(v) for v in values])
    return [v for v in values if f(v)/mx > 0.1]


def _modes(data, kernel, x_tol=1e-5, rnd=2, neighbour=1):
    """
    Find modes in a dataset using a kernel density estimation.

    Args:
        data (array-like): The dataset.
        kernel (function): The kernel function for density estimation.
        x_tol (float, optional): The tolerance for comparing values. Defaults to 1e-5.
        rnd (int, optional): The number of decimal places to round the modes. Defaults to 2.
        neighbour (int, optional): The number of neighbors to consider when checking for modes. Defaults to 1.

    Returns:
        tuple: A tuple containing the number of modes found and the list of modes.
    """
    result = list()
    length = np.max(data) - np.min(data)
    line = np.linspace(start=np.min(data) - 0.1*length,
                       stop=np.max(data) + 0.1 * length, num=100)
    f = kernel(line)

    for i in np.arange(neighbour, len(line) - neighbour):
        is_max = True
        for j in np.arange(neighbour):
            if f[i] < f[i + j + 1] or f[i] < f[i - j - 1]:
                is_max = False
                break
        if is_max:
            result.append(np.round(line[i], rnd))
    if len(result) == 0:
        if neighbour > 1:
            return _modes(data, kernel, x_tol, rnd, neighbour - 1)
        else:
            return (len(result), result)
    result.sort()
    result = _filter_bymax(_remove_dup(result, x_tol), kernel)
    return (len(result), result)


def _kde_silverman(data, rnd=2, neighbour=10):
    """
    Perform kernel density estimation using the Silverman's rule of thumb for bandwidth selection.

    Args:
        data (array-like): The dataset.
        rnd (int, optional): The number of decimal places to round the modes. Defaults to 2.
        neighbour (int, optional): The number of neighbors to consider when checking for modes. Defaults to 10.

    Returns:
        tuple: A tuple containing the number of modes found and the kernel density estimation object.
    """
    kernel = stats.gaussian_kde(data, bw_method='silverman')
    result = _modes(data, kernel, neighbour=neighbour, rnd=rnd)
    return (result, kernel)


def modes(dists):
    """
    Calculate the modes of a distribution.

    Args:
        dists (array-like): The distribution.

    Returns:
        array: An array containing the means and standard deviations of the modes.
    """
    common_stats = np.array([[np.mean(dists)], [np.std(dists)]])
    if len(dists) < 30:
        return common_stats
    n_modes = _kde_silverman(dists, rnd=1, neighbour=3)[0][0]
    if n_modes < 2:
        return common_stats

    clf = sklearn.mixture.GaussianMixture(
        n_components=n_modes, covariance_type='full', random_state=0)
    clf.fit(dists.reshape(-1, 1))

    means = clf.means_.squeeze()
    stds = np.sqrt(clf.covariances_).squeeze()
    index = np.argsort(means)

    return np.array([means[index], stds[index]])
