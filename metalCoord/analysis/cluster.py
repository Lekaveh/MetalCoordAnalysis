import numpy as np
from scipy import stats
import sklearn.mixture
  



#This function double value and list of doubles. Returns true if value contains in the list
def _inlist(x, l, x_tol = 1e-5):
    for v in l:
        if np.abs(x - v) < x_tol:
            return True
    return False    

 #This function remove duplicates in list and sorts its values
def _remove_dup(duplicate, x_tol = 1e-5): 
    final_list = [] 
    for num in duplicate: 
        if not _inlist(num, final_list, x_tol): 
            final_list.append(num) 
    return final_list  


def _filter_bymax(l, f):
    mx = max([f(v) for v in l])
    return [v for v in l if f(v)/mx > 0.1]

  

def _modes(data, kernel, x_tol = 1e-5, rnd = 2, neighbour = 1):    
    result = list()
    length = np.max(data) - np.min(data)
    line = np.linspace(start= np.min(data) - 0.1*length, stop = np.max(data) + 0.1 * length, num = 100)
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
            (len(result), result)
    result.sort()        
    result = _filter_bymax(_remove_dup(result, x_tol), kernel)
    return (len(result), result) 


    


def _kde_silverman(data, rnd=2, neighbour = 10):
    kernel = stats.gaussian_kde(data, bw_method='silverman')
    result = _modes(data, kernel, neighbour = neighbour, rnd = rnd)
    return (result, kernel)
           

def modes(dists):
    common_stats = np.array([[np.mean(dists)], [np.std(dists)]])
    if  len(dists) < 30:
        return common_stats
    n_modes = _kde_silverman(dists, rnd=1, neighbour = 3)[0][0]  
    if n_modes < 2:
        return common_stats
    
    clf = sklearn.mixture.GaussianMixture(n_components=n_modes, covariance_type='full', random_state=0)
    clf.fit(dists.reshape(-1, 1))
    
    means = clf.means_.squeeze()
    stds = np.sqrt(clf.covariances_).squeeze()
    index = np.argsort(means)

    return np.array([means[index], stds[index]])

