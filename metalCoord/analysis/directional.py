from abc import ABC, abstractmethod
import numpy as np
from scipy.optimize import root_scalar
from scipy.special import i0, i1
from scipy.optimize import minimize

class VonMises(ABC):
    def __init__(self, X):
        self._X = np.deg2rad(X)
        self._c = np.mean(np.cos(self._X))
        self._s = np.mean(np.sin(self._X))
        self._alpha = np.arctan2(self._s, self._c)
        self._n = len(X)
        self._kappa = self._kappa_est()
        self._m = self.m(self._kappa)
 

    
    
    def _f(self, x):
        return self.m(x) - np.sqrt(self._c**2 + self._s**2)
    
    def _df(self, x):
        if x > 300:
            return 1/np.sqrt(1 - 1/x)/2/x**2 
        i_0 = i0(x)
        i_1 = i1(x)
        return (i_0 **2 - i_1**2 - i_0*i_1/x)/ i_0**2
    
    def _kappa_est(self):
        return root_scalar(self._f, x0=0.001, fprime=self._df,  method='newton').root
    
    @property
    def mu(self):
        return np.rad2deg(self._alpha)
    
    @property
    def sigma(self):
        return np.rad2deg(np.sqrt(1/self._kappa))
    
    @property
    def kappa(self):
        return self._kappa


    def m(self, x):
        if x > 300:
            return np.sqrt(1 - 1/x) 
        return i1(x) / i0(x)
    
    @abstractmethod
    def pdf(self, x):
        pass

    abstractmethod
    def _loglikelihood(self, X):
        pass    

  
    @abstractmethod 
    def _dldx1(self, X):
        pass

    @abstractmethod    
    def _dldx2(self, X):
        pass
    

    def _gradient(self, X):
        return np.array([self._dldx1(X), self._dldx2(X)])
    

    def loglikelihood(self):
        return self._loglikelihood(self._kappa * np.cos(self._alpha), self._kappa * np.sin(self._alpha)) 
    
    
    def fit(self):
        res = minimize(self._loglikelihood, x0=[self._kappa * np.cos(self._alpha), self._kappa * np.sin(self._alpha)], jac=self._gradient, method='BFGS')
        if res.success:
            self._kappa = np.sqrt(res.x[0]**2 + res.x[1]**2)
            self._alpha = np.arctan2(res.x[1], res.x[0])
        return res.success
    
    def __repr__(self):
        return f"VonMises(mu={self.mu}, kappa={self.kappa})"
        
class UnsymmetrisedVonMises(VonMises):
    def __init__(self, X):
        super().__init__(X)
    
    def pdf(self, x):
        return 1 / (2 * np.pi * i0(x))* np.exp(self._kappa * np.cos(x - self._alpha))

     
    def _loglikelihood(self, X):
        x1, x2 = X
        x = np.sqrt(x1**2 + x2**2)
        return self._n*np.log(i0(x)) - np.sum(x1*np.cos(self._X) + x2*np.sin(self._X))

    def _dldx1(self, X):
        x1, x2 = X
        x = np.sqrt(x1**2 + x2**2)
        return  -self._n*(self._c - self.m(x)*x1/x)
        
    def _dldx2(self, X):
        x1, x2 = X
        x = np.sqrt(x1**2 + x2**2)
        return  -self._n*(self._s - self.m(x)*x2/x)

class SymmetrisedVonMises(VonMises):
    def __init__(self, X):
        super().__init__(X)
    

    def pdf(self, x):
        return 1 / (2 * np.pi * i0(x))* np.exp(self._kappa * np.cos(x - self._alpha)) + self._kappa * np.cos(x + self._alpha)
     
    def _loglikelihood(self, X):
        x1, x2 = X
        x = np.sqrt(x1**2 + x2**2)
        return self._n*np.log(i0(x)) - np.sum(x1*np.cos(self._X) +np.log(np.cosh(x2*np.sin(self._X))))

    def _dldx1(self, X):
        x1, x2 = X
        x = np.sqrt(x1**2 + x2**2)
        return  -self._n*self._c + self.m(x)*self._n*x1/x
        
    def _dldx2(self, X):
        x1, x2 = X
        x = np.sqrt(x1**2 + x2**2)
        return  -np.sum(np.tanh(x2*np.sin(self._X))*np.sin(self._X)) + self.m(x)*self._n*x2/x

def calculate_stats(X):
    vonMises = None
    if np.mean(X) > 150:
        vonMises = SymmetrisedVonMises(X)
    else:
        vonMises = UnsymmetrisedVonMises(X)
    vonMises.fit()
    return np.clip(np.abs(vonMises.mu), 0, 180), vonMises.sigma   