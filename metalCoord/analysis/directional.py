from abc import ABC, abstractmethod
import numpy as np
from scipy.optimize import root_scalar
from scipy.special import i0, i1
from scipy.optimize import minimize


class VonMises(ABC):
    """
    Abstract base class representing a Von Mises distribution.

    Attributes:
        _X (numpy.ndarray): Array of angles in radians.
        _c (float): Mean cosine of the angles.
        _s (float): Mean sine of the angles.
        _alpha (float): Mean direction of the angles.
        _n (int): Number of angles.
        _kappa (float): Concentration parameter of the distribution.
        _m (float): Function of the concentration parameter.

    Methods:
        _f(x): Function used to estimate the concentration parameter.
        _df(x): Derivative of the function used to estimate the concentration parameter.
        _kappa_est(): Estimate the concentration parameter.
        mu(): Mean direction in degrees.
        sigma(): Standard deviation in degrees.
        kappa(): Concentration parameter.
        m(x): Function of the concentration parameter.
        pdf(x): Probability density function.
        _loglikelihood(X): Log-likelihood function.
        _dldx1(X): Partial derivative of the log-likelihood function with respect to x1.
        _dldx2(X): Partial derivative of the log-likelihood function with respect to x2.
        _gradient(X): Gradient of the log-likelihood function.
        loglikelihood(): Log-likelihood value.
        fit(): Fit the distribution to the data.
        __repr__(): String representation of the VonMises object.
    """

    def __init__(self, X):
        """
        Initialize the VonMises object.

        Args:
            X (numpy.ndarray): Array of angles in degrees.
        """
        self._X = np.deg2rad(X)
        self._c = np.mean(np.cos(self._X))
        self._s = np.mean(np.sin(self._X))
        self._alpha = np.arctan2(self._s, self._c)
        self._n = len(X)
        self._kappa = self._kappa_est()
        self._m = self.m(self._kappa)

    def _f(self, x):
        """
        Function used to estimate the concentration parameter.

        Args:
            x (float): Value of the concentration parameter.

        Returns:
            float: Difference between the function value and the square root of the mean cosine and sine squared.
        """
        return self.m(x) - np.sqrt(self._c ** 2 + self._s ** 2)

    def _df(self, x):
        """
        Derivative of the function used to estimate the concentration parameter.

        Args:
            x (float): Value of the concentration parameter.

        Returns:
            float: Derivative of the function.
        """
        if x > 300:
            return 1 / np.sqrt(1 - 1 / x) / 2 / x ** 2
        i_0 = i0(x)
        i_1 = i1(x)
        return (i_0 ** 2 - i_1 ** 2 - i_0 * i_1 / x) / i_0 ** 2

    def _kappa_est(self):
        """
        Estimate the concentration parameter.

        Returns:
            float: Estimated concentration parameter.
        """
        return root_scalar(self._f, x0=0.001, fprime=self._df, method='newton').root

    @property
    def mu(self):
        """
        Mean direction in degrees.

        Returns:
            float: Mean direction in degrees.
        """
        return np.rad2deg(self._alpha)

    @property
    def sigma(self):
        """
        Standard deviation in degrees.

        Returns:
            float: Standard deviation in degrees.
        """
        return np.rad2deg(np.sqrt(1 / self._kappa))

    @property
    def kappa(self):
        """
        Concentration parameter.

        Returns:
            float: Concentration parameter.
        """
        return self._kappa

    def m(self, x):
        """
        Function of the concentration parameter.

        Args:
            x (float): Value of the concentration parameter.

        Returns:
            float: Function value.
        """
        if x > 300:
            return np.sqrt(1 - 1 / x)
        return i1(x) / i0(x)

    @abstractmethod
    def pdf(self, x):
        """
        Probability density function.

        Args:
            x (float): Value at which to evaluate the PDF.

        Returns:
            float: PDF value.
        """
        pass

    @abstractmethod
    def _loglikelihood(self, X):
        """
        Log-likelihood function.

        Args:
            X (numpy.ndarray): Array of angles.

        Returns:
            float: Log-likelihood value.
        """
        pass

    @abstractmethod
    def _dldx1(self, X):
        """
        Partial derivative of the log-likelihood function with respect to x1.

        Args:
            X (numpy.ndarray): Array of angles.

        Returns:
            float: Partial derivative value.
        """
        pass

    @abstractmethod
    def _dldx2(self, X):
        """
        Partial derivative of the log-likelihood function with respect to x2.

        Args:
            X (numpy.ndarray): Array of angles.

        Returns:
            float: Partial derivative value.
        """
        pass

    def _gradient(self, X):
        """
        Gradient of the log-likelihood function.

        Args:
            X (numpy.ndarray): Array of angles.

        Returns:
            numpy.ndarray: Gradient vector.
        """
        return np.array([self._dldx1(X), self._dldx2(X)])

    def loglikelihood(self):
        """
        Log-likelihood value.

        Returns:
            float: Log-likelihood value.
        """
        return self._loglikelihood(self._kappa * np.cos(self._alpha), self._kappa * np.sin(self._alpha))

    def fit(self):
        """
        Fit the distribution to the data.

        Returns:
            bool: True if the fitting is successful, False otherwise.
        """
        res = minimize(self._loglikelihood, x0=[self._kappa * np.cos(self._alpha), self._kappa * np.sin(self._alpha)],
                       jac=self._gradient, method='BFGS')
        if res.success:
            self._kappa = np.sqrt(res.x[0] ** 2 + res.x[1] ** 2)
            self._alpha = np.arctan2(res.x[1], res.x[0])
        return res.success

    def __repr__(self):
        """
        String representation of the VonMises object.

        Returns:
            str: String representation.
        """
        return f"VonMises(mu={self.mu}, kappa={self.kappa})"


class UnsymmetrisedVonMises(VonMises):
    """
    Class representing an unsymmetrised Von Mises distribution.

    Inherits from VonMises.

    Methods:
        pdf(x): Probability density function.
        _loglikelihood(X): Log-likelihood function.
        _dldx1(X): Partial derivative of the log-likelihood function with respect to x1.
        _dldx2(X): Partial derivative of the log-likelihood function with respect to x2.
    """

    def __init__(self, X):
        """
        Initialize the UnsymmetrisedVonMises object.

        Args:
            X (numpy.ndarray): Array of angles in degrees.
        """
        super().__init__(X)

    def pdf(self, x):
        """
        Probability density function.

        Args:
            x (float): Value at which to evaluate the PDF.

        Returns:
            float: PDF value.
        """
        return 1 / (2 * np.pi * i0(x)) * np.exp(self._kappa * np.cos(x - self._alpha))

    def _loglikelihood(self, X):
        """
        Log-likelihood function.

        Args:
            X (numpy.ndarray): Array of angles.

        Returns:
            float: Log-likelihood value.
        """
        x1, x2 = X
        x = np.sqrt(x1 ** 2 + x2 ** 2)
        return self._n * np.log(i0(x)) - np.sum(x1 * np.cos(self._X) + x2 * np.sin(self._X))

    def _dldx1(self, X):
        """
        Partial derivative of the log-likelihood function with respect to x1.

        Args:
            X (numpy.ndarray): Array of angles.

        Returns:
            float: Partial derivative value.
        """
        x1, x2 = X
        x = np.sqrt(x1 ** 2 + x2 ** 2)
        return -self._n * (self._c - self.m(x) * x1 / x)

    def _dldx2(self, X):
        """
        Partial derivative of the log-likelihood function with respect to x2.

        Args:
            X (numpy.ndarray): Array of angles.

        Returns:
            float: Partial derivative value.
        """
        x1, x2 = X
        x = np.sqrt(x1 ** 2 + x2 ** 2)
        return -self._n * (self._s - self.m(x) * x2 / x)


class SymmetrisedVonMises(VonMises):
    """
    Class representing a symmetrised Von Mises distribution.

    Inherits from VonMises.

    Methods:
        pdf(x): Probability density function.
        _loglikelihood(X): Log-likelihood function.
        _dldx1(X): Partial derivative of the log-likelihood function with respect to x1.
        _dldx2(X): Partial derivative of the log-likelihood function with respect to x2.
    """

    def __init__(self, X):
        """
        Initialize the SymmetrisedVonMises object.

        Args:
            X (numpy.ndarray): Array of angles in degrees.
        """
        super().__init__(X)

    def pdf(self, x):
        """
        Probability density function.

        Args:
            x (float): Value at which to evaluate the PDF.

        Returns:
            float: PDF value.
        """
        return 1 / (2 * np.pi * i0(x)) * np.exp(self._kappa * np.cos(x - self._alpha)) + self._kappa * np.cos(
            x + self._alpha)

    def _loglikelihood(self, X):
        """
        Log-likelihood function.

        Args:
            X (numpy.ndarray): Array of angles.

        Returns:
            float: Log-likelihood value.
        """
        x1, x2 = X
        x = np.sqrt(x1 ** 2 + x2 ** 2)
        return self._n * np.log(i0(x)) - np.sum(x1 * np.cos(self._X) + np.log(np.cosh(x2 * np.sin(self._X))))

    def _dldx1(self, X):
        """
        Partial derivative of the log-likelihood function with respect to x1.

        Args:
            X (numpy.ndarray): Array of angles.

        Returns:
            float: Partial derivative value.
        """
        x1, x2 = X
        x = np.sqrt(x1 ** 2 + x2 ** 2)
        return -self._n * self._c + self.m(x) * self._n * x1 / x

    def _dldx2(self, X):
        """
        Partial derivative of the log-likelihood function with respect to x2.

        Args:
            X (numpy.ndarray): Array of angles.

        Returns:
            float: Partial derivative value.
        """
        x1, x2 = X
        x = np.sqrt(x1 ** 2 + x2 ** 2)
        return -np.sum(np.tanh(x2 * np.sin(self._X)) * np.sin(self._X)) + self.m(x) * self._n * x2 / x


def calculate_stats(X):
    """
    Calculate statistics of the given data.

    Args:
        X (numpy.ndarray): Array of angles in degrees.

    Returns:
        tuple: Mean direction and standard deviation.
    """
    von_mises = None
    if np.mean(X) > 150:
        von_mises = SymmetrisedVonMises(X)
    else:
        von_mises = UnsymmetrisedVonMises(X)
    von_mises.fit()
    return np.clip(np.abs(von_mises.mu), 0, 180), von_mises.sigma
