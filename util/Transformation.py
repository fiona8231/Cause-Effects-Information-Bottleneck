import numpy as np
import scipy.stats as stats
from scipy.stats import rankdata


class Transformation:
    """
    This class implements the copula transformation procedure.
    """

    @staticmethod
    def toUniform(A):
        """
        This methods maps a variable to uniform.
        :param A: the variable to be mapped
        :return: the transformed variable
        """

        r = np.apply_along_axis(rankdata, 0, A, method='average')
        u = (r) / (1 + r.shape[0])
        return u

    @staticmethod
    def UniformToOrig(A, B):
        """
        This method maps a transformed uniform variable back to its original.
        :param A: original variable
        :param B: transformed variable
        :return: transformed variable
        """
        B[B < 1e-6] = 1e-6
        B[B > 1 - 1e-6] = 1 - 1e-6
        p = B.copy()
        for c in range(A.shape[1]):
            p[:, c] = np.percentile(A[:, c], 100.0 * p[:, c], interpolation='linear')
        return p

    @staticmethod
    def to_beta(v, a, b):
        """
        This method transforms a variable to a beta transformed variable.
        :param v: the variable to be transformed
        :param a: beta shape parameter alpha
        :param b: beta shape parameter beta
        :return: the transformed variable
        """
        t = np.argsort(v)
        r = np.argsort(t).astype(np.float32)
        u = (r + 1) / (1 + r.shape[0])
        u = stats.beta.ppf(u, a, b)
        return u
