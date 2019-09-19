import numpy as np
import scipy as sp
import scipy.stats

__all__ = ['spm_Ncdf', 'spm_phi']


# Just adapt scipy norm.cdf
def spm_Ncdf(x, u=0, V=1):
    return sp.stats.norm.cdf(x, u, np.sqrt(V))


def spm_phi(x):
    """
    logistic function
    y = 1 / (1 + exp(-x))
    """
    return 1 / (1 + np.exp(-x))

