import pydcm
from ..globs import *
from .. import spm

def gx_all_fmri(x, u, P, M=None):
    """
    Simulated BOLD response and copied state vector
    FORMAT [y] = spm_gx_state_fmri(x,u,P,M)
    y          - BOLD response and copied state vector

    x          - state vector     (see spm_fx_fmri)
    P          - Parameter vector (see spm_fx_fmri)
    M          - model specification structure (see spm_nlsi)

    Return the BOLD response and all hidden state vectors.
    """
    y = np.atleast_2d(spm.gx_fmri(x, u, P, M)).T  # bold
    x = np.atleast_2d(x)  # state vector
    y = np.vstack((y,
                   x[:, [0]], # neuronal state
                   x[:, [1]], # signal
                   x[:, [2]], # flow
                   x[:, [3]], # volume
                   x[:, [4]], # dHb
                  ))
    return y.T


