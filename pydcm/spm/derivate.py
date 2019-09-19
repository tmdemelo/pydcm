from ..globs import *
from ..utils import (
    dense,
    diags,
    expv,
    isempty,
    isnumericarray,
    isobjectarray,
    isvector,
    maxnorm,
    maxshape,
    norm1,
    sptrace
)
from .. import spm
from .. import wrappers

__all__ = ['spm_dfdx', 'spm_dfdx_cat', 'spm_diff']


# used for cell subtraction in matlab
def spm_dfdx(f, f0, dx):
    if isinstance(f, (tuple, list)) or isobjectarray(f):
        dfdx = copy.copy(f)
        for i in range(size(f)):
            dfdx[i] = spm.dfdx(f[i], f0[i], dx)
    # elif matlab struct
    else:
        dfdx = (f - f0) / dx
    return dfdx


def spm_dfdx_cat(J):
    # concatenate into a matrix
    J0 = J.ravel()[0]
    if isvector(J0):
        # assume 1d arrays are column vectors
        if J0.ndim == 1 or (J0.ndim > 1 and J0.shape[1] == 1):
            return spm.cat(J)
        else:
            return spm.cat(J.T).T
    else:
        return J


def spm_diff(*args):
    """
    matrix high-order numerical differentiation
    FORMAT [dfdx] = spm_diff(f,x,...,n)
    FORMAT [dfdx] = spm_diff(f,x,...,n,V)
    FORMAT [dfdx] = spm_diff(f,x,...,n,'q')

    f      - [inline] function f(x{1},...)
    x      - input argument[s]
    n      - arguments to differentiate w.r.t.

    V      - cell array of matrices that allow for differentiation w.r.t.
    to a linear transformation of the parameters: i.e., returns

    df/dy{i};    x = V{i}y{i};    V = dx(i)/dy(i)

    q      - (char) flag to preclude default concatenation of dfdx

    dfdx          - df/dx{i}                     ; n =  i
    dfdx{p}...{q} - df/dx{i}dx{j}(q)...dx{k}(p)  ; n = [i j ... k]


    This routine has the same functionality as spm_ddiff, however it
    uses one sample point to approximate gradients with numerical (finite)
    differences:

    dfdx  = (f(x + dx)- f(x))/dx
    _________________________________________________________________________
    Copyright (C) 2008 Wellcome Trust Centre for Neuroimaging

    Karl Friston
    $Id: spm_diff.m 7143 2017-07-29 18:50:38Z karl $
    """

    # IMPORTANT!
    # n are array indexes
    # from matlab to python, n = n - 1

    # step size for numerical derivatives
    # --------------------------------------------------------------------------
    # TODO: check for GLOBAL_DX
    dx = exp(-8)

    f = spm.funcheck(args[0])

    # parse input arguments
    # --------------------------------------------------------------------------

    # matlab cell like
    if isinstance(args[-1], (list, tuple)) or isobjectarray(args[-1]):
        x = array(args[1:-2])
        n = atleast_1d(array(args[-2]))
        V = args[-1]
        # increase list V beforehand
        # matlab version does it dynamically
        if len(V) < len(x):
            V = V + ([None] * (len(x) - len(V)))
        q = 1
    # numeric
    elif isinstance(args[-1], (int, float)) or isnumericarray(args[-1]):
        x = array(args[1:-1])
        n = atleast_1d(array(args[-1]))
        V = [None] * size(x)
        q = 1
    # characters
    elif isinstance(args[-1], str):
        x = array(args[1:-2])
        n = array(args[-2])
        V = [None] * size(x)
        q = 0
    else:
        raise Exception('improper call')

    # check transform matrices V = dxdy
    # --------------------------------------------------------------------------
    for i in range(maxshape(x)):
        # no checking if V{i} exists, unlike matlab version
        # V was already extended to len(x) above
        if V[i] is None and np.any(n == i):
            V[i] = wrappers.eye_fn(spm.length(x[i]))  # matlab: speye

    # initialise
    # --------------------------------------------------------------------------
    m = n[-1]
    xm = spm.vec(x[m])
    vmlen = atleast_2d(V[m]).shape[1]
    J = full((1, vmlen), None)  # matlab: cell

    # proceed to derivatives
    # ==========================================================================

    if size(n) == 1:
        # dfdx
        # ----------------------------------------------------------------------
        f0 = f(*x)  # f(x[:])
        for i in range(size(J)):
            xi = copy.copy(x)
            xi[m] = spm.unvec(xm + V[m][:, [i]] * dx, x[m])
            J[0][i] = spm.dfdx(f(*xi), f0, dx)  # f(xi{:}), ...
            # J[i] = spm.dfdx(f(*xi), f0, dx)  # f(xi{:}), ...

        # return numeric array for first-order derivatives
        # ======================================================================

        # vectorise f
        # ----------------------------------------------------------------------
        f = spm.vec(f0)

        # if there are no arguments to differentiate w.r.t. ...
        # ----------------------------------------------------------------------
        if isempty(xm):  # isempty(xm)
            J = wrappers.zeros_fn((maxshape(f), 0))  # sparse(length(f),0)

        # or there are no arguments to differentiate
        # ----------------------------------------------------------------------
        elif isempty(f):  # isempty(f)
            J = wrappers.zeros_fn((0, maxshape(xm)))  # sparse(0,length(xm))

        # differentiation of a scalar or vector
        # ----------------------------------------------------------------------
        if isnumericarray(f0) and isobjectarray(J) and q:
            J = spm.dfdx_cat(J)

        if isobjectarray(J):
            J = J[0].tolist()

        result = [J, f0]
    else:
        # dfdxdxdx....
        # ----------------------------------------------------------------------
        f0 = full((1, size(n)), None)
        f0.ravel()[:] = spm.diff(f, *x, n[0:-1], V)
        p = True

        for i in range(size(J)):
            xi = copy.copy(x)
            xmi = xm + V[m][:, [i]] * dx
            xi[m] = spm.unvec(xmi, x[m])
            fi = spm.diff(f, *xi, n[0:-1], V)[0]  # spm_diff returns two outputs
            J[0][i] = spm.dfdx(fi, f0.ravel()[0], dx)
            p = p and isnumericarray(J[0][i])

        if p and q:
            J = spm.dfdx_cat(J)

        if isobjectarray(J):
            J = J[0].tolist()

        result = [J, *f0.ravel()]  # matlab: [{J} f0]

    return result

