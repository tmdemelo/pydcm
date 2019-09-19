import pydcm
from ..globs import *
from ..utils import (
    isempty,
    isnumericarray,
    isnumericscalar,
    isobjectarray
)
from .. import wrappers
from .. import spm

__all__ = ['spm_cat', 'spm_cell_swap', 'spm_data_id', 'spm_funcheck',
           'spm_length', 'spm_vec', 'spm_unvec']

def spm_cat(x, d=None):
    """
    Convert a cell array into a matrix - a compiled routine
    FORMAT [x] = spm_cat(x,d)
    x - cell array
    d - dimension over which to concatenate [default - both]
   __________________________________________________________________________
    Empty array elements are replaced by sparse zero partitions and single 0
    entries are expanded to conform to the non-empty non zero elements.

    e.g.:
    > x       = spm_cat({eye(2) []; 0 [1 1; 1 1]})
    > full(x) =

        1     0     0     0
        0     1     0     0
        0     0     1     1
        0     0     1     1

    If called with a dimension argument, a cell array is returned.
   __________________________________________________________________________
    Copyright (C) 2005-2013 Wellcome Trust Centre for Neuroimaging

    Karl Friston
    $Id: spm_cat.m 5731 2013-11-04 18:11:44Z guillaume $
    """

    x = copy.deepcopy(x)
    # check x is not already a matrix
    # --------------------------------------------------------------------------
    if isobjectarray(x):
        x = atleast_2d(x)
        [n, m] = shape(x)
    elif isinstance(x, (list, tuple)):
        if isinstance(x[0], (list, tuple)) and len(x[0]) > 0:
            pass
        else:
            x = [x]
    else:
        return x

    # if concatenation over a specific dimension
    # --------------------------------------------------------------------------
    [n, m] = shape(x)[0:2]
    if d is not None:

        # concatenate over first dimension
        # ----------------------------------------------------------------------
        if d == 0:
            y = full((1, m), None)
            for i in range(m):
                y[i] = spm.cat(x[:], i)

        # concatenate over second
        # ----------------------------------------------------------------------
        elif d == 2:
            y = full((n, 1))
            for i in range(n):
                y[i] = spm.cat(x[i, :])

        # only viable for 2-D arrays
        # ----------------------------------------------------------------------
        else:
            raise Exception('unknown option')

        return y

    # find dimensions to fill in empty partitions
    # --------------------------------------------------------------------------
    I = zeros((n, m), dtype=int)
    J = zeros((n, m), dtype=int)
    for i in range(n):
        for j in range(m):
            xij = x[i][j]
            if isobjectarray(xij):
                xij = spm.cat(xij)
                x[i][j] = xij
            if xij is None  \
               or (isinstance(xij, (tuple, list)) and len(xij) == 0):
                u = v = 0
            elif np.isscalar(xij):
                u = v = 1
            elif xij.ndim == 1:
                xij = xij.reshape(-1, 1)
                x[i][j] = xij
                [u, v] = xij.shape
            else:
                [u, v] = xij.shape
            I[i, j] = u
            J[i, j] = v

    I = np.max(I, axis=1)
    J = np.max(J, axis=0)

    # sparse and empty partitions
    # --------------------------------------------------------------------------
    # [n, m] = shape(x)
    for i in range(n):
        for j in range(m):
            if isempty(x[i][j]) or x[i][j] is 0:
                x[i][j] = wrappers.zeros_fn((I[i], J[j]))  # matlab: sparse

    # concatenate
    y = full(n, None)
    for i in range(n):
        y[i] = concatenate(x[i][:], axis=1)

    x = concatenate(y)

    return x


# spm_cell_swap may return an object array where elements are
# 1x1 numeric arrays instead of scalars
def spm_cell_swap(x):
    """
    Swap columns for cells in matrix arrays
    FORMAT [y] = spm_cell_swap(x)
    y{:,i}(:,j) = x{:,j}(:,i);
    _________________________________________________________________________
    Copyright (C) 2008 Wellcome Trust Centre for Neuroimaging

    Karl Friston
    $Id: spm_cell_swap.m 5219 2013-01-29 17:07:07Z spm $
    """

    # return if empty
    # --------------------------------------------------------------------------
    if size(x) == 0:  # matlab: isempty(x)
        return []

    # swap columns for cells
    # --------------------------------------------------------------------------
    m, n = atleast_2d(x[0][0]).shape
    k, l = x.shape
    y = full((k, n), None)
    # matlab: [y{:}] = deal(zeros(m,l))
    for j in range(k):
        for i in range(n):
            y[j, i] = zeros((m, l))
    for r in range(k):
        for j in range(l):
            for i in range(n):
                y[r, i][:, j] = atleast_2d(x[r, j])[:, i]

    return y


def spm_data_id(*X):
    """
    generates a specific real number in a deterministic way
    from any data structure
    FORMAT ID = spm_data_id(X);
    X  - numeric, character, cell or stucture array[s]
    ID - specific ID
    _________________________________________________________________________
    Copyright (C) 2009 Wellcome Trust Centre for Neuroimaging

    Vladimir Litvak (based on Karl's spm_vec)
    $Id: spm_data_id.m 6712 2016-02-04 15:12:25Z peter $
    """

    if len(X) == 1:
        X = X[0]

    ID = 0

    if isinstance(X, str):
        pass
    elif isinstance(X, float):
        return X
    elif isinstance(X, int):
        return float(X)
    elif isnumericarray(X):
        Y = X.astype(float)
        ID = np.sum(np.abs(Y[np.isfinite(Y)]))
    elif isinstance(X, dict) or hasattr(X, '__dict__'):
        if hasattr(X, '__dict__'):
            X = X.__dict__
        for key in sorted(X.keys()):
            ID = ID + spm.data_id(X[key])
    elif isinstance(X, (tuple, list)):
        for item in X:
            ID = ID + spm.data_id(item)
    elif isobjectarray(X):
        for item in np.nditer(X):
            ID = ID + spm.data_id(item)

    if ID > 0:
        ID = 10 ** - (floor(log10(ID)) - 2) * ID

    return ID


def spm_funcheck(f):
    """
    Convert strings and inline objects to function handles
    FORMAT [h] = spm_funcheck(f)

    f   - filename, character expression or inline function
    h   - corresponding function handle
    _________________________________________________________________________
    Copyright (C) 2013 Wellcome Trust Centre for Neuroimaging

    Karl Friston
    $Id: spm_funcheck.m 6481 2015-06-16 17:01:47Z karl $
    """
    h = None

    # -Create function handle
    # ==========================================================================

    # if f is already a function handle
    # --------------------------------------------------------------------------
    if callable(f):
        h = f

    # if f is filename or expression
    # --------------------------------------------------------------------------
    elif isinstance(f, str):
        h = eval(f)

    # if f is an inline object
    # --------------------------------------------------------------------------
    #
    # TODO...?

    return h


def spm_length(X):
    """
    Length of a vectorised numeric, cell or structure array
    FORMAT [n] = spm_length(X)
    X    - numeric, cell or stucture array[s]
    n    - length(spm_vec(X))

    See spm_vec, spm_unvec
    _________________________________________________________________________

    e.g.:
    spm_length({eye(2) 3}) = 5
    _________________________________________________________________________
    Copyright (C) 2014 Wellcome Trust Centre for Neuroimaging

    Karl Friston
    $Id: spm_length.m 6233 2014-10-12 09:43:50Z karl $
    """
    return spm.vec(X).size


def spm_unvec(vX, *args):
    """
    Unvectorise a vectorised array - a compiled routine
    FORMAT [varargout] = spm_unvec(vX,varargin)
    varargin  - numeric, cell or structure array
    vX        - spm_vec(X)

    i.e. X           = spm_unvec(spm_vec(X),X)
     [X1,X2,...] = spm_unvec(spm_vec(X1,X2,...),X1,X2,...)

    See spm_vec
    _________________________________________________________________________
    Copyright (C) 2005-2014 Wellcome Trust Centre for Neuroimaging

    Karl Friston
    $Id: spm_unvec.m 6238 2014-10-13 09:38:23Z karl $


    error('spm_unvec.c not compiled - see Makefile')
    """

    # deal to multiple outputs if necessary... Not
    # --------------------------------------------------------------------------
    # python can just unpack output as needed

    if len(args) == 1:
        X = copy.copy(args[0])
    else:
        X = copy.copy(args)

    # check vX is numeric (...numpy array)
    # --------------------------------------------------------------------------
    if not isnumericarray(vX):
        vX = spm.vec(vX)

    # reshape numerical arrays
    # --------------------------------------------------------------------------
    if isnumericarray(X):
        # forcefully convert to float, to be in the safe side
        X = X.astype(float, copy=False)
        # no need to check for X dimensions
        X[:] = vX.reshape(X.shape, order='F')
        return X

    if isnumericscalar(X):
        return vX[0]

    # fill in object arrays (equivalent to matlab cells)
    if isobjectarray(X):
        for i in range(size(X)):
            if isinstance(X[i], np.ndarray):
                n = X[i].size
            else:
                n = spm.length(X[i])
            X[i] = spm.unvec(vX[0:n], X[i])
            vX = vX[n:]
        return X

    # fill in lists
    # --------------------------------------------------------------------------
    if isinstance(X, (list, tuple)):
        X = list(X)
        for i in range(len(X)):
            if isinstance(X[i], np.ndarray):
                n = X[i].size
            else:
                n = spm.length(X[i])
            X[i] = spm.unvec(vX[0:n], X[i])
            vX = vX[n:]
        return X

    # fill in generic objects
    # equivalent to matlab structs?
    if hasattr(X, '__dict__'):
        for k in sorted(X.__dict__.keys()):
            c = copy.copy(X.__dict__[k])
            if isinstance(c, np.ndarray):
                n = size(c)
            else:
                n = spm.length(c)
            c = spm.unvec(vX[0:n], c)
            # deal...?
            setattr(X, k, c)
            vX = vX[n:]
        return X

    # fill in dictionaries
    # redundant code is redundant
    if isinstance(X, dict):
        for k in sorted(X.keys()):
            c = copy.copy(X[k])
            if isinstance(c, np.ndarray):
                n = size(c)
            else:
                n = spm.length(c)
            c = spm.unvec(vX[0:n], c)
            # deal...?
            X[k] = c
            vX = vX[n:]
        return X

    # else
    # --------------------------------------------------------------------------
    # X         = empty(0)
    return None


def spm_vec(X, *args):
    """
    Vectorise a numeric, cell or structure array - a compiled routine
    FORMAT [vX] = spm_vec(X)
    X  - numeric, cell or stucture array[s]
    vX - vec(X)

    See spm_unvec
    _________________________________________________________________________

    e.g.:
    spm_vec({eye(2) 3}) = [1 0 0 1 3]'
    _________________________________________________________________________
    Copyright (C) 2005-2013 Wellcome Trust Centre for Neuroimaging

    Karl Friston
    $Id: spm_vec.m 6110 2014-07-21 09:36:13Z karl $


    error('spm_vec.c not compiled - see Makefile')
    """

    # initialise X and vX
    # --------------------------------------------------------------------------
    vX = empty(shape=(0, 1))
    if len(args) > 0:
        X = [X, args]

    # vectorize numeric arrays
    if isnumericarray(X):
        # using order='F' is an order of magnitude slower
        # but it makes things equivalent to matlab
        vX = atleast_2d(X.ravel(order='F')).T
        # to use the following, for performance, must makes changes in many
        # other equations
        # vX = atleast_2d(X.ravel()).T

    # vectorize numbers
    elif isnumericscalar(X):
        vX = atleast_2d(array(X))

    # vectorize dictionaries
    elif isinstance(X, dict):
        for k in sorted(X.keys()):
            vX = concatenate((vX, spm.vec(X[k])))

    # vectorize generic objects
    # TODO: handle objects with __slots__
    elif hasattr(X, '__dict__'):
        for k in sorted(X.__dict__.keys()):
            vX = concatenate((vX, spm.vec(X.__dict__[k])))

    # vectorise tuples and lists into numpy arrays
    # --------------------------------------------------------------------------
    elif isinstance(X, (list, tuple)):
        for i in X:
            vX = concatenate((vX, spm.vec(i)))

    # vectorize object arrays
    elif isobjectarray(X):
        for item in np.nditer(X.ravel(order='F'), flags=('refs_ok',)):
            vX = concatenate((vX, spm.vec(item.item())))

    return vX
