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

__all__ = ['spm_diag', 'spm_expm', 'spm_inv', 'spm_logdet',
            'spm_pinv', 'spm_speye', 'spm_svd', 'spm_trace']


def spm_diag(v, k=0):
    """
    Diagonal matrices and diagonals of a matrix

    Like numpy.diag, but with special handling of lists and object arrays.

    Based on numpy.diag code, but if output has dtype=object,
    fill it with None instead of zeros

    Attempts to emulate spm_diag behavior
    """
    if isinstance(v, (list, tuple)):
        # needed because numpy.diag gives error with lists
        # where each element has the same size
        # eg: np.diag([eye(2), eye(2)]) raises error
        #     np.diag([eye(2), eye(3)]) works
        vv = np.empty(len(v), dtype='O')
        vv[:] = v
        v = vv
    if isobjectarray(v):
        pass
    else:
        v = np.asanyarray(v)
    s = v.shape
    if len(s) == 1:
        n = s[0]+abs(k)
        if v.dtype == object:
            res = full((n,n), None)
        else:
            res = zeros((n, n), v.dtype)
        if k >= 0:
            i = k
        else:
            i = (-k) * n
        res[:n-k].flat[i::n+1] = v
        return res
    elif len(s) == 2:
        return diagonal(v, k)
    else:
        raise ValueError("Input must be 1- or 2-d.")



def spm_expm(J, x=None):
    """
    approximate matrix exponential using a Taylor expansion
    FORMAT [y] = spm_expm(J,x)
    FORMAT [y] = spm_expm(J)
    y          = expm(J)*x:
    y          = expm(J);

    This routine covers and extends expm  functionality  by  using  a
    comoutationally  expedient  approximation  that can handle sparse
    matrices when dealing with the special case of expm(J)*x, where x
    is a vector, in an efficient fashion
    _________________________________________________________________________
    Copyright (C) 2008 Wellcome Trust Centre for Neuroimaging

    Karl Friston
    $Id: spm_expm.m 5691 2013-10-11 16:53:00Z karl $
    """

    # expm(J) use Pade approximation
    # ======================================================================

    # ensure norm is < 1/2 by scaling by power of 2
    # ----------------------------------------------------------------------
    I = wrappers.eye_fn(J.shape[0])  # matlab: speye
    e = frexp(maxnorm(J))[1]
    s = max(0, e + 1)
    J = J / 2**s
    X = J
    c = 1 / 2
    E = I + c * J
    D = I - c * J
    q = 6
    p = True
    for k in range(2, q + 1):
        c = c * (q - k + 1) / (k * (2 * q - k + 1))
        X = J @ X
        cX = c * X
        E = E + cX
        if p:
            D = D + cX
        else:
            D = D - cX
        p = not p

    # E = inv(D)*E
    # --------------------------------------------------------------------------
    E = solve(D, E)

    # Undo scaling by repeated squaring E = E^(2^s)
    # --------------------------------------------------------------------------
    for k in range(s):
        E = E @ E

    # Multiply by x if necessary
    # --------------------------------------------------------------------------
    if x is None:
        return E
    else:
        return E @ x


def spm_inv(A, TOL=None):
    """
    inverse for ill-conditioned matrices
    FORMAT X = spm_inv(A,TOL)

    A   - matrix
    X   - inverse

    TOL - tolerance: default = max(eps(norm(A,'inf'))*max(m,n),exp(-32))

    This routine simply adds a small diagonal matrix to A and calls inv.m
    _________________________________________________________________________
    Copyright (C) 2008 Wellcome Trust Centre for Neuroimaging

    Karl Friston
    $Id: spm_inv.m 7143 2017-07-29 18:50:38Z karl $
    """
    # check A
    # --------------------------------------------------------------------------
    m, n = shape(atleast_2d(A))
    if A is None:
        X = zeros(n, m)
        return X

    # tolerance
    # --------------------------------------------------------------------------
    if TOL is None:
        TOL = max(spacing(maxnorm(A)) * max(m, n), exp(-32))

    # inverse
    # matlab: X = inv(A + speye(m,n)*TOL)
    X = LA.inv(A + wrappers.eye_fn(m, n) * TOL)

    return X


def spm_logdet(C):
    """
    Compute the log of the determinant of positive (semi-)definite matrix C
    FORMAT H = spm_logdet(C)
    H = log(det(C))

    spm_logdet is a computationally efficient operator that can deal with
    full or sparse matrices. For non-positive definite cases, the determinant
    is considered to be the product of the positive singular values.
    _________________________________________________________________________
    Copyright (C) 2008-2013 Wellcome Trust Centre for Neuroimaging

    Karl Friston and Ged Ridgway
    $Id: spm_logdet.m 6321 2015-01-28 14:40:44Z karl $

    Note that whether sparse or full, rank deficient cases are handled in the
    same way as in spm_logdet revision 4068, using svd on a full version of C
    """

    # remove null variances
    # --------------------------------------------------------------------------
    i = nonzero(diagonal(C))[0]
    C = C[ix_(i, i)]
    i, j = nonzero(C)
    s = C[i, j]
    if np.any(isnan(s)):
        H = nan
        return H

    # TOL = max(size(C)) * eps(max(s)); % as in MATLAB's rank function
    # --------------------------------------------------------------------------
    TOL = 1e-16

    if np.any(i != j):

        # assymetric matrix
        # ------------------------------------------------------------------
        if maxnorm(spm.vec(C - C.T)) > TOL:
            s = LA.svd(dense(C), compute_uv=False)  # matlab: s = svd(full(C))

        else:
            # TODO? handle sparse matrix case

            # non-diagonal full matrix
            # --------------------------------------------------------------
            try:
                R = cholesky(C)
                H = 2 * np.sum(log(np.diag(R)))
            except np.linalg.LinAlgError:
                s = LA.svd(C, compute_uv=False)

    # if still here, singular values in s (diagonal values as a special case)
    # --------------------------------------------------------------------------
    H = np.sum(log(s[(s > TOL) & (s < 1 / TOL)]))
    return H


def spm_pinv(A, TOL=None):
    """
    pseudo-inverse for sparse matrices
    FORMAT X = spm_pinv(A,TOL)

    A   - matrix
    TOL - Tolerance to force singular value decomposition
    X   - generalised inverse
    _________________________________________________________________________
    Copyright (C) 2008 Wellcome Trust Centre for Neuroimaging

    Karl Friston
    $Id: spm_pinv.m 5877 2014-02-11 20:03:34Z karl $
    """
    # TODO?
    X = LA.pinv(A)
    return X


def spm_speye(m, n=None, k=0):
    """
    sparse leading diagonal matrix
    FORMAT [D] = spm_speye(m,n,k)

    returns an m x n matrix with ones along the k-th leading diagonal
    _________________________________________________________________________
    Copyright (C) 2008 Wellcome Trust Centre for Neuroimaging

    Karl Friston
    $Id: spm_speye.m 1131 2008-02-06 11:17:09Z spm $
    """
    if n is None:
        n = m
    k = atleast_1d(k)

    # leading diagonal matrix
    # --------------------------------------------------------------------------
    D = diags(ones((m, 1)), k, m, n)  # matlab: spdiags

    return D


def spm_svd(X, U=1e-6, *args, **kwargs):
    """
    Computationally efficient SVD (that can handle sparse arguments)
    FORMAT [U,S,V] = spm_svd(X,u)
    X    - (m x n) matrix
    u    - threshold (1 > u > 0) for normalized eigenvalues (default = 1e-6)
         - a value of zero induces u = 64*eps

    U    - {m x p} singular vectors
    V    - {m x p} singular variates
    S    - {p x p} singular values
    _________________________________________________________________________
    Copyright (C) 1994-2011 Wellcome Trust Centre for Neuroimaging

    Karl Friston
    $Id: spm_svd.m 6110 2014-07-21 09:36:13Z karl $
    """

    # default thresholds - preclude singular vectors with small singular values
    # --------------------------------------------------------------------------
    if U >= 1:
        U = U - 1e-6
    if U <= 0:
        U = 64 * np.spacing(1)

    # deal with sparse matrices
    # --------------------------------------------------------------------------
    # TODO: sparse, someday
    M, N = X.shape
    p = nonzero(np.any(X, axis=1))[0]
    q = nonzero(np.any(X, axis=0))[0]
    X = X[ix_(p, q)]

    # SVD
    # --------------------------------------------------------------------------
    i, j = np.nonzero(X)
    s = X[i, j]
    m, n = X.shape
    # TODO: check for bugs with potential 0 indices
    if np.any(i - j):  # matlab: any(i - j)

        # off-leading diagonal elements - full SVD
        # ----------------------------------------------------------------------
        # matlab: X = full(X) ... remove sparseness
        # TODO!!
        raise Exception('spm_svd: off-leading diagonal elements not implemented')
        # return LA.svd(X, *args, **kwargs)
    else:
        S = wrappers.zeros_fn((m, n))  # matlab: sparse(1:n, 1:n, s, m, n)
        S[(arange(n), arange(n))] = s  # why ix_ is not needed here?
        u = wrappers.eye_fn(m, n)  # matlab: speye
        v = wrappers.eye_fn(m, n)  # matlab: speye
        j = argsort(-s)  # matlab: sort(-s)
        i = -s[j]
        S = S[ix_(j, j)]  # but ix_ is needed here?
        v = v[:, j]
        u = u[:, j]
        s = S.diagonal()**2
        j = nonzero((s * max(shape(s) / np.sum(s))).ravel() > U)[0]
        v = v[:, j]
        u = u[:, j]
        S = S[ix_(j, j)]

    # replace in full matrices
    # --------------------------------------------------------------------------
    j = max(j.shape)
    U = wrappers.zeros_fn((M, j))  # matlab: sparse(...)
    V = wrappers.zeros_fn((N, j))  # matlab: sparse(...)
    if j:
        U[p, :] = u
        V[q, :] = v

    return U, S, V


def spm_trace(A, B):
    """
    fast trace for large matrices: C = spm_trace(A,B) = trace(A*B)
    FORMAT [C] = spm_trace(A,B)

    C = spm_trace(A,B) = trace(A*B) = sum(sum(A'.*B));
    _________________________________________________________________________
    Copyright (C) 2008 Wellcome Trust Centre for Neuroimaging

    Karl Friston
    $Id: spm_trace.m 4805 2012-07-26 13:16:18Z karl $

    fast trace for large matrices: C = spm_trace(A,B) = trace(A*B)
    -------------------------------------------------------------------------
    """
    C = np.sum(A.T * B)
    return C

