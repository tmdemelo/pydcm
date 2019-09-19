# -*- coding: utf-8 -*-

# expv function from the Quantum Information Toolkit project
# v0.11.0, commit c06a73
# http://qit.sourceforge.net
# Ville Bergholm 2008-2012

from __future__ import division, absolute_import, print_function, unicode_literals

import numpy as np
import scipy as sp
import scipy.sparse
from numpy import (
    array,
    ceil,
    dtype,
    empty,
    equal,
    sqrt,
    exp,
    floor,
    dot,
    inf,
    isscalar,
    log10,
    maximum,
    minimum,
    nonzero,
    pi,
    vdot,
    zeros,
)
from scipy.linalg import (
    norm
)
from scipy import linalg as sLA


def expv(t, A, v, tol=1.0e-7, m=30, iteration='arnoldi'):
    r"""Multiply a vector by an exponentiated matrix.

    Approximates :math:`exp(t A) v` using a Krylov subspace technique.
    Efficient for large sparse matrices.
    The basis for the Krylov subspace is constructed using either Arnoldi or Lanczos iteration.

    Input:
    t           vector of nondecreasing time instances >= 0
    A           n*n matrix (usually sparse) (as an (n,n)-shaped ndarray)
    v           n-dimensional vector (as an (n,)-shaped ndarray)
    tol         tolerance
    m           Krylov subspace dimension, <= n
    iteration   'arnoldi' or 'lanczos'. Lanczos is faster but requires a Hermitian A.

    Output:
    W       result matrix, :math:`W[i,:] \approx \exp(t[i] A) v`
    error   total truncation error estimate
    hump    :math:`\max_{s \in [0, t]}  \| \exp(s A) \|`

    Uses the sparse algorithm from :cite:`EXPOKIT`.
    """
    # Ville Bergholm 2009-2012

    # just in case somebody tries to use numpy.matrix instances here
    if isinstance(A, np.matrix) or isinstance(v, np.matrix):
        raise ValueError("A and v must be plain numpy.ndarray instances, not numpy.matrix.")

    n = A.shape[0]
    m = min(n, m)  # Krylov space dimension, m <= n

    if isscalar(t):
        tt = array([t])
    else:
        tt = t

    # take max to avoid division by 0, when input is all zeros
    a_norm = max(norm(A, inf), np.spacing(1))
    v_norm = max(norm(v), np.spacing(1))

    happy_tol = 1.0e-7  # "happy breakdown" tolerance
    min_error = a_norm * np.finfo(float).eps  # due to roundoff

    # step size control
    max_stepsize_changes = 10
    # safety factors
    gamma = 0.9
    delta = 1.2
    # initial stepsize
    fact = sqrt(2 * pi * (m + 1)) * ((m + 1) / exp(1)) ** (m + 1)

    def ceil_at_nsd(x, n=2):
        temp = 10 ** (floor(log10(x)) - n + 1)
        return ceil(x / temp) * temp

    def update_stepsize(step, err_loc, r):
        step *= gamma * (tol * step / err_loc) ** (1 / r)
        return ceil_at_nsd(step, 2)

    dt = dtype(complex)
    # TODO don't use complex matrices unless we have to: dt = result_type(t, A, v)

    # TODO shortcuts for Hessenberg matrix exponentiation?
    H = zeros((m + 2, m + 2), dt)  # upper Hessenberg matrix for the Arnoldi process + two extra rows/columns for the error estimate trick
    H[m + 1, m] = 1           # never overwritten!
    V = zeros((n, m + 1), dt)   # orthonormal basis for the Krylov subspace + one extra vector

    W = empty((len(tt), len(v)), dt)  # results
    t = 0  # current time
    beta = v_norm
    error = 0  # error estimate
    hump = [[v_norm, t]]
    # v_norm_max = v_norm  # for estimating the hump

    def iterate_lanczos(v, beta):
        """Lanczos iteration, for Hermitian matrices.
        Produces a tridiagonal H, cheaper than Arnoldi.

        Returns the number of basis vectors generated, and a boolean indicating a happy breakdown.
        NOTE that the we _must_not_ change global variables other than V and H here
        """
        # beta_0 and alpha_m are not used in H, beta_m only in a single position for error control
        prev = 0
        for k in range(0, m):
            vk = (1 / beta) * v
            V[:, k] = vk  # store the now orthonormal basis vector
            # construct the next Krylov vector beta_{k+1} v_{k+1}
            v = dot(A, vk)
            H[k, k] = alpha = vdot(vk, v)
            v += -alpha * vk - beta * prev
            # beta_{k+1}
            beta = norm(v)
            if beta < happy_tol:  # "happy breakdown": iteration terminates, Krylov approximation is exact
                return k + 1, True
            if k == m - 1:
                # v_m and one beta_m for error control (alpha_m not used)
                H[m, m - 1] = beta
                V[:, m] = (1 / beta) * v
            else:
                H[k + 1, k] = H[k, k + 1] = beta
                prev = vk
        return m + 1, False

    def iterate_arnoldi(v, beta):
        """Arnoldi iteration, for generic matrices.
        Produces a Hessenberg-form H.
        """
        V[:, 0] = (1 / beta) * v  # the first basis vector v_0 is just v, normalized
        for j in range(1, m + 1):
            p = dot(A, V[:, j - 1])  # construct the Krylov vector v_j
            # orthogonalize it with the previous ones
            for i in range(j):
                H[i, j - 1] = vdot(V[:, i], p)
                p -= H[i, j - 1] * V[:, i]
            temp = norm(p)
            if temp < happy_tol:  # "happy breakdown": iteration terminates, Krylov approximation is exact
                return j, True
            # store the now orthonormal basis vector
            H[j, j - 1] = temp
            V[:, j] = (1 / temp) * p
        return m + 1, False  # one extra vector for error control

    # choose iteration type
    iteration = iteration.lower()
    if iteration == 'lanczos':
        iteration = iterate_lanczos  # only works for Hermitian matrices!
    elif iteration == 'arnoldi':
        iteration = iterate_arnoldi
    else:
        raise ValueError("Only 'arnoldi' and 'lanczos' iterations are supported.")

    # loop over the time instances (which must be in increasing order)
    for kk in range(len(tt)):
        t_end = tt[kk]
        # initial stepsize
        # TODO we should inherit the stepsize from the previous interval
        r = m
        t_step = (1 / a_norm) * ((fact * tol) / (4 * beta * a_norm)) ** (1 / r)
        t_step = ceil_at_nsd(t_step, 2)

        while t < t_end:
            t_step = min(t_end - t, t_step)  # step at most the remaining distance

            # Arnoldi/Lanczos iteration, (re)builds H and V
            j, happy = iteration(v, beta)
            # now V^\dagger A V = H  (just the first m vectors, or j if we had a happy breakdown!)
            # assert(norm(dot(dot(V[:, :m].conj().transpose(), A), V[:, :m]) -H[:m,:m]) < tol)

            # error control
            if happy:
                # "happy breakdown", using j Krylov basis vectors
                t_step = t_end - t  # step all the rest of the way
                F = sLA.expm(t_step * H[:j, :j])
                err_loc = happy_tol
                r = m
            else:
                # no happy breakdown, we need the error estimate (using all m+1 vectors)
                av_norm = norm(dot(A, V[:, m]))
                # find a reasonable step size
                for k in range(max_stepsize_changes + 1):
                    F = sLA.expm(t_step * H)
                    err1 = beta * abs(F[m, 0])
                    err2 = beta * abs(F[m + 1, 0]) * av_norm
                    if err1 > 10 * err2:  # quick convergence
                        err_loc = err2
                        r = m
                    elif err1 > err2:  # slow convergence
                        err_loc = (err2 * err1) / (err1 - err2)
                        r = m
                    else:  # asymptotic convergence
                        err_loc = err1
                        r = m - 1
                    # should we accept the step?
                    if err_loc <= delta * tol * t_step:
                        break
                    if k >= max_stepsize_changes:
                        raise RuntimeError('Requested tolerance cannot be achieved in {0} stepsize changes.'.format(max_stepsize_changes))
                    t_step = update_stepsize(t_step, err_loc, r)

            # step accepted, update v, beta, error, hump
            v = dot(V[:, :j], beta * F[:j, 0])
            beta = norm(v)
            error += max(err_loc, min_error)
            # v_norm_max = max(v_norm_max, beta)

            t += t_step
            t_step = update_stepsize(t_step, err_loc, r)
            hump.append([beta, t])

        W[kk, :] = v

    hump = array(hump) / v_norm
    return W, error, hump

# ===============================


numeric_typecodes = ''.join(set(np.typecodes['AllInteger']
                                + np.typecodes['AllFloat']
                                + np.typecodes['Complex']))

real_typecodes = ''.join(set(np.typecodes['AllInteger']
                             + np.typecodes['AllFloat']))

# based on numpy.ScalarTypes
numeric_scalar_types = (
    int,
    float,
    complex,
    int,
    bool,
    bytes,
    np.complex128,
    np.float64,
    np.uint32,
    np.int32,
    np.bytes_,
    np.complex64,
    np.float32,
    np.uint16,
    np.int16,
    np.bool_,
    # np.timedelta64,
    np.float16,
    np.uint8,
    np.int8,
    # np.datetime64,
    np.uint64,
    np.int64,
    np.void,
    np.complex256,
    np.float128,
    np.uint64,
    np.int64,
)


def sptrace(x):
    """
    Trace function that works with sparse arrays
    """
    return x.diagonal().sum()


def dense(x):
    if sp.sparse.issparse(x):
        return x.toarray()
    else:
        return x


def diags(v, d, m, n):
    """
    Dense matrix version of spdiags
    Based on octave implementation
    Unlike scipy spdiags, arg v has diagonals in column direction
    """
    j, i = nonzero(v)
    v = v[j, i]
    if m >= n:
        offset = maximum(minimum(d, n - m), 0)
    else:
        offset = d
    j = j + offset[i]
    i = j - d[i]
    idx = (i >= 0) & (i < m) & (j >= 0) & (j < n)
    B = zeros((m, n), dtype=v.dtype)
    B[i[idx], j[idx]] = v[idx]
    return B


def hasdict(x):
    if hasattr(x, '__dict__'):
        return True


def norm1(x):
    return np.max(np.sum(np.abs(x), axis=0))


def maxshape(x):
    if isinstance(x, np.ndarray):
        smin = min(x.shape)
        if smin == 0:
            # as matlab behavior
            return smin
        else:
            return max(x.shape)
    if isinstance(x, (tuple, list)):
        return len(x)
    if isinstance(x, (float, int)):
        return 1
    return 0


def maxnorm(x):
    """
    About same speed of scipy.linalg.norm(x, inf)
    Works with sparse arrays, unlike scipy version
    """
    if x.ndim == 1:
        return np.max(np.abs(x))
    else:
        return np.max(np.sum(np.abs(x), axis=1))


def isempty(x):
    if x is None:
        return True
    if isinstance(x, np.ndarray) and x.size == 0:
        return True
    if isinstance(x, (list, tuple)) and len(x) == 0:
        return True
    return False

def isrealarray(x):
    r = isinstance(x, np.ndarray) and x.dtype.kind in real_typecodes
    return r


def isnumericarray(x):
    r = isinstance(x, np.ndarray) and x.dtype.char in numeric_typecodes
    return r


def isnumericscalar(x):
    r = isinstance(x, numeric_scalar_types)
    return r


def isobjectarray(x):
    r = isinstance(x, np.ndarray) and x.dtype.kind == 'O'
    return r


def issize1(x):
    if isinstance(x, (int, float)):
        return True
    elif isinstance(x, np.ndarray) and x.size == 1:
        return True
    # note: matlab isscalar returns True for single characters
    # elif...
    else:
        return False


def isvector(x):
    r = False
    if isinstance(x, np.ndarray):
        r = np.sum(equal(x.shape, 1)) == (x.ndim - 1)
    elif isinstance(x, (float, int)):
        r = True
    elif isinstance(x, (tuple, list)):
        r = len(x) > 0
    return r
