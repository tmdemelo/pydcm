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

__all__ = ['spm_bireduce', 'spm_Ce', 'spm_dctmtx',
           'spm_detrend', 'spm_kernels', 'spm_Q' ]


def spm_bireduce(M, P, nout=2):
    """
    reduction of a fully nonlinear MIMO system to Bilinear form
    FORMAT [M0,M1,L1,L2] = spm_bireduce(M,P);

    M   - model specification structure
    Required fields:
      M.f   - dx/dt    = f(x,u,P,M)                 {function string or m-file}
      M.g   - y(t)     = g(x,u,P,M)                 {function string or m-file}
      M.bi  - bilinear form [M0,M1,L1,L2] = bi(M,P) {function string or m-file}
      M.m   - m inputs
      M.n   - n states
      M.l   - l outputs
      M.x   - (n x 1) = x(0) = expansion point: defaults to x = 0;
      M.u   - (m x 1) = u    = expansion point: defaults to u = 0;

      M.D   - delay operator df/dx -> D*df/dx [optional]

    P   - model parameters

    A Bilinear approximation is returned where the states are

           q(t) = [1; x(t) - x(0)]

    __________________________________________________________________________
    Returns Matrix operators for the Bilinear approximation to the MIMO
    system described by

          dx/dt = f(x,u,P)
           y(t) = g(x,u,P)

    evaluated at x(0) = x and u = 0

          dq/dt = M0*q + u(1)*M1{1}*q + u(2)*M1{2}*q + ....
           y(i) = L1(i,:)*q + q'*L2{i}*q/2;

    -------------------------------------------------------------------------
    Copyright (C) 2008 Wellcome Trust Centre for Neuroimaging

    Karl Friston
    $Id: spm_bireduce.m 6856 2016-08-10 17:55:05Z karl $
    """

    # set up
    # ==========================================================================

    # create inline functions
    # --------------------------------------------------------------------------
    # TODO? try catch?
    funx = spm.funcheck(M['f'])

    # expansion point
    # --------------------------------------------------------------------------
    x = spm.vec(M['x'])
    if 'u' in M:
        u = spm.vec(M['u'])
    else:
        u = zeros((M['m'], 1))  # sparse(M.m,1)

    # Partial derivatives for 1st order Bilinear operators
    # ==========================================================================

    # f(x(0),0) and derivatives
    # --------------------------------------------------------------------------

    # TODO
    # if all(isfield(M,{'dfdxu','dfdx','dfdu','f0'})):
    if False:
        dfdxu = M['dfdxu']
        dfdx = M['dfdx']
        dfdu = M['dfdu']
        f0 = M['f0']
    else:
        dfdxu, dfdx, *_ = spm.diff(funx, M['x'], u, P, M, array([0, 1]))
        dfdu, f0, *_ = spm.diff(funx, M['x'], u, P, M, 1)
    f0 = spm.vec(f0)
    m = len(dfdxu)  # m inputs # matlab: length
    n = max(f0.shape)  # n states  # matlab: length

    # delay operator
    # --------------------------------------------------------------------------
    # TODO
    # if 'D' in M and D is not None and D.ndim > ...

    # Bilinear operators
    # ==========================================================================

    # Bilinear operator - M0
    # --------------------------------------------------------------------------
    M0 = spm.cat(array([[0,               None],
                        [(f0 - dfdx @ x), dfdx]]))

    # Bilinear operator - M1 = dM0/du
    # --------------------------------------------------------------------------
    # M1 = full((m, 1), None)  # matlab: cell(m,1);
    M1 = [None] * m
    for i in range(m):
        M1[i] = spm.cat(
            array([[0,                                 None],
                   [(dfdu[:, [i]] - dfdxu[i] @ x), dfdxu[i]]]))

    if nout < 3:
        return M0, M1

    # Output operators
    # ==========================================================================

    # add observer if not specified
    # --------------------------------------------------------------------------
    if 'g' in M:
        fung = spm.funcheck(M['g'])  # matlab: fcnchk(M.g,'x','u','P','M')
    else:
        M['g'] = 'lambda x, u, P, M: spm.vec(x)'
        M['l'] = n
        fung = spm.funcheck(M['g'])  # matlab: fcnchk(M.g,'x','u','P','M')

    # g(x(0),0)
    # --------------------------------------------------------------------------
    dgdx, g0 = spm.diff(fung, M['x'], u, P, M, 0)
    g0 = spm.vec(g0)
    l = max(g0.shape)

    # Output matrices - L1
    # --------------------------------------------------------------------------
    # TODO: improve spm_cat so it works with a simple list like this:
    # [spm_vec(g0) - dgdx @ x, dgdx]
    L1 = full((1, 2), None)
    L1[0] = [spm.vec(g0) - dgdx @ x, dgdx]
    L1 = spm.cat(L1)

    if nout < 4:
        return M0, M1, L1

    # Output matrices - L2
    # --------------------------------------------------------------------------
    dgdxx, *_ = spm.diff(fung, M['x'], u, P, M, [0, 0], 'nocat')
    D = [None] * l  # full((l, n, dgdxx[0].shape[1])  # preallocate
    for i in range(l):
        D[i] = np.zeros((n, dgdxx[0][0].size))
        for j in range(n):
            D[i][j, :] = dgdxx[j][i, :]
    L2 = [None] * l
    for i in range(l):
        L2[i] = spm.cat(spm.diag([0, D[i]]))

    return M0, M1, L1, L2


# spm_Ce requires integer input array
def spm_Ce(t, v=None, a=None):
    """
    Error covariance constraints (for serially correlated data)
    FORMAT [C] = spm_Ce(v,a)
    FORMAT [C] = spm_Ce('ar',v,a)
    v  - (1 x n) v(i) = number of observations for i-th block
    a  - AR coefficient expansion point  [Default: a = []]

    a  = [] (default) - block diagonal identity matrices specified by v:

      C{i}  = blkdiag( zeros(v(1),v(1)),...,AR(0),...,zeros(v(end),v(end)))
      AR(0) = eye(v(i),v(i))

    otherwise:

      C{i}     = AR(a) - a*dAR(a)/da;
      C{i + 1} = AR(a) + a*dAR(a)/da;

    FORMAT [C] = spm_Ce('fast',v,tr)
    v  - (1 x n) v(i) = number of observations for i-th block
    tr - repetition time

    See also: spm_Q.m
    _________________________________________________________________________
    Copyright (C) 2000-2017 Wellcome Trust Centre for Neuroimaging

    Karl Friston
    $Id: spm_Ce.m 7203 2017-11-08 12:49:15Z guillaume $
    """

    # Defaults (and backward compatibility with spm_Ce(v,a) == spm_Ce('ar',v,a))
    # --------------------------------------------------------------------------
    if not isinstance(t, str):
        if v is not None:
            a = v
        v = t
        t = 'ar'

    if a is not None:
        a = atleast_1d(a)

    # Error covariance constraints
    # --------------------------------------------------------------------------
    if t == 'ar':

        # Create block diagonal components
        # ------------------------------------------------------------------
        C = []
        l = size(v)
        n = np.sum(v)
        k = 0
        if l > 1:
            for i in range(l):
                dCda = spm.Ce(v[i], a)
                for j in range(len(dCda)):
                    x, y = nonzero(dCda[j])
                    q = dCda[j][x, y]
                    x = x + k
                    y = y + k
                    # matlab: C{end + 1} = sparse(x,y,q,n,n);
                    C.append(wrappers.zeros_fn((n, n)))
                    C[-1][x, y] = q
                k = v[i] + k
        else:

            # dCda
            # --------------------------------------------------------------
            if a is not None:
                Q = spm.Q(a, v)
                dQda, *_ = spm.diff('spm.Q', a, v, 0)
                A = a
                if size(a) == 1:
                    # numpy doesn't allow matmul with scalar
                    A = eye(Q.shape[0]) * a
                else:
                    A = a
                C.append(Q - dQda.ravel()[0] @ A)
                C.append(Q + dQda.ravel()[0] @ A)
            else:
                # matlab: speye(v,v)
                C.append(wrappers.eye_fn(v))

    elif t == 'fast':
        dt = a
        C = []
        n = np.sum(v)
        k = 0
        for m in range(size(v)):
            T = arange(v[m]) * dt
            d = 2 ** arange(floor(log2(dt / 4)), 7)  # log2(64) = 6
            for i in range(min(6, size(d))):
                for j in range(3):
                    QQ = toeplitz((T ** j) * exp(-T / d[i]))
                    x, y = nonzero(QQ)
                    q = QQ[x, y]
                    x = x + k
                    y = y + k
                    # matlab: C{end + 1} = sparse(x,y,q,n,n);
                    C.append(wrappers.zeros_fn((n, n)))
                    C[-1][x, y] = q
            k = k + v[m]
    else:
        raise Exception('Unknown error covariance constraints.')

    return C


def spm_dctmtx(N, K=None, n=None, f=None):
    """
    Create basis functions for Discrete Cosine Transform
    FORMAT C = spm_dctmtx(N)
    FORMAT C = spm_dctmtx(N,K)
    FORMAT C = spm_dctmtx(N,K,n)
    FORMAT D = spm_dctmtx(N,K,'diff')
    FORMAT D = spm_dctmtx(N,K,n,'diff')

    N        - dimension
    K        - order
    n        - optional points to sample

    C        - DCT matrix or its derivative
    _________________________________________________________________________

    spm_dctmtx creates a matrix for the first few basis functions of a one
    dimensional discrete cosine transform.
    With the 'diff' argument, spm_dctmtx produces the derivatives of the DCT.

    Reference:
    Fundamentals of Digital Image Processing (p 150-154). Anil K. Jain, 1989.
    _________________________________________________________________________
    Copyright (C) 1996-2015 Wellcome Trust Centre for Neuroimaging

    John Ashburner
    $Id: spm_dctmtx.m 6416 2015-04-21 15:34:10Z guillaume $
    """

    d = 0

    if K is None:  # nargin == 1
        K = N

    if n is None:  # nargin < 3
        n = arange(0, N)  # matlab: (0:(N-1))'
    elif f is None:  # nargin == 3
        if n == 'diff':
            d = 1
            n = arange(0, N)  # matlab: (0:(N-1))'
        elif n == 'diff2':
            d = 2
            n = arange(0, N)  # matlab: (0:(N-1))'
        else:
            n = n.ravel(order='F')
    else:  # nargin == 4
        n = n.ravel(order='F')
        if f == 'diff':
            d = 1
        elif f == 'diff2':
            d = 2
        else:
            raise Exception('Incorrect Usage.')

    C = zeros((n.shape[0], K))
    if d == 0:
        C[:, 0] = ones((n.shape[0])) / sqrt(N)
        for k in range(2, K + 1):
            C[:, k - 1] = sqrt(2 / N) * cos(pi * (2 * n + 1) * (k - 1) / (2 * N))
    elif d == 1:
        for k in range(2, K + 1):
            C[:, k - 1] =  \
                -2 ** (1 / 2)  \
                * (1 / N) ** (1 / 2)  \
                * sin(1 / 2 * pi
                      * (2 * n * k - 2 * n + k - 1) / N)  \
                * pi * (k - 1) / N
    elif d == 2:
        for k in range(2, K + 1):
            C[:, k - 1] =  \
                -2 ** (1 / 2)  \
                * (1 / N) ** (1 / 2)  \
                * cos(1 / 2 * pi
                      * (2 * n + 1) * (k - 1) / N)  \
                * pi ** 2  \
                * (k - 1) ** 2 / N ** 2
    else:
        raise Exception('Incorrect usage.')

    return C


def spm_detrend(x, p=0):
    """
    Polynomial detrending over columns
    FORMAT y = spm_detrend(x,p)
    x   - data matrix
    p   - order of polynomial [default: 0]

    y   - detrended data matrix
    _________________________________________________________________________

    spm_detrend removes linear and nonlinear trends from column-wise data
    matrices.
    _________________________________________________________________________
    Copyright (C) 2008 Wellcome Trust Centre for Neuroimaging

    Karl Friston
    $Id: spm_detrend.m 7271 2018-03-04 13:11:54Z karl $
    """
    # Check for tuples, lists, object arrays...
    # -------------------------------------------------------------------------
    # TODO: check for object arrays?
    if isinstance(x, (list, tuple)) and (not isinstance(x, np.ndarray)):
        y = [None] * len(x)
        for i in range(len(x)):
            y[i] = spm.detrend(x[i], p)
        return y

    # defaults
    m, n = shape(atleast_2d(x))
    if (m == 0) or (n == 0):
        y = None
        return y

    if not p:
        y = x - ones((m, 1)) * mean(x, axis=0)
        return y

    # polynomial adjustment
    # --------------------------------------------------------------------------
    G = zeros((m, p + 1))
    for i in range(p + 1):
        d = arange(1, m + 1)**i
        G[:, i] = d.ravel()
    # matlab: x - G*(LA.pinv(full(G))*x)
    y = x - G * (LA.pinv(dense(G)) @ atleast_2d(x))
    return y


# [K0,K1,K2,H1] = spm_kernels(varargin)
def spm_kernels(*args, nout=1):
    """
    returns global Volterra kernels for a MIMO Bilinear system
    FORMAT [K0,K1,K2] = spm_kernels(M,P,N,dt)            - output kernels
    FORMAT [K0,K1,K2] = spm_kernels(M0,M1,N,dt)          - state  kernels
    FORMAT [K0,K1,K2] = spm_kernels(M0,M1,L1,N,dt)       - output kernels (1st)
    FORMAT [K0,K1,K2] = spm_kernels(M0,M1,L1,L2,N,dt)    - output kernels (2nd)

    M,P   - model structure and parameters;
            or its bilinear reduction:

    M0    - (n x n)     df(q(0),0)/dq                    - n states
    M1    - {m}(n x n)  d2f(q(0),0)/dqdu                 - m inputs
    L1    - (l x n)     dldq                             - l outputs
    L2    - {m}(n x n)  dl2dqq

    N     - kernel depth       {intervals}
    dt    - interval           {seconds}

    Volterra kernels:
    --------------------------------------------------------------------------
    K0    - (1 x l)             = K0(t)         = y(t)
    K1    - (N x l x m)         = K1i(t,s1)     = dy(t)/dui(t - s1)
    K2    - (N x N x l x m x m) = K2ij(t,s1,s2) = d2y(t)/dui(t - s1)duj(t - s2)

    __________________________________________________________________________
    Returns Volterra kernels for bilinear systems of the form

            dq/dt   = f(q,u) = M0*q + M1{1}*q*u1 + ... M1{m}*q*um
               y(i) = L1(i,:)*q + q'*L2{i}*q

    where q = [1 x(t)] are the states augmented with a constant term

    _________________________________________________________________________
    Copyright (C) 2008 Wellcome Trust Centre for Neuroimaging

    Karl Friston
    $Id: spm_kernels.m 6937 2016-11-20 12:30:40Z karl $
    """

    # assign inputs
    # --------------------------------------------------------------------------
    if len(args) == 4:
        M0 = args[0]
        M1 = args[1]
        N = args[2]
        dt = args[3]

    elif len(args) == 5:
        M0 = args[0]
        M1 = args[1]
        L1 = args[2]
        N = args[3]
        dt = args[4]

    elif len(args) == 6:
        M0 = args[0]
        M1 = args[1]
        L1 = args[2]
        L2 = args[3]
        N = args[4]
        dt = args[5]

    # bilinear reduction if necessary
    # --------------------------------------------------------------------------
    if hasattr(M0, '__dict__'):  # matlab: isstruct
        M0, M1, L1, L2 = spm.bireduce(M0, M1)

    # Volterra kernels for bilinear systems
    # ==========================================================================

    # make states the outputs (i.e. remove constant) if L1 is not specified
    # --------------------------------------------------------------------------
    try:
        L1
    except:
        L1 = wrappers.eye_fn(M0.shape)  # matlab: speye
        L1 = L1[1:, :]
    try:
        L2
    except:
        L2 = []

    # parameters
    # --------------------------------------------------------------------------
    N = int(N)  # kernel depth
    n = M0.shape[0]  # state variables
    m = len(M1)  # inputs
    l = L1.shape[0]  # outputs
    H1 = zeros((N, n, m))
    # K1 and K2 preallocation further down
    M0 = dense(M0)  # matlab: full(M0)

    # pre-compute matrix exponentials
    # --------------------------------------------------------------------------
    e1 = sLA.expm(dt * M0)
    e2 = sLA.expm(-dt * M0)
    M = full((N, m), None)  # preallocate
    for p in range(m):
        M[0, p] = e1 @ M1[p] @ e2
    ei = e1
    for i in range(1, N):
        ei = e1 * ei
        for p in range(m):
            M[i, p] = e1 @ M[i - 1, p] @ e2

    result = []

    # 0th order kernel
    # --------------------------------------------------------------------------
    if nout > 0:
        X0 = wrappers.zeros_fn((n, 1))  # matlab: sparse(1, 1, 1, n, 1)
        X0[0, 0] = 1
        H0 = ei @ X0
        K0 = L1 @ H0
        result.append(K0)

    # 1st order kernel
    # --------------------------------------------------------------------------
    if nout > 1:
        K1 = zeros((N, l, m))
        for p in range(m):
            for i in range(N):
                H1[i, :, p] = np.ravel(M[i, p] @ H0)
                K1[i, :, p] = np.ravel(H1[i, :, p]  @  L1.T)
        result.append(K1)

    # 2nd order kernels
    # --------------------------------------------------------------------------
    if nout > 2:
        K2 = zeros((N, N, l, m, m))
        for p in range(m):
            for q in range(m):
                for j in range(N):
                    H = L1 @ M[j, q] @ H1[j:N, :, p].T
                    K2[j, j:N, :, q, p] = H.T
                    K2[j:N, j, :, p, q] = H.T

        if L2 is None:  # matlab; isempty(L2)
            result.append(K2)
            return result

    # add output nonlinearity
    # ----------------------------------------------------------------------
        for i in range(m):
            for j in range(m):
                for p in range(l):
                    K2[:, :, p, i, j] = K2[:, :, p, i, j] + H1[:, :, i] @ L2[p] @ H1[:, :, j].T
        result.append(K2)

    return result


def spm_Q(a, n, q=False):
    """
    returns an (n x n) (inverse) autocorrelation matrix for an AR(p) process
    FORMAT [Q] = spm_Q(a,n,q)

    a  - vector of (p) AR coefficients
    n  - size of Q
    q  - switch to return inverse autocorrelation or precision [default q = 0]
    _________________________________________________________________________
    spm_Q uses a Yule-Walker device to compute K where:

    y = K*z

    such that y is an AR(p) process generated from an i.i.d innovation
    z.  This means

    cov(y) = <K*z*z'*K> = K*K'

    If called with q ~= 0, a first order process is assumed when evaluating
    the precision (inverse covariance) matrix; i.e., a = a(1)
    _________________________________________________________________________
    Copyright (C) 2008 Wellcome Trust Centre for Neuroimaging

    Karl Friston
    $Id: spm_Q.m 5838 2014-01-18 18:40:37Z karl $
    """

    if q:
        # compute P (precision)
        # ----------------------------------------------------------------------
        A = hstack((-a[0], 1 + a[0]**2, -a[0]))
        # matlab: spdiags(...)
        Q = diags(ones((n, 1)) * A, arange(-1, 2), n, n)
    else:
        p = size(a)
        A = hstack((1, -a.ravel()))  # matlab: [1 -a(:)']
        # matlab: spdiags(...)
        P = diags(ones((n, 1)) * A, - arange(p + 1), n, n)
        K = LA.inv(P)
        K = K * (np.abs(K) > 1e-4)
        Q = K @ K.T
        Q = toeplitz(Q[:, 0])
    return Q


