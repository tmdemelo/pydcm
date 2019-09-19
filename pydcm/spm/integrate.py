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

__all__ = ['spm_dx', 'spm_int']


def spm_dx(dfdx, f, t=inf):
    """
    returns dx(t) = (sLA.expm(dfdx*t) - I) * LA.inv(dfdx) * f
    FORMAT [dx] = spm_dx(dfdx,f,[t])
    dfdx   = df/dx
    f      = dx/dt
    t      = integration time: (default t = Inf);
             if t is a cell (i.e., {t}) then t is set to:
             exp(t - log(np.diag(-dfdx))

    dx     = x(t) - x(0)
    -------------------------------------------------------------------------
    Integration of a dynamic system using local linearization.  This scheme
    accommodates nonlinearities in the state equation by using a functional of
    f(x) = dx/dt.  This uses the equality

                expm([0   0     ]) = (sLA.expm(t * dfdx) - I) * LA.inv(dfdx)*f
                     [t*f t*dfdx]

    When t -> Inf this reduces to

                 dx(t) = -LA.inv(dfdx) * f

    These are the solutions to the gradient ascent ODE

               dx/dt   = k*f = k*dfdx*x =>

               dx(t)   = sLA.expm(t*k*dfdx)*x(0)
                       = sLA.expm(t * k * dfdx) * LA.inv(dfdx) * f(0) -
                         sLA.expm(0 * k * dfdx) * LA.inv(dfdx) * f(0)

    When f = dF/dx (and dfdx = dF/dxdx), dx represents the update from a
    Gauss-Newton ascent on F.  This can be regularised by specifying {t}
    A heavy regularization corresponds to t = -4 and a light
    regularization would be t = 4. This version of spm_dx uses an augmented
    system and the Pade approximation to compute requisite matrix
    exponentials

    references:

    Friston K, Mattout J, Trujillo-Barreto N, Ashburner J, Penny W. (2007).
    Variational free energy and the Laplace approximation. NeuroImage.
    34(1):220-34

    Ozaki T (1992) A bridge between nonlinear time-series models and
    nonlinear stochastic dynamical systems: A local linearization approach.
    Statistica Sin. 2:113-135.

    _________________________________________________________________________
    Copyright (C) 2008 Wellcome Trust Centre for Neuroimaging

    Karl Friston
    $Id: spm_dx.m 7144 2017-07-31 13:55:55Z karl $
    """

    # defaults
    # --------------------------------------------------------------------------
    nmax = 512  # threshold for numerical approximation
    xf = copy.copy(f)
    f = spm.vec(f)  # vectorise
    n = size(f)  # dimensionality

    # t is a regulariser
    # --------------------------------------------------------------------------
    # matlab: sw  = warning('off','MATLAB:log:logOfZero');
    if isinstance(t, (tuple, list)):

        # relative integration time
        # ----------------------------------------------------------------------
        # matlab: t = t{:}
        t = t[0]
        if size(t) == 1:  # scalar
            t = exp(t - spm.logdet(dfdx) / n)
        else:
            t = exp(t - log(np.diag(-dfdx)))
    # matlab: warning(sw)

    # use a [pseudo]inverse if all t > TOL
    # ==========================================================================
    if np.min(t) > exp(16):

        dx = -spm.pinv(dfdx) @ f

    else:

        # ensure t is a scalar or matrix
        # ----------------------------------------------------------------------
        # matlab: if isvector(t), ...
        if (not isinstance(t, (float, int))) and isvector(t):
            t = np.diag(t)

        # augment Jacobian and take matrix exponential
        # ======================================================================
        J = spm.cat(array([[0,     None],
                           [t * f, t * dfdx]]))

        # solve using matrix expectation
        # ----------------------------------------------------------------------
        if n <= nmax:
            dx = wrappers.expm_fn(J)
            dx = dx[:, 0]
        else:
            # matlab: sparse(1,1,1,n + 1,1)
            x = wrappers.zeros_fn(n + 1)
            x[0] = 1
            dx, *_ = expv(1, J, x)

        # recover update
        # ----------------------------------------------------------------------
        dx = dx.ravel()[1:]

    dx = spm.unvec(dx.real, xf)

    return dx


def spm_int(P, M, U):
    """
    integrates a MIMO bilinear system dx/dt = f(x,u) = A*x + B*x*u + Cu + D;
    FORMAT [y] = spm_int(P,M,U)
    P   - model parameters
    M   - model structure
      M.delays - sampling delays (s); a vector with a delay for each output

    U   - input structure or matrix

    y   - response y = g(x,u,P)
    _________________________________________________________________________
    Integrates the bilinear approximation to the MIMO system described by

       dx/dt = f(x,u,P) = A*x + u*B*x + C*u + D
       y     = g(x,u,P) = L*x;

    at v = M.ns is the number of samples [default v = size(U.u,1)]

    spm_int will also handle static observation models by evaluating
    g(x,u,P).  It will also handle timing delays if specified in M.delays

    -------------------------------------------------------------------------

    SPM solvers or integrators

    spm_int_ode:  uses ode45 (or ode113) which are one and multi-step solvers
    respectively.  They can be used for any ODEs, where the Jacobian is
    unknown or difficult to compute; however, they may be slow.

    spm_int_J: uses an explicit Jacobian-based update scheme that preserves
    nonlinearities in the ODE: dx = (expm(dt * J) - I) * inv(J) * f.  If the
    equations of motion return J = df/dx, it will be used; otherwise it is
    evaluated numerically, using spm_diff at each time point.  This scheme is
    infallible but potentially slow, if the Jacobian is not available (calls
    spm_dx).

    spm_int_E: As for spm_int_J but uses the eigensystem of J(x(0)) to eschew
    matrix exponentials and inversion during the integration. It is probably
    the best compromise, if the Jacobian is not available explicitly.

    spm_int_B: As for spm_int_J but uses a first-order approximation to J
    based on J(x(t)) = J(x(0)) + dJdx*x(t).

    spm_int_L: As for spm_int_B but uses J(x(0)).

    spm_int_U: like spm_int_J but only evaluates J when the input changes.
    This can be useful if input changes are sparse (e.g., boxcar functions).
    It is used primarily for integrating EEG models

    spm_int: Fast integrator that uses a bilinear approximation to the
    Jacobian evaluated using spm_bireduce. This routine will also allow for
    sparse sampling of the solution and delays in observing outputs. It is
    used primarily for integrating fMRI models (see also spm_int_D)
    _________________________________________________________________________
    Copyright (C) 2008 Wellcome Trust Centre for Neuroimaging

    Karl Friston
    $Id: spm_int.m 6856 2016-08-10 17:55:05Z karl $
    """

    # convert U to U['u'] if necessary
    # --------------------------------------------------------------------------
    if not isinstance(U, dict):
        u = {}
        u['u'] = U
        U = u
    if 'dt' in U:
        dt = U['dt']
    else:
        U['dt'] = 1
        dt = U['dt']

    # number of times to sample (v) and number of microtime bins (u)
    # --------------------------------------------------------------------------
    u = shape(U['u'])[0]
    if 'ns' in M:
        v = M['ns']
    else:
        v = u

    # get expansion point
    # --------------------------------------------------------------------------
    x = spm.vec(1, M['x'])  # matlab: [1; spm_vec(M.x)]

    # add [0] states if not specified
    # --------------------------------------------------------------------------
    try:
        M['f'] = spm.funcheck(M['f'])
    except:
        M['f'] = lambda x, u, P, M: zeros((0, 1))  # sparse(0,1)
        M['x'] = zeros((0, 0))  # sparse(0,0)

    # output nonlinearity, if specified
    # --------------------------------------------------------------------------
    try:
        g = spm.funcheck(M['g'])
    except:
        g = lambda x, u, P, M: x
        M['g'] = g

    # Bilinear approximation (1st order)
    # --------------------------------------------------------------------------
    M0, M1 = spm.bireduce(M, P)
    m = len(M1)  # m inputs  # matlab: length

    if 'delays' in M:
        D = maximum(np.round(M['delays'] / U['dt']), 1)
    else:
        D = ones((M['l'], 1)) @ np.round(u / v)

    # Evaluation times (t) and indicator array for inputs (su) and output (sy)
    # ==========================================================================

    # get times that the input changes
    # --------------------------------------------------------------------------
    # matlab: i = [1 (1 + find(any(diff(U.u),2))')]
    i = hstack((0, 1 + nonzero(np.any(np.diff(U['u'], axis=0), axis=1))[0]))
    su = wrappers.zeros_fn((1, u), dtype=bool)  # matlab: sparse(1,i,1,1,u)
    su.ravel()[i] = 1

    # get times that the response is sampled
    # --------------------------------------------------------------------------
    s = ceil(arange(v) * u / v).astype(int)
    sy = zeros((M['l'], u), dtype=int)
    for j in range(M['l']):
        i = s + D.ravel()[j].astype(int) - 1
        # matlab: sy[j,:] = sparse(1,i,1:v,1,u)
        # TODO: make this efficient/streamline it/test
        sy_ji = wrappers.zeros_fn(u)
        sy_ji[i] = np.arange(1, v + 1)
        sy[j, :] = sy_ji
        # sy[j, i] = arange(v) + 1

    # time in seconds
    # --------------------------------------------------------------------------
    t = nonzero(su.ravel() | np.any(sy, 0))[0]
    su = dense(su[:, t])  # matlab: full(su(:,t))
    sy = dense(sy[:, t])  # matlab: full(sy(:,t))
    dt = hstack((np.diff(t), 0)) * U['dt']

    # Integrate
    # --------------------------------------------------------------------------

    y = zeros((M['l'], v))
    J = copy.copy(M0)
    U['u'] = dense(U['u'])  # matlab: U.u = full(U.u)
    for i in range(size(t)):

        # input dependent changes in Jacobian
        # ----------------------------------------------------------------------
        if su[:, i]:
            u = U['u'][t[i], :]
            J = copy.copy(M0)
            for j in range(m):
                J = J + u[j] * M1[j]

        # output sampled
        # ----------------------------------------------------------------------
        if np.any(sy[:, i]):
            q = spm.unvec(x[1:], M['x'])
            q = spm.vec(g(q, u, P, M))
            j = nonzero(sy[:, i])[0]
            s = sy[j[0], i] - 1
            y[j, s] = q.ravel()[j]

        # matlab: x = spm_expm(J*dt[i],x)
        # x = sLA.expm(J * dt[i]) @ x
        x = wrappers.expm_mult_fn(J * dt[i], x)

        # check for convergence
        # ----------------------------------------------------------------------
        # norm(x, 1) is kinda slow here
        if norm1(x) > 1e6:
            break

    return y.T

