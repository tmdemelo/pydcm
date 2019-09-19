from .. import spm
import pydcm
from ..globs import *
from ..utils import *
from .. import wrappers

__all__ = ['spm_nlsi_GN']

# [Ep,Cp,Eh,F,L,dFdp,dFdpp] = spm_nlsi_GN(M,U,Y)
def spm_nlsi_GN(M, U=[], Y={}):
    """
    Bayesian inversion of nonlinear models - Gauss-Newton/Variational Laplace
    FORMAT [Ep,Cp,Eh,F] = spm_nlsi_GN(M,U,Y)

    [Dynamic] MIMO models
    _________________________________________________________________________

    M.IS - function name f(P,M,U) - generative model
           This function specifies the nonlinear model:
             y = Y.y = IS(P,M,U) + X0*P0 + e
           where e ~ N(0,C). For dynamic systems this would be an integration
           scheme (e.g. spm_integ). spm_integ expects the following:

        M.f  - f(x,u,P,M)
        M.g  - g(x,u,P,M)
        M.h  - h(x,u,P,M)
          x  - state variables
          u  - inputs or causes
          P  - free parameters
          M  - fixed functional forms and parameters in M

    M.FS - function name f(y,M)   - feature selection
           This [optional] function performs feature selection assuming the
           generalized model y = FS(y,M) = FS(IS(P,M,U),M) + X0*P0 + e

    M.P  - starting estimates for model parameters [optional]

    M.pE - prior expectation      - E{P}   of model parameters
    M.pC - prior covariance       - Cov{P} of model parameters

    M.hE - prior expectation      - E{h}   of log-precision parameters
    M.hC - prior covariance       - Cov{h} of log-precision parameters

    U.u  - inputs (or just U)
    U.dt - sampling interval

    Y.y  - outputs (samples x observations x ...)
    Y.dt - sampling interval for outputs
    Y.X0 - confounds or null space      (over size(y,1) samples or all vec(y))
    Y.Q  - q error precision components (over size(y,1) samples or all vec(y))


    Parameter estimates
    -------------------------------------------------------------------------
    Ep  - (p x 1)         conditional expectation    E{P|y}
    Cp  - (p x p)         conditional covariance     Cov{P|y}
    Eh  - (q x 1)         conditional log-precisions E{h|y}

    log evidence
    -------------------------------------------------------------------------
    F   - [-ve] free energy F = log evidence = p(y|f,g,pE,pC) = p(y|m)

    _________________________________________________________________________
    Returns the moments of the posterior p.d.f. of the parameters of a
    nonlinear model specified by IS(P,M,U) under Gaussian assumptions.
    Usually, IS is an integrator of a dynamic MIMO input-state-output model

                 dx/dt = f(x,u,P)
                 y     = g(x,u,P)  + X0*P0 + e

    A static nonlinear observation model with fixed input or causes u
    obtains when x = []. i.e.

                 y     = g([],u,P) + X0*P0e + e

    but static nonlinear models are specified more simply using

                 y     = IS(P,M,U) + X0*P0 + e

    Priors on the free parameters P are specified in terms of expectation pE
    and covariance pC. The E-Step uses a Fisher-Scoring scheme and a Laplace
    approximation to estimate the conditional expectation and covariance of P
    If the free-energy starts to increase,  an abbreviated descent is
    invoked.  The M-Step estimates the precision components of e, in terms
    of log-precisions.  Although these two steps can be thought of in
    terms of E and N steps they are in fact variational steps of a full
    variational Laplace scheme that accommodates conditional uncertainty
    over both parameters and log precisions (c.f. hyperparameters with hyper
    priors)

    An optional feature selection can be specified with parameters M.FS.

    For generic aspects of the scheme see:

    Friston K, Mattout J, Trujillo-Barreto N, Ashburner J, Penny W.
    Variational free energy and the Laplace approximation.
    NeuroImage. 2007 Jan 1;34(1):220-34.

    This scheme handels complex data along the lines originally described in:

    Sehpard RJ, Lordan BP, and Grant EH.
    Least squares analysis of complex data with applications to permittivity
    measurements.
    J. Phys. D. Appl. Phys 1970 3:1759-1764.

    _________________________________________________________________________
    Copyright (C) 2001-2015 Wellcome Trust Centre for Neuroimaging

    Karl Friston
    $Id: spm_nlsi_GN.m 7279 2018-03-10 21:22:44Z karl $
    """
    # options
    # --------------------------------------------------------------------------
    if not 'nograph' in M:
        M['nograph'] = True  # matlab: = 0, let's not use guis for now
    if not 'noprint' in M:
        M['noprint'] = False
    if not 'Nmax' in M:
        M['Nmax'] = 128

    # figure (unless disabled)
    # --------------------------------------------------------------------------
    # TODO?

    # check integrator
    # --------------------------------------------------------------------------
    if not 'IS' in M:
        M['IS'] = 'spm.integ'

    # composition of feature selection and prediction (usually an integrator)
    # --------------------------------------------------------------------------
    if 'y' in Y:
        y = Y['y']
    else:
        y = Y

    # TODO: make this section safer, without eval?
    try:

        # try FS(y,M)
        # ----------------------------------------------------------------------
        try:
            # obs: M['FS'] might not exist
            y = eval(M['FS'])(y, M)
            # TODO: make this more clear, change formatting?
            IS_code = 'lambda P, M, U:' + M['FS'] + '(' + M['IS'] + '(P, M, U), M)'
            IS = eval(IS_code)

            # try FS(y)
            # ------------------------------------------------------------------
        except:
            y = eval(M['FS'])(y)
            IS_code = 'lambda P, M, U:' + M['FS'] + '(' + M['IS'] + '(P, M, U))'
            IS = eval(IS_code)
    except:
        # otherwise FS(y) = y
        # ----------------------------------------------------------------------
        try:
            IS_code = 'lambda P, M, U:' + M['IS'] + '(P, M, U)'
            IS = eval(IS_code)
        except:
            IS = M['IS']

    # converted to function handle
    # --------------------------------------------------------------------------
    # TODO: check if really needed
    IS = spm.funcheck(IS)

    # TODO: report typo below
    # parameter update equation
    # --------------------------------------------------------------------------
    if 'f' in M:
        M['f'] = spm.funcheck(M['f'])
    if 'g' in M:
        M['g'] = spm.funcheck(M['g'])
    if 'h' in M:
        M['h'] = spm.funcheck(M['h'])

    # size of data (samples x response component x response component ...)
    # --------------------------------------------------------------------------
    if isinstance(y, (list, tuple)):
        ns = max(y[0].shape)
    else:
        ns = y.shape[0]
    ny = spm.vec(y).shape[0]  # total number of response variables
    nr = ny / ns  # number response components
    M['ns'] = ns  # number of samples M['ns']

    # initial states
    # --------------------------------------------------------------------------
    if not 'x' in M:
        if not 'n' in M:
            M['n'] = 0
        M['x'] = wrappers.zeros_fn(M['n'], 1)  # matlab: sparse(...)

    # input
    # --------------------------------------------------------------------------
    # unlike matlab, we can set default U = [] at function declaration

    # initial parameters
    # --------------------------------------------------------------------------
    try:
        # TODO: test/guess what could go wrong here (M['P'] might not exist. sizes could be different)
        spm.vec(M['P']) - spm.vec(M['pE'])
        print('\nParameter initialisation successful\n')
    except:
        M['P'] = copy.deepcopy(M['pE'])

    # time-step
    # --------------------------------------------------------------------------
    if 'dt' in Y:
        dt = Y['dt']
    else:
        dt = 1

    # precision components Q
    # --------------------------------------------------------------------------
    try:
        # TODO: check/guess what could go wrong here
        Q = Y['Q']
        if isnumericarray(Q):
            Q = [Q]  # matlab: Q = {Q}
    except:
        Q = spm.Ce(ns * ones((1, nr)))  # TODO: test this
    nh = len(shape(Q))  # number of precision components
    nq = int(ny / max(shape(Q[0])))  # for compact Kronecker form of M-step

    # prior moments (assume uninformative priors if not specifed)
    # --------------------------------------------------------------------------
    # Note: very important to copy any data that will be unvec'ed
    # otherwise, it might mess up with the serialization process
    pE = copy.deepcopy(M['pE'])
    if 'pC' in M:
        pC = M['pC']
    else:
        nP = spm.length(M['pE'])
        pC = wrappers.eye_fn(nP) * exp(16)  # matlab: speye(...)

    # confounds (if specified)
    # --------------------------------------------------------------------------
    try:
        # TODO: test/guess what could go wrong here
        nb = Y['X0'].shape[0]  # number of bins
        nx = int(ny / nb)  # number of blocks
        dfdu = kron(wrappers.eye_fn(nx), Y['X0'])  # matlab: speye(...)
    except:
        dfdu = wrappers.zeros_fn((ny, 0))  # matlab: sparse(...)
    if dfdu.size == 0:
        dfdu = wrappers.zeros_fn((ny, 0))  # matlab: sparse(...)

    # hyperpriors - expectation (and initialize hyperparameters)
    # --------------------------------------------------------------------------
    try:
        # TODO: test/guess what could go wrong here
        # likely a missing hE attribute
        hE = M['hE']
        if max(hE.shape) != nh:
            hE = hE + wrappers.zeros_fn((nh, 1))  # matlab: sparse(...)
    except:
        hE = zeros((nh, 1)) - log(var(spm.vec(y), ddof=1)) + 4
    h = hE

    # hyperpriors - covariance
    # --------------------------------------------------------------------------
    try:
        # TODO: test/guess what could go wrong here
        #       likely a missing M.hC attribute
        ihC = spm.inv(M['hC'])
        if max(ihC.shape) != nh:
            ihC = ihC @ wrappers.eye_fn(nh)  # matlab: speye
    except:
        # TODO: find out what is this magic number 4
        ihC = eye(nh) * exp(4)

    # unpack covariance
    # --------------------------------------------------------------------------
    if isinstance(pC, dict):  # matlab: isstruct(pC)
        pC = spm.diag(spm.vec(pC))

    # dimension reduction of parameter space
    # --------------------------------------------------------------------------

    V, *_ = spm.svd(pC, 0)  # matlab: spm_svd(pC, 0)
    nu = dfdu.shape[1]  # number of parameters (confounds)
    nP = V.shape[1]  # number of parameters (effective)
    ip = arange(0, nP).T
    iu = arange(0, nu).T + nP  # matlab: (1:nu) ...

    # second-order moments (in reduced space)
    # --------------------------------------------------------------------------
    pC = V.T @ pC @ V
    uC = wrappers.eye_fn(nu) / 1e-8  # matlab: speye
    ipC = LA.inv(spm.cat(spm.diag([pC, uC])))  # TODO: test this

    # initialize conditional density
    # --------------------------------------------------------------------------
    # TODO: be mindful that variable y requires transposition here
    # unless spm_vec returns output.ravel(order='F')
    # otherwise, wrong result
    Eu = spm.pinv(dfdu) @ spm.vec(y)
    p = vstack((V.T @ (spm.vec(M['P']) - spm.vec(M['pE'])), Eu))
    Ep = spm.unvec(spm.vec(pE) + V @ p[ip], pE)

    # EM
    # ==========================================================================
    criterion = zeros(4)

    C = {}
    C['F'] = -inf  # free energy
    v = -4  # log ascent rate
    dFdh = zeros((nh, 1))
    dFdhh = zeros((nh, nh))
    F0 = None  # for printing F

    e_step_iters = 4
    m_step_iters = 8

    for k in range(M['Nmax']):

        # time
        # ----------------------------------------------------------------------
        tStart = time.time()  # matlab: tStart = tic

        # E-Step: prediction f, and gradients; dfdp
        # ======================================================================
        try:
            # gradients
            # ------------------------------------------------------------------
            dfdp, f = spm.diff(IS, Ep, M, U, 0, [V])
            dfdp = spm.vec(dfdp).reshape((ny, nP), order='F')

            # check for stability
            # ------------------------------------------------------------------
            normdfdp = maxnorm(dfdp)
            revert = isnan(normdfdp) or normdfdp > exp(32)
        except ArithmeticError:
            # TODO: check for other kinds of exceptions when it doesnt converge
            logging.warning(traceback.format_exc())
            revert = True

        if revert and k > 0:
            # TODO: tests for this block
            for i in range(e_step_iters):

                # reset expansion point and increase regularization
                # --------------------------------------------------------------
                v = minimum(v - 2, -4)

                # E-Step: update
                # --------------------------------------------------------------
                p = C['p'] + spm.dx(dFdpp, dFdp, [v])
                Ep = spm.unvec(spm.vec(pE) + V * p[ip], pE)

                # try again
                # --------------------------------------------------------------
                try:

                    dfdp, f, *_ = spm.diff(IS, Ep, M, U, 0, [V])
                    dfdp = spm.vec(dfdp).reshape((ny, nP), order='F')

                    # check for stability
                    # ----------------------------------------------------------
                    normdfdp = maxnorm(dfdp)
                    revert = isnan(normdfdp) or normdfdp > exp(32)
                except ArithmeticError:
                    # TODO: check for other for other kinds of exceptions when it doesnt converge
                    logging.warning(traceback.format_exc())
                    revert = True

                # break
                # --------------------------------------------------------------
                if not revert:
                    break

        # convergence failure
        # ----------------------------------------------------------------------
        if revert:
            # matlab: 'SPM:spm_nlsi_GN' is actually an error identifier
            # TODO: define custom Exception?
            raise Exception('SPM:spm_nlsi_GN: Convergence failure.')

        # prediction error and full gradients
        # ----------------------------------------------------------------------
        e = spm.vec(y) - spm.vec(f) - dfdu @ p[iu]
        J = -hstack((dfdp, dfdu))

        # M-step: Fisher scoring scheme to find h = max{F(p,h)}
        # ======================================================================
        for m in range(m_step_iters):

            # precision and conditional covariance
            # ------------------------------------------------------------------
            iS = 0  # matlab: sparse(0)
            for i in range(nh):
                iS = iS + Q[i] * (exp(-32) + exp(h.ravel()[i]))
            S = spm.inv(iS)
            iS = kron(wrappers.eye_fn(nq), iS)  # matlab: speye
            Pp = (J.T @ iS @ J).real
            Cp = spm.inv(Pp + ipC)

            # precision operators for M-Step
            # ------------------------------------------------------------------
            P = [None] * nh
            PS = [None] * nh
            JPJ = [None] * nh
            for i in range(nh):
                P[i] = Q[i] * exp(h.ravel()[i])
                PS[i] = P[i] @ S
                P[i] = kron(eye(nq), P[i])
                JPJ[i] = J.T @ P[i] @ J

            # derivatives: dLdh = dL/dh,...
            # ------------------------------------------------------------------
            for i in range(nh):
                dFdh[i, 0] = sptrace(PS[i]) * nq / 2        \
                             - (e.T @ P[i] @ e).real / 2  \
                             - spm.trace(Cp, JPJ[i]) / 2
                for j in range(i, nh):
                    dFdhh[i, j] = - spm.trace(PS[i], PS[j]) * nq / 2
                    dFdhh[j, i] = dFdhh[i, j]

            # add hyperpriors
            # ------------------------------------------------------------------
            d = h - hE
            dFdh = dFdh - ihC @ d
            dFdhh = dFdhh - ihC
            Ch = spm.inv(-dFdhh.real)

            # update ReML estimate
            # ------------------------------------------------------------------
            dh = spm.dx(dFdhh, dFdh, [4])
            dh = minimum(maximum(dh, -1), 1)
            h = h + dh

            # convergence
            # ------------------------------------------------------------------
            dF = dFdh.T @ dh
            if dF < 1e-2:
                break

        # E-Step with Levenberg-Marquardt regularization
        # ======================================================================

        # objective function: F(p) = log evidence - divergence
        # ----------------------------------------------------------------------
        L = np.empty(3)  # preallocate
        L[0] = spm.logdet(iS) * nq / 2 - (e.T @ iS @ e).real / 2 - ny * log(8 * arctan(1)) / 2
        L[1] = spm.logdet(ipC @ Cp) / 2 - p.T @ ipC @ p / 2
        L[2] = spm.logdet(ihC @ Ch) / 2 - d.T @ ihC @ d / 2
        F = np.sum(L)

        # record increases and reference log-evidence for reporting
        # ----------------------------------------------------------------------
        if F0 is not None:
            if not M['noprint']:
                print(' actual: %.3e (%.2f sec)' % ((F - C['F']), time.time() - tStart))
        else:
            F0 = F

        # if F has increased, update gradients and curvatures for E-Step
        # ----------------------------------------------------------------------
        if F > C['F'] or k < 2:

            # accept current estimates
            # ------------------------------------------------------------------
            C['p'] = p
            C['h'] = h
            C['F'] = F
            C['L'] = L
            C['Cp'] = Cp

            # E-Step: Conditional update of gradients and curvature
            # ------------------------------------------------------------------
            dFdp = - (J.T @ iS @ e).real - ipC @ p
            dFdpp = - (J.T @ iS @ J).real - ipC

            # decrease regularization
            # ------------------------------------------------------------------
            v = minimum(v + 1 / 2, 4)
            msg = 'EM:(+)'

        else:

            # reset expansion point
            # ------------------------------------------------------------------
            p = C['p']
            h = C['h']
            Cp = C['Cp']

            # and increase regularization
            # ------------------------------------------------------------------
            v = minimum(v - 2, -4)
            msg = 'EM:(-)'

        # E-Step: update
        # ======================================================================
        dp = spm.dx(dFdpp, dFdp, [v])
        p = p + dp
        Ep = spm.unvec(spm.vec(pE) + V @ p[ip], pE)

        # Graphics
        # ======================================================================
        # TODO: graphics, maybe...

        # convergence
        # ----------------------------------------------------------------------
        dF = dFdp.T @ dp

        if not M['noprint']:
            print('%-6s: %i %6s %-6.3e %6s %.3e ' % (msg, k, 'F:', C['F'] - F0, 'dF predicted:', dF), end='', flush=True)

        criterion = hstack((dF.item() < 1e-1, criterion[:-1]))
        if np.all(criterion):
            if not M['noprint']:
                print(' convergence')
            break

    # TODO: graphics, maybe, focusing the figure with Fsi

    # outputs
    # --------------------------------------------------------------------------

    Ep = spm.unvec(spm.vec(pE) + V @ C['p'][ip], pE)
    Cp = V @ C['Cp'][ix_(ip, ip)] @ V.T
    Eh = C['h']
    F = C['F']
    L = C['L']

    return Ep, Cp, Eh, F, L, dFdp, dFdpp
