from ..globs import *
from ..utils import dense, isvector
from .. import spm
from .. import wrappers


# TODO: docstring
def pdcm_fmri_priors(A, B, C, D, options):

    pE = {}
    pC = {}

    # number of regions
    # --------------------------------------------------------------------------
    n = A.shape[0]

    # connectivity priors and intitial states
    # ==========================================================================
    # Havlicek 2015, supplementary info 5

    # initial states (6)
    #----------------------------------------------------------------------
    x  = zeros((n, 6))

    # priors for A borrowed from 1S-DCM
    # precision of connections
    # ---------------------------------------------------------------------
    if 'precision' in options:
        pA = exp(options['precision'])
    else:
        pA = 64
    if 'decay' in options:
        dA = options['decay']
    else:
        dA = 1

    # prior expectations
    # ----------------------------------------------------------------------
    if isvector(A):
        A = A.astype(bool)
        pE['A']  = (A.ravel() - 1) * dA
    else:
        A = (A - np.diag(np.diag(A))).astype(bool)
        pE['A']  = A / 128
    pE['B']  = B * 0
    pE['C']  = C * 0
    pE['D']  = D * 0

    # prior covariances
    # ----------------------------------------------------------------------
    if isvector(A):
        pC['A']  = A.ravel()
    else:
        A = atleast_3d(A)
        pC['A'] = zeros(A.shape)
        for i in range(A.shape[2]):
            pC['A'][:, :, i] = A[:, :, i] / pA + eye(n,n) / pA
    pC['B']  = B.astype(bool) * exp(-2)  # = B
    pC['C']  = C.astype(bool) * exp(-1)  # = C
    pC['D']  = D.astype(bool) * exp(-2)  # = D

    # other neuronal priors
    # ----------------------------------------------------------------------
    pE['sigmas']   = zeros((n, 1))
    pC['sigmas']   = zeros((n, 1)) + exp(-4)
    pE['mus']      = zeros((n, 1))
    pC['mus']      = zeros((n, 1)) + exp(-4)
    pE['lambdas']  = zeros((n, 1))
    pC['lambdas']  = zeros((n, 1)) + exp(-4)

    # hemodynamic priors
    # =======================================================================
    pE['transit'] = zeros((n, 1))
    pC['transit'] = zeros((n, 1)) + exp(-4)
    pE['signaldecay'] = zeros((1, 1))
    pC['signaldecay'] = exp(-4)  # not fit?
    # pE['decay'] = zeros((1, 1))
    # pC['decay'] = exp(-4)

    pE['epsilon'] = zeros((1, 1))
    pC['epsilon'] = exp(-6)

    # p-dcm specific
    # ----------------------------------------------------------------------
    pE['gain']      = zeros((1, 1))
    pC['gain']      = exp(-4)
    pE['flowdecay'] = zeros((1, 1))
    pC['flowdecay'] = 1  # not fit
    pE['visco']     = zeros((1, 1))
    pC['visco']     = exp(-2)

    # prior covariance matrix
    # --------------------------------------------------------------------------
    pC = np.diag(spm.vec(pC).ravel())

    return pE, pC, x



def fx_fmri_pdcm(x, u, P, M=None):
    """
    P-DCM State equation for a dynamic [bilinear/nonlinear/Balloon] model of fMRI
    responses
    FORMAT [f,dfdx,D,dfdu] = spm_fx_fmri(x,u,P,M)
    x      - state vector
    x(:,1) - excitatory neuronal activity            ue
    x(:,2) - vascular signal                          s
    x(:,3) - rCBF                                  ln(f)
    x(:,4) - venous volume                         ln(v)
    x(:,5) - deoyxHb                               ln(q)
    [x(:,6) - inhibitory neuronal activity             ui
    """

    # neuronal parameters
    #--------------------------------------------------------------------------
    MU = 0.4      # μ, excitatory self connection (Hz)
    SIGMA = 0.5   # σ, inhibitory-excitatory connection (Hz)
    LAMBDA = 0.2  # λ, inhibitory gain factor (Hz)

    # neurovascular coupling parameters
    #--------------------------------------------------------------------------
    PHId  = 0.6   # φ, decay of vasoactive signal(Hz), maybe fixed?
    PHIg  = 1.5   # Φ, gain of vasoactive signal (Hz), maybe fixed?
    CHI   = 0.6   # χ, decay of blood inflow signal (Hz), fixed

    # hemodynamic model parameters
    #--------------------------------------------------------------------------
    MTT   = 2.00  # mean transit time (sec)
    TAU   = 4     # τ, viscoelastic time (sec)
    ALPHA = 0.32  # α, aka Grubb's exp
    E0    = 0.4   # oxygen extraction fraction at rest

    P = copy.deepcopy(P)
    # Neuronal motion
    #==========================================================================
    # matlab: takes full() of A, B and D
    P['A'] = atleast_1d(dense(P['A']))                       # linear parameters
    P['B'] = atleast_1d(dense(P['B']))                       # bi-linear parameters
    P['C'] = atleast_1d(P['C']) / 16                    # exogenous parameters

    n  = P['A'].shape[0]            # number of regions
    uB = zeros((n, n))

    # implement differential state equation y = dx/dt (neuronal)
    #--------------------------------------------------------------------------
    f = copy.copy(x)
    x = copy.copy(x)

    # two neuronal states per region
    #======================================================================

    # input dependent modulation
    #----------------------------------------------------------------------
    for i in range(P['B'].shape[2]):
        uB = uB + u[i] * P['B'][:,:,i]

    # extrinsic (two neuronal states)
    #----------------------------------------------------------------------

    # P-DCM equations:
    #  d/dt Xe[t] = J[+] * Xe[t] + J[-] * Xi[t] +  C * U[t]
    #  d/dt Xi[t] = 𝔊[Xe[t] - Xi[t]]
    #
    #  J[+]_ij = A + uB
    #  J[-]_ij = 0
    #  G_ij    = 0
    #
    #  J[+]_ii = -σ * exp(σ~ + uB)
    #  J[-]_ii = -μ * exp(μ~_i + Σb_μi * u_μk)
    #  G_ii    =  λ * exp(λ~_i + Σb_λi * u_λl )

    I = eye(n).astype(bool)
    JP = P['A'] + uB
    JN = zeros((n, n))
    G  = zeros((n, n))


    JP[I] = - SIGMA * exp(P['sigmas'].ravel() + np.diag(uB))
    JN[I] = - MU * exp(P['mus'].ravel() + np.diag(uB))
    G[I]  = LAMBDA * exp(P['lambdas'].ravel() + np.diag(uB))

    # motion - excitatory and inhibitory: f = dx/dt
    #----------------------------------------------------------------------
    # d/dt Xe[t] = J[+] *  Xe[t] + J[-] *  Xi[t] +   C * U[t]
    f[:, 0] =  JP  @ x[:, 0] + JN   @ x[:, 5] + P['C'] @ u.ravel()
    # d/dt Xi[t] = G * ( Xe[t] - Xi[t] )
    f[:, 5] =  G @ (x[:,0] - x[:,5])

    # Hemodynamic motion
    #==========================================================================

    # neurovascular coupling and hemodynamic variables
    #--------------------------------------------------------------------------
    #  a[t]: vasoactive signal
    #  f[t]: blood flow
    #  v[t]: blood volume
    #  q[t]: dHb content
    #
    # neurovascular equations
    #--------------------------------------------------------------------------
    #  d/dt a[t] = -φ * a[t] + x[t]
    #  d/dt f[t] =  Φ * a[t] - χ * [f[t] - 1]
    #
    # hemodynamic equations (same as 1s and 2d dcm)
    #--------------------------------------------------------------------------
    #  d/dt v[t] = 1/MTT * [f[t] - fout(v,t)]
    #  d/dt q[t] = 1/MTT * [f[t] * E[f] / E0 - fout(v,t) * q[t]/v[t]]

    # exponentiation of hemodynamic state variables
    #--------------------------------------------------------------------------
    x[:, 2:5] = np.exp(x[:, 2:5])  # f, v, q

    # scale variables
    #--------------------------------------------------------------------------
    sd  = PHId * exp(P['signaldecay']).ravel()  # φ signal decay
    sg  = PHIg * exp(P['gain']).ravel()         # Φ signal gain
    #fd  = CHI * exp(P['flowdecay']).ravel()    # χ flow decay
    fd  = CHI                        # not fit (suppl. info 5)
    tt  = MTT * exp(P['transit']).ravel()       # transit time, fit (suppl. info 5)
    vt  = TAU * exp(P['visco']).ravel()         # τ viscoelastic time, fit

    # Fout = f[v] - outflow             fout(v,t)
    #--------------------------------------------------------------------------
    # P-DCM includes a viscoelastic effect
    fv = (tt *  x[:, 3] ** (1 / ALPHA) + vt * x[:, 2]) / (vt + tt)

    # e = f[f] - oxygen extraction      E[f]/E0
    #--------------------------------------------------------------------------
    ff = (1 - (1 - E0) ** (1 / x[:, 2])) / E0

    # a[t]: vasoactive signal
    #--------------------------------------------------------------------------
    f[:, 1] = - sd * x[:, 1] + x[:, 0]

    # f[t]: flow  (log units)
    #--------------------------------------------------------------------------
    f[:, 2] = (sg * x[:, 1] - fd * (x[:, 2] - 1)) / x[:, 2]

    # v[t]: blood volume  (log units)
    #--------------------------------------------------------------------------
    f[:, 3] =  (x[:, 2] - fv) / (tt * x[:, 3])

    # q[t]: dHB content  (log units)
    #--------------------------------------------------------------------------
    f[:, 4] = (ff  * x[:, 2] - fv  * x[:, 4] / x[:, 3]) / (tt * x[:, 4])

    #import pdb; pdb.set_trace()

    f = f.ravel(order='F')

    # if nargout < 2, return, end

    # TODO: Jacobians

    return f
