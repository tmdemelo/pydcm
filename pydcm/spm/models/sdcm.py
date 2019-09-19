from ...globs import *
from ...utils import dense, isvector
from ... import spm
from ... import wrappers

__all__ = ['spm_dcm_fmri_priors', 'spm_fx_fmri', 'spm_gx_fmri',
           'spm_gx_state_fmri']


def spm_dcm_fmri_priors(A, B, C, D, options):
    """
    Returns the priors for a two-state DCM for fMRI.
    FORMAT:[pE,pC,x] = spm_dcm_fmri_priors(A,B,C,D,options)

      options.two_state:  (0 or 1) one or two states per region
      options.stochastic: (0 or 1) exogenous or endogenous fluctuations
      options.precision:           log precision on connection rates

    INPUT:
       A,B,C,D - constraints on connections (1 - present, 0 - absent)

    OUTPUT:
       pE     - prior expectations (connections and hemodynamic)
       pC     - prior covariances  (connections and hemodynamic)
       x      - prior (initial) states
    _________________________________________________________________________

    References for state equations:
    1. Marreiros AC, Kiebel SJ, Friston KJ. Dynamic causal modelling for
       fMRI: a two-state model.
       Neuroimage. 2008 Jan 1;39(1):269-78.

    2. Stephan KE, Kasper L, Harrison LM, Daunizeau J, den Ouden HE,
       Breakspear M, Friston KJ. Nonlinear dynamic causal models for fMRI.
       Neuroimage 42:649-662, 2008.
    _________________________________________________________________________
    Copyright (C) 2008 Wellcome Trust Centre for Neuroimaging

    Karl Friston
    $Id: spm_dcm_fmri_priors.m 7270 2018-03-04 13:08:10Z karl $
    """

    options = copy.copy(options)

    # number of regions
    # --------------------------------------------------------------------------
    n = A.shape[0]

    # check options and D (for nonlinear coupling)
    # --------------------------------------------------------------------------
    if not 'stochastic' in options:
        options['stochastic'] = 0
    if not 'induced' in options:
        options['induced'] = 0
    if not 'two_state' in options:
        options['two_state'] = 0
    if not 'backwards' in options:
        options['backwards'] = 0
    if not D:
        D = zeros((n, n, 0))

    pE = {}
    pC = {}
    # connectivity priors and intitial states
    # ==========================================================================
    if options['two_state']:

        # (6) initial states
        # ----------------------------------------------------------------------
        x = wrappers.zeros_fn((n, 6))  # matlab: sparse(n,6)
        A = (A - np.diag(np.diag(A))).astype(bool)

        if 'precision' in options:
            pA = options['precision']
        else:
            pA = 16

        # prior expectations and variances
        # ----------------------------------------------------------------------
        pE['A'] = (A + eye(n)) * 32 - 32
        pE['B'] = B * 0
        pE['C'] = C * 0
        pE['D'] = D * 0

        # prior covariances
        # ----------------------------------------------------------------------
        A = atleast_3d(A)
        pC['A'] = zeros(A.shape)
        for i in range(A.shape[2]):
            pC['A'][:, :, i] = A[:, :, i] / pA + eye(n) / pA
        pC['B'] = B / 4
        pC['C'] = C * 4
        pC['D'] = D / 4

        # excitatory proportion
        # ----------------------------------------------------------------------
        if options['backwards']:
            pE['A'][:, :, 1] = A * 0
            pC['A'][:, :, 1] = A / pA

    else:

        # one hidden state per node
        # ======================================================================

        # (6 - 1) initial states
        # ----------------------------------------------------------------------
        x = wrappers.zeros_fn((n, 5))  # matlab: sparse(n,5)

        # precision of connections (one-state)
        # ---------------------------------------------------------------------
        if 'precision' in options:
            pA = options['precision']
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
            pE['A'] = (A.ravel() - 1) * dA
        else:
            A = (A - np.diag(np.diag(A))).astype(bool)
            pE['A'] = A / 128
        pE['B'] = B * 0
        pE['C'] = C * 0
        pE['D'] = D * 0

        # prior covariances
        # ----------------------------------------------------------------------
        if isvector(A):
            pC['A'] = A.ravel()
        else:
            A = atleast_3d(A)
            pC['A'] = zeros(A.shape)
            for i in range(A.shape[2]):
                pC['A'][:, :, i] = A[:, :, i] / pA + eye(n) / pA
        pC['B'] = B
        pC['C'] = C
        pC['D'] = D

    # and add hemodynamic priors
    # ==========================================================================

    # matlab: sparse(...) for all vars below
    pE['transit'] = wrappers.zeros_fn((n, 1))
    pC['transit'] = zeros((n, 1)) + 1 / 256
    pE['decay'] = wrappers.zeros_fn((1, 1))
    pC['decay'] = zeros((1, 1)) + 1 / 256
    pE['epsilon'] = wrappers.zeros_fn((1, 1))
    pC['epsilon'] = zeros((1, 1)) + 1 / 256

    # add prior on spectral density of fluctuations (amplitude and exponent)
    # --------------------------------------------------------------------------
    # matlab: sparse(...) for all vars below
    if options['induced']:
        # neuronal fluctuations
        pE['a'] = wrappers.zeros_fn((2, 1))
        pC['a'] = zeros((2, 1)) + 1 / 64
        # channel noise global
        pE['b'] = wrappers.zeros_fn((2, 1))
        pC['b'] = zeros((2, 1)) + 1 / 64
        # channel noise specific
        pE['c'] = wrappers.zeros_fn((1, n))
        pC['c'] = zeros((1, n)) + 1 / 64

    # prior covariance matrix
    # --------------------------------------------------------------------------
    pC = np.diag(spm.vec(pC).ravel())

    return pE, pC, x


def spm_fx_fmri(x, u, P, M=None):
    """
    State equation for a dynamic [bilinear/nonlinear/Balloon] model of fMRI
    responses
    FORMAT [f,dfdx,D,dfdu] = spm_fx_fmri(x,u,P,M)
    x      - state vector
      x(:,1) - excitatory neuronal activity            ue
      x(:,2) - vascular signal                          s
      x(:,3) - rCBF                                  ln(f)
      x(:,4) - venous volume                         ln(v)
      x(:,5) - deoyxHb                               ln(q)
     [x(:,6) - inhibitory neuronal activity             ui

    f      - dx/dt
    dfdx   - df/dx
    dfdu   - df/du
    D      - delays

    _________________________________________________________________________

    References for hemodynamic & neuronal state equations:
    1. Buxton RB, Wong EC & Frank LR. Dynamics of blood flow and oxygenation
       changes during brain activation: The Balloon model. MRM 39:855-864,
       1998.
    2. Friston KJ, Mechelli A, Turner R, Price CJ. Nonlinear responses in
       fMRI: the Balloon model, Volterra kernels, and other hemodynamics.
       Neuroimage 12:466-477, 2000.
    3. Stephan KE, Kasper L, Harrison LM, Daunizeau J, den Ouden HE,
       Breakspear M, Friston KJ. Nonlinear dynamic causal models for fMRI.
       Neuroimage 42:649-662, 2008.
    4. Marreiros AC, Kiebel SJ, Friston KJ. Dynamic causal modelling for
       fMRI: a two-state model.
       Neuroimage. 2008 Jan 1;39(1):269-78.
    _________________________________________________________________________
    Copyright (C) 2002-2014 Wellcome Trust Centre for Neuroimaging

    Karl Friston & Klaas Enno Stephan
    $Id: spm_fx_fmri.m 7270 2018-03-04 13:08:10Z karl $

    """

    # options
    # --------------------------------------------------------------------------
    if M is None:
        M = {}

    if 'symmetry' in M:
        symmetry = M['symmetry']
    else:
        symmetry = 0

    # needed to avoid overwriting original object members
    P = copy.deepcopy(P)

    # Neuronal motion
    # ==========================================================================
    # matlab: takes full() of A, B and D
    P['A'] = atleast_1d(dense(P['A']))                       # linear parameters
    P['B'] = atleast_1d(dense(P['B']))                       # bi-linear parameters
    P['C'] = atleast_1d(P['C']) / 16                    # exogenous parameters
    P['D'] = atleast_1d(dense(P['D']))                       # nonlinear parameters

    u = atleast_1d(u)
    x = atleast_2d(x).astype(float)  # astype() copies the data

    # implement differential state equation y = dx/dt (neuronal)
    # --------------------------------------------------------------------------
    f = copy.copy(x)

    # if there are five hidden states per region, only one is neuronal
    # ==========================================================================
    if x.shape[-1] == 5:

        # if P['A'] encodes the eigenvalues of the (average) connectivity matrix
        # ======================================================================
        if size(P['A']) != 1 and isvector(P['A']):
            # TODO
            raise Exception('Not implemented for P["A"] as vector')

        else:
            # otherwise average connections are encoded explicitly
            # ==================================================================

            # input dependent modulation
            # ------------------------------------------------------------------
            # TODO: streamline this
            if np.ndim(P['B']) > 2:
                if np.ndim(P['A']) > 2:
                    for i in range(P['B'].shape[2]):
                        P['A'][:, :, 0] = P['A'][:, :, 0] + u[i] * P['B'][:, :, i]
                else:
                    for i in range(P['B'].shape[2]):
                        P['A'][:, :] = P['A'][:, :] + u[i] * P['B'][:, :, i]
            else:
                P['A'] = P['A'] + u[0] * P['B']

            # and nonlinear (state) terms
            # ------------------------------------------------------------------
            if np.ndim(P['D']) > 2:
                for i in range(P['D'].shape[2]):
                    P['A'][:, :, 0] = P['A'][:, :, 0] + x[i, 0] * P['D'][:, :, i]

            # combine forward and backward connections if necessary
            if np.ndim(P['A']) > 2 and P['A'].shape[2] > 1:
                P['A'] = exp(P['A'][:, :, 0]) - exp(P['A'][:, :, 1])

            # one neuronal state per region: diag(A) is a log self-inhibition
            # ------------------------------------------------------------------
            SE = np.diag(P['A'])
            EE = P['A'] - np.diag(exp(SE) / 2 + SE)

            # symmetry constraints for demonstration purposes
            # ------------------------------------------------------------------
            if symmetry:
                EE = (EE + EE.T) / 2

        # flow
        # ----------------------------------------------------------------------
        f[:, 0] = EE @ x[:, 0] + P['C'] @ u.ravel()

    else:

        # otherwise two neuronal states per region
        # ======================================================================

        # input dependent modulation
        # ----------------------------------------------------------------------
        # TODO: streamline this
        if np.ndim(P['B']) > 2:
            if np.ndim(P['A']) > 2:
                for i in range(P['B'].shape[2]):
                    P['A'][:, :, 0] = P['A'][:, :, 0] + u[i] * P['B'][:, :, i]
            else:
                for i in range(P['B'].shape[2]):
                    P['A'][:, :] = P['A'][:, :] + u[i] * P['B'][:, :, i]
        else:
            P['A'] = P['A'] + u[0] * P['B']

        # and nonlinear (state) terms
        # ----------------------------------------------------------------------
        if np.ndim(P['D']) > 2:
            for i in range(P['D'].shape[2]):
                P['A'][:, :, 0] = P['A'][:, :, 0] + x[i, 0] @ P['D'][:, :, i]

        # extrinsic (two neuronal states): enforce positivity
        # ----------------------------------------------------------------------
        n = P['A'].shape[0]          # number of regions
        # TODO: streamline this
        if np.ndim(P['A']) > 2:
            EE = exp(P['A'][:, :, 0]) / 8
        else:
            EE = exp(P['A'][:, :]) / 8
        IE = diag(np.diag(EE))         # intrinsic inhibitory to excitatory
        EE = EE - IE                # extrinsic excitatory to excitatory
        EI = eye(n)                 # intrinsic excitatory to inhibitory
        SE = eye(n) / 2             # intrinsic self-inhibition (excitatory)
        SI = eye(n)                 # intrinsic self-inhibition (inhibitory)

        # excitatory proportion
        # ----------------------------------------------------------------------
        if P['A'].ndim > 2 and P['A'].shape[2] > 1:
            phi = spm.phi(P['A'][:, :, 1] * 2)
            EI = EI + EE * (1 - phi)
            EE = EE * phi - SE
        else:
            EE = EE - SE

        # motion - excitatory and inhibitory: f = dx/dt
        # ----------------------------------------------------------------------
        f[:, 0] = EE @ x[:, 0] - IE @ x[:, 5] + P['C'] @ u.ravel()
        f[:, 5] = EI @ x[:, 0] - SI @ x[:, 5]

    # Hemodynamic motion
    # ==========================================================================

    # hemodynamic parameters
    # --------------------------------------------------------------------------
    #   H(0) - signal decay                                   d(ds/dt)/ds)
    #   H(1) - autoregulation                                 d(ds/dt)/df)
    #   H(2) - transit time                                   (t0)
    #   H(3) - exponent for Fout(v)                           (alpha)
    #   H(4) - resting oxygen extraction                      (E0)
    #   H(5) - ratio of intra- to extra-vascular components   (epsilon)
    #          of the gradient echo signal
    # --------------------------------------------------------------------------
    if 'H' in P:
        H = P['H']
    else:
        H = array([0.64, 0.32, 2.00, 0.32, 0.4])

    # exponentiation of hemodynamic state variables
    # --------------------------------------------------------------------------
    x[:, 2:5] = exp(x[:, 2:5])

    # signal decay
    # --------------------------------------------------------------------------
    sd = H[0] * exp(P['decay'].ravel())

    # transit time
    # --------------------------------------------------------------------------
    tt = H[2] * exp(P['transit'].ravel())

    # Fout = f(v) - outflow
    # --------------------------------------------------------------------------
    fv = x[:, 3]**(1 / H[3])

    # e = f(f) - oxygen extraction
    # --------------------------------------------------------------------------
    ff = (1 - (1 - H[4])**(1 / x[:, 2])) / H[4]

    # implement differential state equation f = dx/dt (hemodynamic)
    # --------------------------------------------------------------------------
    f[:, 1] = x[:, 0] - sd * x[:, 1] - H[1] * (x[:, 2] - 1)
    f[:, 2] = x[:, 1] / x[:, 2]
    f[:, 3] = (x[:, 2] - fv) / (tt * x[:, 3])
    f[:, 4] = (ff * x[:, 2] - fv * x[:, 4] / x[:, 3]) / (tt * x[:, 4])
    f = f.ravel(order='F')

    return f


# notice that arg M is not used
# not setting its default arg might give issues with 3 arg lambdas at spm_DEM_* functions
def spm_gx_fmri(x, u, P, M=None):
    """
    Simulated BOLD response to input
    FORMAT [g,dgdx] = spm_gx_fmri(x,u,P,M)
    g          - BOLD response (%)
    x          - state vector     (see spm_fx_fmri)
    P          - Parameter vector (see spm_fx_fmri)
    M          - model specification structure (see spm_nlsi)
    _________________________________________________________________________

    This function implements the BOLD signal model described in:

    Stephan KE, Weiskopf N, Drysdale PM, Robinson PA, Friston KJ (2007)
    Comparing hemodynamic models with DCM. NeuroImage 38: 387-401.
    _________________________________________________________________________
    Copyright (C) 2008 Wellcome Trust Centre for Neuroimaging

    Karl Friston & Klaas Enno Stephan
    $Id: spm_gx_fmri.m 6262 2014-11-17 13:47:56Z karl $
    """

    x = atleast_2d(x)

    # Biophysical constants for 1.5T
    # ==========================================================================

    # time to echo (TE) (default 0.04 sec)
    # --------------------------------------------------------------------------
    TE = 0.04

    # resting venous volume (%)
    # --------------------------------------------------------------------------
    V0 = 4

    # estimated region-specific ratios of intra- to extra-vascular signal
    # --------------------------------------------------------------------------
    ep = exp(P['epsilon'])

    # slope r0 of intravascular relaxation rate R_iv as a function of oxygen
    # saturation S:  R_iv = r0*[(1 - S)-(1 - S0)] (Hz)
    # --------------------------------------------------------------------------
    r0 = 25

    # frequency offset at the outer surface of magnetized vessels (Hz)
    # --------------------------------------------------------------------------
    nu0 = 40.3

    # resting oxygen extraction fraction
    # --------------------------------------------------------------------------
    E0 = 0.4

    # Coefficients in BOLD signal model
    # ==========================================================================
    k1 = 4.3 * nu0 * E0 * TE
    k2 = ep * r0 * E0 * TE
    k3 = 1 - ep

    # Output equation of BOLD signal model
    # ==========================================================================
    v = exp(x[:, 3])
    q = exp(x[:, 4])
    # g = V0 @ ...
    g = V0 * (k1 - k1 * q + k2 - k2 * q / v + k3 - k3 * v)

    # matlab: if nargout == 1, return, end

    # TODO: # derivative dgdx
    # ==========================================================================

    return g


def spm_gx_state_fmri(x, u, P, M=None):
    """
    Simulated BOLD response and copied state vector
    FORMAT [y] = spm_gx_state_fmri(x,u,P,M)
    y          - BOLD response and copied state vector

    x          - state vector     (see spm_fx_fmri)
    P          - Parameter vector (see spm_fx_fmri)
    M          - model specification structure (see spm_nlsi)

    The `copied state vector' passes the first hidden variable in each region
    to the output variable y, so that 'neural activities' can be plotted
    by spm_dcm_generate.m

    See spm_fx_fmri.m and spm_dcm_generate.m
    _________________________________________________________________________
    Copyright (C) 2011 Wellcome Trust Centre for Neuroimaging

    Will Penny
    $Id: spm_gx_state_fmri.m 6262 2014-11-17 13:47:56Z karl $
    """
    y = atleast_2d(spm.gx_fmri(x, u, P, M)).T
    x = atleast_2d(x)

    # Copy first hidden state (neural activity) from each region
    # matlab: y=[y;x(i,1)] in a loop
    y = vstack((y, x[:, [0]]))

    y = dense(y)  # matlab: y = full(y)
    return y.T


