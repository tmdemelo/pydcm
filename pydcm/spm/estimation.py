import pydcm
import pydcm.models
from ..globs import *
from .. import spm
from .. import wrappers
from ..utils import (
    dense
)

__all__ = ['spm_dcm_estimate', 'spm_dcm_evidence']

# models must be imported for the eval and lambdas to work on later functions
# otherwise, it might throw a 'function_name' is not defined error
# not sure if still valid, must test
# from .models import *


def spm_dcm_estimate(P):
    """
    Estimates parameters of a DCM (bilinear or nonlinear) for fMRI data
    FORMAT [DCM] = spm_dcm_estimate(DCM)
      DCM - DCM structure or its filename

    Expects
    -------------------------------------------------------------------------
    DCM.a                              % switch on endogenous connections
    DCM.b                              % switch on bilinear modulations
    DCM.c                              % switch on exogenous connections
    DCM.d                              % switch on nonlinear modulations
    DCM.U                              % exogenous inputs
    DCM.Y.y                            % responses
    DCM.Y.X0                           % confounds
    DCM.Y.Q                            % array of precision components
    DCM.n                              % number of regions
    DCM.v                              % number of scans

    Options
    -------------------------------------------------------------------------
    DCM.options.two_state              % two regional populations (E and I)
    DCM.options.stochastic             % fluctuations on hidden states
    DCM.options.centre                 % mean-centre inputs
    DCM.options.nonlinear              % interactions among hidden states
    DCM.options.nograph                % graphical display
    DCM.options.induced                % switch for CSD data features
    DCM.options.P                      % starting estimates for parameters
    DCM.options.hidden                 % indices of hidden regions
    DCM.options.maxnodes               % maximum number of (effective) nodes
    DCM.options.maxit                  % maximum number of iterations
    DCM.options.hE                     % expected precision of the noise
    DCM.options.hC                     % variance of noise expectation

    Evaluates:
    -------------------------------------------------------------------------
    DCM.M                              % Model structure
    DCM.Ep                             % Condition means (parameter structure)
    DCM.Cp                             % Conditional covariances
    DCM.Vp                             % Conditional variances
    DCM.Pp                             % Conditional probabilities
    DCM.H1                             % 1st order hemodynamic kernels
    DCM.H2                             % 2nd order hemodynamic kernels
    DCM.K1                             % 1st order neuronal kernels
    DCM.K2                             % 2nd order neuronal kernels
    DCM.R                              % residuals
    DCM.y                              % predicted data
    DCM.T                              % Threshold for Posterior inference
    DCM.Ce                             % Error variance for each region
    DCM.F                              % Free-energy bound on log evidence
    DCM.ID                             % Data ID
    DCM.AIC                            % Akaike Information criterion
    DCM.BIC                            % Bayesian Information criterion

    _________________________________________________________________________
    Copyright (C) 2002-2012 Wellcome Trust Centre for Neuroimaging

    Karl Friston
    $Id: dcm_estimate.m 7479 2018-11-09 14:17:33Z peter $
    """

    # Load DCM structure
    # --------------------------------------------------------------------------
    # TODO? display progress and file selection dialog
    # - matlab spm opens a progress dialog
    # - if input is not specified, it also opens a file selection dialog
    #   via spm.select()

    # TODO? check if input is DCM object or filename?
    DCM = copy.deepcopy(P)

    M = {}
    if not 'M' in DCM:
        DCM['M'] = M
    if not 'pdcm' in DCM['options']:
        DCM['options']['pdcm'] = 0
    if not 'two_state' in DCM['options']:
        DCM['options']['two_state'] = 0
    if not 'stochastic' in DCM['options']:
        DCM['options']['stochastic'] = 0
    if not 'nonlinear' in DCM['options']:
        DCM['options']['nonlinear'] = 0
    if not 'centre' in DCM['options']:
        DCM['options']['centre'] = 0
    if not 'hidden' in DCM['options']:
        DCM['options']['hidden'] = []  # matlab: []
    if not 'hE' in DCM['options']:
        DCM['options']['hE'] = 6
    if not 'hC' in DCM['options']:
        DCM['options']['hC'] = 1 / 128
    if not 'n' in DCM:
        DCM['n'] = size(DCM['a'], 1)
    if not 'v' in DCM:
        DCM['v'] = size(DCM['Y']['y'], 1)

    # TODO? check for DCM['options']['nograph'] and set M['nograph']?
    if not 'maxit' in DCM['options']:
        if 'nN' in DCM['options']:
            DCM['options']['maxit'] = DCM['options']['nN']
            # matlab: warning about nN deprecation
        elif DCM['options']['stochastic']:
            DCM['options']['maxit'] = 32
        else:
            DCM['options']['maxit'] = 128

    if 'Nmax' in DCM['M']:
        M['Nmax'] = DCM['M']['Nmax']
    else:
        M['Nmax'] = DCM['options']['maxit']

    # check max nodes
    # --------------------------------------------------------------------------
    if not 'maxnodes' in DCM['options']:
        if 'nmax' in DCM['options']:
            DCM['options']['maxnodes'] = DCM['options']['nmax']
            # matlab: warning about nmax deprecation
        else:
            DCM['options']['maxnodes'] = 8

    # analysis and options
    # --------------------------------------------------------------------------
    DCM['options']['induced'] = 0

    # unpack
    # --------------------------------------------------------------------------
    U = DCM['U']   # exogenous inputs
    Y = DCM['Y']   # responses
    n = DCM['n']   # number of regions
    v = DCM['v']   # number of scans

    # detrend outputs (and inputs)
    # --------------------------------------------------------------------------
    Y['y'] = spm.detrend(Y['y'])
    if DCM['options']['centre']:
        U['u'] = spm.detrend(U['u'])

    # check scaling of Y (enforcing a maximum change of 4%
    # --------------------------------------------------------------------------
    scale = np.max(Y['y']) - np.min(Y['y'])
    scale = 4 / maximum(scale, 4)
    Y['y'] = Y['y'] * scale
    Y['scale'] = scale

    # check confounds (add constant if necessary)
    # --------------------------------------------------------------------------
    if not 'X0' in Y:
        Y['X0'] = ones((v, 1))
    if Y['X0'].ndim < 2:  # matlab: if ~size(Y.X0, 2) ...
        Y['X0'] = ones((v, 1))

    # fMRI slice time sampling
    # --------------------------------------------------------------------------
    if 'delays' in DCM:
        M['delays'] = DCM['delays']
    else:
        M['delays'] = ones((n, 1))
    if 'TE' in DCM:
        M['TE'] = DCM['TE']

    # create priors
    # ==========================================================================

    # check DCM['d'] (for nonlinear DCMs)
    # --------------------------------------------------------------------------
    # TODO? better to explicitly check for nonlinear option?
    if 'd' in DCM and DCM['d'].ndim > 2:
        DCM['options']['nonlinear'] = bool(DCM['d'].shape[2])
    else:
        DCM['d'] = zeros((n, n, 0))
        DCM['options']['nonlinear'] = 0

    # specify parameters for spm.integ_D (ensuring updates every second or so)
    # --------------------------------------------------------------------------
    if DCM['options']['nonlinear']:
        M['IS'] = 'spm.integ_D'
        M['nsteps'] = np.round(maximum(Y.dt, 1))
        # TODO: check if should start at 0 in python
        M['states'] = arange(1, n + 1)  # matlab: 1:n
    else:
        M['IS'] = 'spm.integ'

    # check for endogenous DCMs, with no exogenous driving effects
    # --------------------------------------------------------------------------
    if (DCM['c'] is None) or (U['u'] is None):  # matlab: isempty
        DCM['c'] = zeros((n, 1))
        DCM['b'] = zeros((n, n, 1))
        U['u'] = zeros((v, 1))
        U['name'] = ['null']
    if (not np.any(spm.vec(U['u']))) or (not np.any(spm.vec(DCM['c']))):
        DCM['options']['stochastic'] = 1

    # priors (and initial states)
    # --------------------------------------------------------------------------
    if DCM['options']['pdcm']:
        [pE, pC, x] = pydcm.models.pdcm_fmri_priors(DCM['a'], DCM['b'], DCM['c'], DCM['d'], DCM['options'])
    else:
        [pE, pC, x] = spm.dcm_fmri_priors(DCM['a'], DCM['b'], DCM['c'], DCM['d'], DCM['options'])
    msg = 'Using specified priors (any changes to DCM.a,b,c,d will be ignored)'

    # TODO: streamline this

    # initial parameters
    if 'P' in DCM['options']:
        M['P'] = DCM['options']['P']
    # prior expectation
    if 'pE' in DCM['options']:
        pE = DCM['options']['pE']
        print(msg)
    # prior covariance
    if 'pC' in DCM['options']:
        pC = DCM['options']['pC']
        print(msg)

    # initial parameters
    if 'P' in DCM['M']:
        M['P'] = DCM['M']['P']
    # prior expectation
    if hasattr(DCM['M'], 'pE'):
        pE = DCM['M']['pE']
        print(msg)
    # prior covariance
    if hasattr(DCM['M'], 'pC'):
        pC = DCM['M']['pC']
        print(msg)

    # eigenvector constraints on pC for large models
    # --------------------------------------------------------------------------
    if n > DCM['options']['maxnodes']:
        # TODO: test this

        # remove confounds and find principal (nmax) modes
        # ----------------------------------------------------------------------
        y = Y['y'] - Y['X0'] * (LA.pinv(Y['X0']) * Y['y'])
        V = spm.svd(y.T)
        V = V[:, arange(0, DCM['options']['maxnodes'])]

        # remove minor modes from priors on A
        # ----------------------------------------------------------------------
        j = arange(0, n * n)
        V = kron(V * V.T, V * V.T)
        pC[j, j] = V * pC[j, j] * V.T

    # hyperpriors over precision - expectation and covariance
    # --------------------------------------------------------------------------
    hE = zeros((n, 1)) + DCM['options']['hE']  # matlab: sparse(n,1)
    hC = wrappers.eye_fn(n) * DCM['options']['hC']  # matlab: speye(n,n)
    i = DCM['options']['hidden']
    # if `i' is an empty list, indexing with it doesn't return or set anything
    # TODO: make it more explicit?
    hE[i] = -4
    hC[i, i] = exp(-16)

    # complete model specification
    # --------------------------------------------------------------------------
    if DCM['options']['pdcm']:
        M['f'] = 'pydcm.models.fx_fmri_pdcm'   # equations of motion for p-dcm
    else:
        M['f'] = 'spm.fx_fmri'                 # equations of motion
    M['g'] = 'spm.gx_fmri'                     # observation equation
    M['x'] = x                                 # initial condition (states)
    M['pE'] = pE                               # prior expectation (parameters)
    M['pC'] = pC                               # prior covariance  (parameters)
    M['hE'] = hE                               # prior expectation (precisions)
    M['hC'] = hC                               # prior covariance  (precisions)
    M['m'] = U['u'].shape[1]
    M['n'] = size(x)
    M['l'] = x.shape[0]
    M['N'] = 64
    M['dt'] = 32 / M['N']
    M['ns'] = v

    # nonlinear system identification (nlsi)
    # ==========================================================================
    if not DCM['options']['stochastic']:

        # nonlinear system identification (Variational EM) - deterministic DCM
        # ----------------------------------------------------------------------
        Ep, Cp, Eh, F, *_ = spm.nlsi_GN(M, U, Y)

        # predicted responses (y) and residuals (R)
        # ----------------------------------------------------------------------
        y = eval(M['IS'])(Ep, M, U)
        R = Y['y'] - y
        R = R - Y['X0'] @ spm.inv(Y['X0'].T @ Y['X0']) @ (Y['X0'].T @ R)
        Ce = exp(-Eh)

    else:
        raise Exception('Stochastic DCM not implemented.')

    # Bilinear representation and first-order hemodynamic kernel
    # --------------------------------------------------------------------------
    M0, M1, L1, L2 = spm.bireduce(M, Ep, nout=4)
    H0, H1 = spm.kernels(M0, M1, L1, L2, M['N'], M['dt'], nout=2)

    # and neuronal kernels
    # --------------------------------------------------------------------------
    # matlab: L = sparse(1:n,(1:n) + 1,1,n,length(M0))
    L = wrappers.zeros_fn((n, max(M0.shape)))
    L[arange(n), arange(n) + 1] = 1
    K0, K1 = spm.kernels(M0, M1, L, M['N'], M['dt'], nout=2)

    # Bayesian inference and variance {threshold: prior mean plus T = 0}
    # --------------------------------------------------------------------------
    T = dense(spm.vec(pE))  # matlab: full(spm.vec(pE)
    # matlab: sw = warning('off','SPM:negativeVariance')
    with np.errstate(divide='ignore', invalid='ignore'):
        Pp = spm.unvec(1 - spm.Ncdf(T, np.abs(spm.vec(Ep)), np.diag(Cp).reshape(-1, 1)), Ep)
    Vp = spm.unvec(dense(np.diag(Cp)), Ep)  # matlab: full(diag(Cp))

    # matlab: try,  M = rmfield(M,'nograph'); end

    # Store parameter estimates
    # --------------------------------------------------------------------------
    DCM['M'] = M
    DCM['Y'] = Y
    DCM['U'] = U
    DCM['Ce'] = Ce
    DCM['Ep'] = Ep
    DCM['Cp'] = Cp
    DCM['Pp'] = Pp
    DCM['Vp'] = Vp
    DCM['H1'] = H1
    DCM['K1'] = K1
    DCM['R'] = R
    DCM['y'] = y
    DCM['T'] = 0

    # Data ID and log-evidence
    # ==========================================================================
    if 'FS' in M:
        # TODO: test/guess what could go wrong here
        try:
            ID = spm.data_id(eval(M['FS'])(Y['y'], M))
        except:
            ID = spm.data_id(eval(M['FS'])(Y['y']))
    else:
        ID = spm.data_id(Y['y'])

    # Save approximations to model evidence: negative free energy, AIC, BIC
    # --------------------------------------------------------------------------
    evidence = spm.dcm_evidence(DCM)
    DCM['F'] = F
    DCM['ID'] = ID
    DCM['AIC'] = evidence['aic_overall']
    DCM['BIC'] = evidence['bic_overall']

    # Save SPM version and revision number of code used
    # --------------------------------------------------------------------------
    # hardcoded for now
    DCM['version'] = {}
    DCM['version']['SPM'] = {}
    DCM['version']['DCM'] = {}
    DCM['version']['SPM']['version'] = 'SPM12'  # matlab: spm('Ver')
    DCM['version']['SPM']['revision'] = '7487'  # matlab: spm('Ver'), second output
    DCM['version']['DCM']['version'] = 'DCM12.5'  # matlab: spm.dcm_ui('Version')
    DCM['version']['DCM']['revision'] = '$Rev: 7479 $'  #  likely relevant for Friston's team

    # Save DCM
    # --------------------------------------------------------------------------
    # not for now
    # matlab: check if P is not struct, so it should be a filename or file handle
    # matlab: save(P,'DCM','F','Ep','Cp', spm.get_defaults('mat.format'))

    return DCM

def spm_dcm_evidence(DCM):
    """
    Compute evidence of DCM model
    FORMAT evidence = spm_dcm_evidence(DCM)

    DCM       - DCM data structure

    evidence  - structure with the following fields
      .region_cost(i)  - The cost of prediction errors in region i
      .bic_penalty     - Bayesian information criterion penalty
      .bic_overall     - The overall BIC value
      .aic_penalty     - Akaike's information criterion penalty
      .aic_overall     - The overall AIC value

    All of the above are in units of NATS (not bits).
    _________________________________________________________________________
    Copyright (C) 2008 Wellcome Trust Centre for Neuroimaging

    Will Penny
    $Id: spm_dcm_evidence.m 6856 2016-08-10 17:55:05Z karl $
    """

    evidence = {}

    # Only look at those parameters with non-zero posterior covariance
    # --------------------------------------------------------------------------
    v = DCM['v']  # number of samples
    n = DCM['n']  # number of regions
    wsel = nonzero(np.diag(DCM['Cp']))[0]

    # Look at costs of coding prediction errors by region
    # --------------------------------------------------------------------------
    # TODO: test/guess what could go wrong here
    evidence['region_cost'] = np.empty(n)
    for i in range(n):
        try:
            lambda_i = DCM['Ce'][i * v, i * v]  # Ce is error covariance
        except:
            try:
                lambda_i = DCM['Ce'][i]  # Ce is a hyperparameter
            except:
                lambda_i = DCM['Ce']  # Ce is the hyperparameter

        evidence['region_cost'][i] = - 0.5 * v * log(lambda_i)  \
                                  - 0.5 * DCM['R'][:, i].T @ ((1 / lambda_i) * eye(v)) @ DCM['R'][:, i]

    # Results
    # --------------------------------------------------------------------------
    evidence['aic_penalty'] = np.max(wsel.shape)
    evidence['bic_penalty'] = 0.5 * np.max(wsel.shape) * log(v)
    evidence['aic_overall'] = np.sum(evidence['region_cost']) - evidence['aic_penalty']
    evidence['bic_overall'] = np.sum(evidence['region_cost']) - evidence['bic_penalty']

    return evidence

