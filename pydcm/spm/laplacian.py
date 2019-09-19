# laplacian

__all__ = ['spm_DEM_embed', 'spm_DEM_eval', 'spm_DEM_eval_diff',
           'spm_DEM_M_set', 'spm_DEM_qU', 'spm_DEM_R', 'spm_DEM_set',
           'spm_LAP', 'spm_LAP_eval', 'spm_LAP_pg', 'spm_LAP_ph']

def spm_DEM_embed(Y, n, t, dt=1, d=0):
    raise Exception('spm_DEM_embed: not implemented')


def spm_DEM_eval(M, qu, qp, nout=2):
    raise Exception('spm_DEM_eval: not implemented')


spm_DEM_eval.D = None


def spm_DEM_eval_diff(x, v, qp, M, bilinear=1):
    raise Exception('spm_DEM_diff: not implemented')


def spm_DEM_M_set(M):
    raise Exception('spm_DEM_M_set: not implemented')


def spm_DEM_qU(qU, pU):
    raise Exception('spm_DEM_qU: not implemented')


def spm_DEM_R(n, s, form='Gaussian'):
    raise Exception('spm_DEM_R: not implemented')


def spm_DEM_set(DEM, nout=1):
    raise Exception('spm_DEM_set: not implemented')


def spm_LAP(DEM):
    raise Exception('spm_DEM_LAP: not implemented')


def spm_LAP_eval(M, qu, qh, nout=1):
    raise Exception('spm_LAP_eval: not implemented')


def spm_LAP_pg(x, v, h, M):
    raise Exception('spm_LAP_pg: not implemented')


def spm_LAP_ph(x, v, h, M):
    raise Exception('spm_LAP_ph: not implemented')
