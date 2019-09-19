import numpy as np
import scipy as sp
import scipy.io


def loadmat_dcm(filename):
   mat = loadmat(filename)
   if 'DCM' in mat:
       dcm = mat['DCM']
   else:
       raise Exception('No \'DCM\' attribute in mat file.')
   return dcm


def loadmat(filename):
    '''
    Load a mat file and convert it into nested dictionaries for convenience.

    References:
    https://stackoverflow.com/questions/7008608/scipy-io-loadmat-nested-structures-i-e-dictionaries
    https://pyhogs.github.io/reading-mat-files.html
    '''
    mat = sp.io.loadmat(filename, struct_as_record=False, squeeze_me=False)
    return _check_type(mat)


def _check_type(x):

    if isinstance(x, np.ndarray) and x.size == 1:
        # squeeze singleton arrays
        x = x.item()

    if isinstance(x, sp.io.matlab.mio5_params.mat_struct):
        x = _iterate_matstruct(x)
    elif isinstance(x, np.ndarray) and x.dtype.type is np.object_:
        x = x.squeeze()
        x = _iterate_ndarray(x)
        if x.ndim == 1:
            x = x.tolist()
    elif isinstance(x, dict):
        x =_iterate_dict(x)
    elif sp.sparse.issparse(x):
        # convert sparse arrays to dense
        x = x.toarray()

    return x


def _iterate_ndarray(x):
    for idx in range(x.size):
        x.ravel()[idx] = _check_type(x.ravel()[idx])
    return x


def _iterate_dict(x):
    for key, val in x.items():
        x[key] = _check_type(val)
    return x


def _iterate_matstruct(x):
    r = {}
    for key in x._fieldnames:
        val = x.__dict__[key]
        r[key] = _check_type(val)
    return r
