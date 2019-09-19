import numpy as np
from .spm import linalg

zeros_fn = np.zeros
eye_fn = np.eye
expm_fn = linalg.spm_expm
expm_mult_fn = linalg.spm_expm
