def diag(F, A):
    import numpy as np
    Fp = A.T @ F @ A  # F prime
    eps, Cp = np.linalg.eigh(Fp)
    C = A @ Cp
    return eps, C