import numpy as np
from scipy.stats import ortho_group


def dim_ens(E):
    """
    Dimension of ensemble of vectors E defined in Patil et al., PRL, 2001.
    Arguments:
        - E (d, k): ensemble of k vectors in d-dimension.
    """
    S = np.linalg.svd(E, compute_uv=False)
    return np.sum(S) ** 2 / np.sum(S**2)


def gen_vectors_with_svals(svals, d, k):
    """
    Generate ensemble of vectors E with prescribed singular values
    Args:
        - svals (float > 0)(d, ): singular values 
        - d (int): dimension of each vector
        - k (int): number of vectors
    Return:
        - E (d, k): k vectors in d-dimension possesing singular values `svals`
    """
    assert svals.shape[0] == min(d, k)
    S = np.diag(svals) # (min(d, k), min(d, k))

    if d >= k:
        U = ortho_group.rvs(d)[:, :k] # [u1, u2, ..., uk] (d, k)
        V = ortho_group.rvs(k) # [v1, v2, ..., vk] (k, k)
    else:
        U = ortho_group.rvs(d) # [u1, u2, ..., uk] (d, d)
        V = ortho_group.rvs(k)[:, :d] # [v1, v2, ..., vk] (k, d)

    E = U@S@V.T

    return E