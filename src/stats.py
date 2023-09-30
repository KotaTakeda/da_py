def cov(Ens, axis=0):
    """
    Compute ensemble covariance
    """
    assert len(Ens.shape) == 2
    dEns = Ens - Ens.mean(axis=axis)
    m = Ens.shape[axis]
    if axis == 0:
        return dEns.T@dEns/(m-1)
    else:
        return dEns@dEns.T/(m-1)
