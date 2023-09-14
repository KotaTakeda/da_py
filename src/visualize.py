import matplotlib.pyplot as plt


def plot_loss(result1, result2, loss_func, ax=None, **kargs):
    assert len(result1) == len(result2)
    loss = loss_func(result1, result2)
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(loss, **kargs)
    return loss