import matplotlib.pyplot as plt


def plot_loss(result1, result2, loss_func, x_index=None, ax=None, **kargs):
    assert len(result1) == len(result2)
    loss = loss_func(result1, result2)
    if ax is None:
        fig, ax = plt.subplots()
    if x_index is None:
        ax.plot(loss, **kargs)
    else:
        assert len(x_index) == len(loss)
        ax.plot(x_index, loss, **kargs)
    return loss