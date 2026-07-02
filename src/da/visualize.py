"""Domain plotting helpers for da_py.

These build on the shared, domain-independent figure layer in :mod:`da.viz`
(a vendored copy of ``research_figures`` from ``research-design-system``) so
that loss/RMSE curves use the common publication style and primitives instead
of ad hoc, per-script Matplotlib code. The numerical content (the loss values)
is computed here; only the drawing is delegated to the shared layer.
"""

from da import viz


def plot_loss(result1, result2, loss_func, x_index=None, ax=None, **kargs):
    """Plot a per-step loss/RMSE curve and return the computed loss.

    Signature and return value are unchanged from the original helper: ``loss``
    is ``loss_func(result1, result2)`` and callers may still pass their own
    ``ax``. The curve is now drawn with :func:`da.viz.line_plot` under the
    shared publication style, applied via a reversible ``style_context`` so it
    does not leak into unrelated figures.
    """
    assert len(result1) == len(result2)
    loss = loss_func(result1, result2)
    with viz.style_context():
        if x_index is None:
            viz.line_plot(loss, ax=ax, **kargs)
        else:
            assert len(x_index) == len(loss)
            viz.line_plot(x_index, loss, ax=ax, **kargs)
    return loss
