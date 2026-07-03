"""Domain plotting helpers for da_py.

These build on the shared, domain-independent figure layer in :mod:`da.viz`
(a vendored copy of ``research_figures`` from ``research-design-system``) so
that loss/RMSE curves use the common publication style and primitives instead
of ad hoc, per-script Matplotlib code. The numerical content (the loss values)
is computed here; only the drawing is delegated to the shared layer.
"""


def plot_loss(result1, result2, loss_func, x_index=None, ax=None, **kargs):
    """Plot a per-step loss/RMSE curve and return the computed loss.

    Signature and return value are unchanged from the original helper: ``loss``
    is ``loss_func(result1, result2)`` and callers may still pass their own
    ``ax``. The curve is drawn with :func:`da.viz.line_plot`. When ``ax`` is
    ``None`` a new figure is created under the shared publication style; when
    the caller supplies ``ax``, the axes' existing style is respected (open
    :func:`da.viz.style_context` around figure creation to get the publication
    style, as the examples do).
    """
    from da import viz

    assert len(result1) == len(result2)
    loss = loss_func(result1, result2)
    if x_index is not None:
        assert len(x_index) == len(loss)
        args = (x_index, loss)
    else:
        args = (loss,)
    if ax is None:
        with viz.style_context():
            viz.line_plot(*args, **kargs)
    else:
        viz.line_plot(*args, ax=ax, **kargs)
    return loss
