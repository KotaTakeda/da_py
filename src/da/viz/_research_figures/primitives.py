"""General plotting primitives.

Thin wrappers over Matplotlib that apply consistent publication defaults while
staying completely domain-independent. There are no PDE-, data-assimilation-,
or HMC-specific APIs here: functions are named after the *kind of plot*
(``line_plot``, ``image_plot``, ``loglog_plot``), never after the research
quantity being plotted.

Every function accepts an optional ``ax``; when omitted a new figure/axes pair
is created. Extra keyword arguments are forwarded to the underlying Matplotlib
call so callers keep full control.

Return-value convention: primitives return the ``ax`` they drew on, except the
image-like primitives (``image_plot``, ``heatmap_plot``), which return
``(ax, im)`` because the mappable is needed for a colorbar
(:func:`~research_figures.layouts.colorbar` or
:func:`~research_figures.layouts.shared_colorbar`).
"""

from __future__ import annotations

import numpy as np

from .core import setup_figure


def _ensure_ax(ax):
    if ax is None:
        _, ax = setup_figure()
    return ax


def _ensure_ax3d(ax):
    if ax is None:
        _, ax = setup_figure(subplot_kw={"projection": "3d"})
    return ax


def line_plot(x, y=None, *, ax=None, label=None, **kwargs):
    """Plot one or more lines.

    If ``y`` is ``None``, ``x`` is treated as the y-values against an implicit
    integer index. Returns the axes.
    """
    ax = _ensure_ax(ax)
    if y is None:
        ax.plot(x, label=label, **kwargs)
    else:
        ax.plot(x, y, label=label, **kwargs)
    return ax


def scatter_plot(x, y, *, ax=None, label=None, **kwargs):
    """Scatter plot of ``y`` versus ``x``."""
    ax = _ensure_ax(ax)
    ax.scatter(x, y, label=label, **kwargs)
    return ax


def histogram_plot(
    data,
    *,
    ax=None,
    bins="auto",
    density: bool = False,
    label=None,
    alpha: float = 0.6,
    edgecolor: str = "white",
    linewidth: float = 0.4,
    **kwargs,
):
    """Histogram of a 1D sample.

    Defaults favour overlaying several distributions on one axes: translucent
    bars with thin white edges. Pass ``density=True`` to compare samples of
    different sizes on a common scale.
    """
    ax = _ensure_ax(ax)
    ax.hist(
        np.asarray(data).ravel(),
        bins=bins,
        density=density,
        label=label,
        alpha=alpha,
        edgecolor=edgecolor,
        linewidth=linewidth,
        **kwargs,
    )
    return ax


def loglog_plot(x, y, *, ax=None, label=None, base: float = 10.0, **kwargs):
    """Log-log line plot, e.g. for spectra or convergence rates."""
    ax = _ensure_ax(ax)
    ax.plot(x, y, label=label, **kwargs)
    ax.set_xscale("log", base=base)
    ax.set_yscale("log", base=base)
    return ax


def image_plot(
    field,
    *,
    ax=None,
    extent=None,
    origin: str = "lower",
    cmap=None,
    vmin=None,
    vmax=None,
    symmetric: bool = False,
    interpolation: str = "nearest",
    aspect: str = "equal",
    **kwargs,
):
    """Display a 2D field as an image.

    Parameters
    ----------
    symmetric:
        When ``True`` and ``vmin``/``vmax`` are not given, choose a symmetric
        range ``[-v, v]`` where ``v`` is the 99th percentile of ``|field|``.
        Useful for signed fields (vorticity, differences) with a diverging
        colormap, but the colormap choice itself is left to the caller.

    Returns
    -------
    (ax, im):
        The axes and the ``AxesImage`` (pass ``im`` to a colorbar helper).
    """
    ax = _ensure_ax(ax)
    field = np.asarray(field)
    if symmetric and vmin is None and vmax is None:
        v = float(np.percentile(np.abs(field), 99.0))
        vmin, vmax = -v, v
    im = ax.imshow(
        field,
        extent=extent,
        origin=origin,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation=interpolation,
        aspect=aspect,
        **kwargs,
    )
    return ax, im


def heatmap_plot(
    matrix,
    *,
    ax=None,
    cmap=None,
    vmin=None,
    vmax=None,
    xticklabels=None,
    yticklabels=None,
    **kwargs,
):
    """Display a matrix as a heatmap with categorical axes.

    Unlike :func:`image_plot`, cells map to discrete rows/columns and pixels are
    not interpolated, which suits parameter sweeps and confusion-matrix-style
    tables. Returns ``(ax, im)``.
    """
    ax = _ensure_ax(ax)
    matrix = np.asarray(matrix)
    im = ax.imshow(
        matrix,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        aspect="auto",
        interpolation="nearest",
        origin="upper",
        **kwargs,
    )
    if xticklabels is not None:
        ax.set_xticks(np.arange(len(xticklabels)))
        ax.set_xticklabels(xticklabels)
    if yticklabels is not None:
        ax.set_yticks(np.arange(len(yticklabels)))
        ax.set_yticklabels(yticklabels)
    return ax, im


def line_plot_3d(x, y, z, *, ax=None, label=None, **kwargs):
    """3D line plot, e.g. for a phase-space trajectory.

    ``ax`` must be a 3D axes when given (create one with
    ``setup_figure(subplot_kw={"projection": "3d"})``); otherwise a new 3D
    figure is created.
    """
    ax = _ensure_ax3d(ax)
    ax.plot(x, y, z, label=label, **kwargs)
    return ax


def scatter_plot_3d(x, y, z, *, ax=None, label=None, **kwargs):
    """3D scatter plot on a 3D axes (created when ``ax`` is omitted)."""
    ax = _ensure_ax3d(ax)
    ax.scatter(x, y, z, label=label, **kwargs)
    return ax


def vector_field_plot(
    x,
    y,
    u,
    v,
    *,
    ax=None,
    scale=None,
    stride: int = 1,
    **kwargs,
):
    """Quiver plot of a vector field ``(u, v)`` sampled on ``(x, y)``.

    Accepts both quiver coordinate forms: 2D ``x``/``y`` (e.g. from
    ``meshgrid``) or 1D coordinate vectors with 2D ``u``/``v``. ``stride``
    subsamples the grid for legibility, slicing each array along its own
    dimensions so the two forms stay consistent. This subsampling is a
    visualization convenience only and must not be relied upon for any
    numerical quantity.
    """
    ax = _ensure_ax(ax)
    x = np.asarray(x)
    y = np.asarray(y)
    u = np.asarray(u)
    v = np.asarray(v)
    if stride > 1:
        # Slice coordinates and field components by their own ndim: with 1D
        # x/y and 2D u/v, striding only x.ndim axes of u/v would leave the
        # column dimension full length and mismatch the arrow positions.
        xy_sl = (slice(None, None, stride),) * x.ndim
        uv_sl = (slice(None, None, stride),) * u.ndim
        x, y = x[xy_sl], y[(slice(None, None, stride),) * y.ndim]
        u, v = u[uv_sl], v[(slice(None, None, stride),) * v.ndim]
    ax.quiver(x, y, u, v, scale=scale, **kwargs)
    ax.set_aspect("equal")
    return ax
