"""General layout helpers.

Domain-independent helpers for arranging panels, sharing colorbars, and
labelling subplots. Like :mod:`primitives`, nothing here is specific to any
research topic.
"""

from __future__ import annotations

from string import ascii_lowercase

import numpy as np

from .core import setup_figure


def single_panel(*, width: float | None = None, height: float | None = None, **kwargs):
    """Create a one-panel figure. Returns ``(fig, ax)``."""
    return setup_figure(width=width, height=height, **kwargs)


def multi_panel(
    nrows: int,
    ncols: int,
    *,
    panel_width: float = 3.1,
    panel_height: float = 3.15,
    sharex: bool = False,
    sharey: bool = False,
    layout: str | None = "constrained",
    **kwargs,
):
    """Create a grid of panels sized per-panel rather than per-figure.

    Returns ``(fig, axes)`` where ``axes`` is a flat 1D array in row-major
    order, so callers can ``zip`` it with their data without worrying about the
    2D shape. Unused trailing panels can be hidden with :func:`hide_unused`.

    ``layout`` defaults to ``"constrained"`` so titles, tick labels, and a
    :func:`shared_colorbar` do not overlap across rows; pass ``None`` to manage
    spacing manually.
    """
    fig, axes = setup_figure(
        figsize=(panel_width * ncols, panel_height * nrows),
        nrows=nrows,
        ncols=ncols,
        sharex=sharex,
        sharey=sharey,
        layout=layout,
        **kwargs,
    )
    axes = np.atleast_1d(axes).reshape(-1)
    return fig, axes


def hide_unused(axes, n_used: int) -> None:
    """Turn off axes beyond ``n_used`` (for ragged panel grids)."""
    for ax in np.asarray(axes).reshape(-1)[n_used:]:
        ax.axis("off")


def colorbar(mappable, ax, *, label: str | None = None, fraction: float = 0.046,
             pad: float = 0.04, **kwargs):
    """Attach a colorbar to a single axes.

    Thin wrapper over ``fig.colorbar`` with proportions that keep the axes
    square-ish. For a colorbar shared by several panels use
    :func:`shared_colorbar`.
    """
    cbar = ax.figure.colorbar(mappable, ax=ax, fraction=fraction, pad=pad, **kwargs)
    if label is not None:
        cbar.set_label(label)
    return cbar


def shared_colorbar(
    fig,
    mappable,
    axes,
    *,
    orientation: str = "horizontal",
    label: str | None = None,
    fraction: float = 0.045,
    pad: float = 0.06,
    **kwargs,
):
    """Attach a single colorbar shared across ``axes``.

    ``mappable`` is typically the ``AxesImage`` returned by
    :func:`~research_figures.primitives.image_plot`.
    """
    axes_list = np.asarray(axes).reshape(-1).tolist()
    cbar = fig.colorbar(
        mappable,
        ax=axes_list,
        orientation=orientation,
        fraction=fraction,
        pad=pad,
        **kwargs,
    )
    if label is not None:
        cbar.set_label(label)
    return cbar


def panel_labels(
    axes,
    *,
    labels=None,
    loc: str = "upper left",
    style: str = "({})",
    weight: str = "bold",
    offset: tuple[float, float] = (0.02, 0.98),
    **text_kwargs,
):
    """Add ``(a)``, ``(b)``, ... labels to a sequence of panels.

    Parameters
    ----------
    labels:
        Explicit label strings. Defaults to lowercase letters formatted with
        ``style`` (e.g. ``"(a)"``).
    loc:
        Either ``"upper left"``/``"upper right"``/``"lower left"``/
        ``"lower right"`` (uses ``offset`` in axes coordinates) — corner
        placement inside each panel.
    """
    axes = np.asarray(axes).reshape(-1)
    if labels is None:
        labels = [style.format(ascii_lowercase[i]) for i in range(len(axes))]

    corners = {
        "upper left": (offset[0], offset[1], "left", "top"),
        "upper right": (1 - offset[0], offset[1], "right", "top"),
        "lower left": (offset[0], 1 - offset[1], "left", "bottom"),
        "lower right": (1 - offset[0], 1 - offset[1], "right", "bottom"),
    }
    if loc not in corners:
        raise ValueError(f"unknown loc {loc!r}; expected one of {sorted(corners)}")
    x, y, ha, va = corners[loc]

    for ax, label in zip(axes, labels):
        ax.text(
            x,
            y,
            label,
            transform=ax.transAxes,
            ha=ha,
            va=va,
            fontweight=weight,
            **text_kwargs,
        )
    return axes


def downsample_for_display(array, max_lines: int, *, seed: int | None = 0):
    """Subsample rows of an ensemble array **for visualization only**.

    WARNING: visualization-only. This selects at most ``max_lines`` rows so a
    spaghetti plot stays legible. It must never be used to reduce an ensemble
    for numerical experiments, diagnostics, or any computed statistic — those
    must always use the full ensemble.

    Returns
    -------
    (subset, indices):
        The selected rows and the indices that were kept (into the original
        array), so a legend or annotation can reference them.
    """
    array = np.asarray(array)
    n = array.shape[0]
    if n <= max_lines:
        return array, np.arange(n)
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(n, size=max_lines, replace=False))
    return array[indices], indices
