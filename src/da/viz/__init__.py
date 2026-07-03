"""da_py visualization layer.

This is a thin adapter over the general-purpose ``research_figures`` figure
layer from ``KotaTakeda/research-design-system``, which is vendored under
``da.viz._research_figures`` (see that directory's ``_VENDORED.md``).

The split is deliberate: ``da_py`` *uses* the shared, domain-independent figure
API (publication style, plot-kind primitives, panel layouts, atomic export) but
does not *depend* on the design-system package being installed, and keeps every
domain-specific figure function (vorticity panels, RMSE curves, ...) here or in
``examples/`` rather than pushing it into the shared layer.

Typical use in an example or notebook::

    from da import viz

    with viz.style_context():
        ax = viz.line_plot(times, rmse, label="ETKF")
        ax.set_xlabel("time")
        ax.set_ylabel("RMSE")
        ax.legend()
        viz.save_png(ax.figure, "data/rmse.png")

The generic primitives and helpers are re-exported below unchanged, with two
``da_py``-specific additions:

- :func:`vorticity_cmap`, a small colormap helper that keeps ``seaborn``
  optional;
- the default categorical color cycle is :data:`DEFAULT_COLOR_CYCLE`
  (``"earth_muted_natural"``) instead of the style file's own cycle: the
  ``style_context`` / ``setup_figure`` / ``single_panel`` / ``multi_panel``
  entry points below apply it, and every one accepts ``cycle=`` to pick
  another palette (or ``cycle=None`` for the plain publication cycle).
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

# Re-export the vendored, domain-independent figure API unchanged so callers
# can do ``from da import viz; viz.line_plot(...)`` without reaching into the
# private vendored package. The exported names are derived from the vendored
# package's own ``__all__`` so a refreshed vendored copy is re-exported here
# without having to keep a second hand-maintained list in sync.
from . import _research_figures as _research_figures
from ._research_figures import *  # noqa: F401,F403
from ._research_figures import __all__ as _RESEARCH_FIGURES_ALL

# da_py's default categorical cycle (see COLOR_CYCLES in the vendored
# palettes module). The wrappers below apply it; pass cycle=None to keep the
# publication style's own cycle.
DEFAULT_COLOR_CYCLE = "earth_muted_natural"


def style_context(name="publication", extra_rc=None, *, cycle=DEFAULT_COLOR_CYCLE):
    """:func:`research_figures.style_context` with da_py's default color cycle."""
    return _research_figures.style_context(name, extra_rc, cycle=cycle)


def _apply_cycle(fig_axes, cycle):
    # The vendored setup_figure re-enters its own publication style context at
    # figure creation, which resets axes.prop_cycle from the style file; apply
    # the requested cycle per axes afterwards so it survives that nesting.
    fig, axes = fig_axes
    if cycle is not None:
        for ax in np.ravel(axes):
            _research_figures.set_color_cycle(cycle, ax=ax)
    return fig, axes


def setup_figure(*, cycle=DEFAULT_COLOR_CYCLE, **kwargs):
    """:func:`research_figures.setup_figure` with da_py's default color cycle."""
    return _apply_cycle(_research_figures.setup_figure(**kwargs), cycle)


def single_panel(*, cycle=DEFAULT_COLOR_CYCLE, **kwargs):
    """:func:`research_figures.single_panel` with da_py's default color cycle."""
    return _apply_cycle(_research_figures.single_panel(**kwargs), cycle)


def multi_panel(nrows, ncols, *, cycle=DEFAULT_COLOR_CYCLE, **kwargs):
    """:func:`research_figures.multi_panel` with da_py's default color cycle."""
    return _apply_cycle(_research_figures.multi_panel(nrows, ncols, **kwargs), cycle)


@lru_cache(maxsize=None)
def vorticity_cmap(name: str = "icefire"):
    """Resolve a colormap for signed vorticity fields, keeping seaborn optional.

    The 2D NSE examples default to seaborn's diverging ``"icefire"`` map, but
    ``research_figures`` only requires ``matplotlib`` + ``numpy``. When
    ``name == "icefire"`` and seaborn is unavailable, degrade cleanly to the
    stock diverging ``"coolwarm"`` colormap. Any other name is passed through to
    Matplotlib as-is.
    """
    if name != "icefire":
        return name
    try:
        import seaborn as sns
    except ModuleNotFoundError:
        return "coolwarm"
    return sns.color_palette("icefire", as_cmap=True)


__all__ = [*_RESEARCH_FIGURES_ALL, "vorticity_cmap", "DEFAULT_COLOR_CYCLE"]
