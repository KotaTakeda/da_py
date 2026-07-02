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

The generic primitives and helpers are re-exported below unchanged; the only
``da_py``-specific addition is :func:`vorticity_cmap`, a small colormap helper
that keeps ``seaborn`` optional.
"""

from __future__ import annotations

from functools import lru_cache

# Re-export the vendored, domain-independent figure API unchanged so callers
# can do ``from da import viz; viz.line_plot(...)`` without reaching into the
# private vendored package. The exported names are derived from the vendored
# package's own ``__all__`` so a refreshed vendored copy is re-exported here
# without having to keep a second hand-maintained list in sync.
from ._research_figures import *  # noqa: F401,F403
from ._research_figures import __all__ as _RESEARCH_FIGURES_ALL


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


__all__ = [*_RESEARCH_FIGURES_ALL, "vorticity_cmap"]
