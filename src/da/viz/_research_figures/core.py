"""General figure setup utilities.

Domain-independent helpers for applying the publication style and creating
Matplotlib figures. Nothing here knows about PDEs, data assimilation, or any
specific research topic; that logic stays in the computational project that
owns the data.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

from .palettes import set_color_cycle

STYLE_DIR = Path(__file__).parent / "styles"
PUBLICATION_STYLE = STYLE_DIR / "publication.mplstyle"


def style_path(name: str = "publication") -> Path:
    """Return the path to a bundled ``.mplstyle`` file.

    Parameters
    ----------
    name:
        Style name without extension (e.g. ``"publication"``).
    """
    path = STYLE_DIR / f"{name}.mplstyle"
    if not path.exists():
        available = sorted(p.stem for p in STYLE_DIR.glob("*.mplstyle"))
        raise FileNotFoundError(
            f"unknown style {name!r}; available styles: {available}"
        )
    return path


def apply_style(
    name: str = "publication",
    extra_rc: dict | None = None,
    *,
    cycle=None,
) -> None:
    """Apply a bundled style globally, optionally overriding ``rcParams``.

    ``cycle`` selects the color cycle: a palette name from
    :data:`~research_figures.palettes.COLOR_CYCLES` (e.g. ``"okabe-ito"``,
    ``"nordic"``) or an explicit list of colors. When omitted the style file's
    own cycle is kept.

    This mutates the global Matplotlib state. Prefer :func:`style_context` when
    a temporary, reversible change is enough.
    """
    plt.style.use(str(style_path(name)))
    if extra_rc:
        mpl.rcParams.update(extra_rc)
    if cycle is not None:
        set_color_cycle(cycle)


@contextmanager
def style_context(
    name: str = "publication",
    extra_rc: dict | None = None,
    *,
    cycle=None,
):
    """Temporarily apply a bundled style within a ``with`` block.

    Accepts the same ``cycle`` selection as :func:`apply_style`. Restores the
    previous ``rcParams`` on exit, so it does not leak style into unrelated
    figures.
    """
    with plt.style.context(str(style_path(name))):
        if extra_rc or cycle is not None:
            with mpl.rc_context(extra_rc or {}):
                if cycle is not None:
                    set_color_cycle(cycle)
                yield
        else:
            yield


def setup_figure(
    *,
    width: float | None = None,
    height: float | None = None,
    figsize: tuple[float, float] | None = None,
    style: str | None = "publication",
    cycle=None,
    **subplots_kwargs,
):
    """Create a figure and axes using the publication style.

    Parameters
    ----------
    width, height:
        Figure size in inches. Convenience alternative to ``figsize``.
    figsize:
        Explicit ``(width, height)``. Takes precedence over ``width``/``height``.
    style:
        Bundled style name to apply, or ``None`` to leave the current style.
    cycle:
        Color cycle for the created axes: a palette name from
        :data:`~research_figures.palettes.COLOR_CYCLES` or an explicit list of
        colors. With a ``style`` it is applied inside the style context (so it
        wins over the style file's own cycle); with ``style=None`` it is set
        per created axes. ``None`` keeps the active cycle.
    **subplots_kwargs:
        Forwarded to :func:`matplotlib.pyplot.subplots` (``nrows``, ``ncols``,
        ``sharex``, ``gridspec_kw``, ...).

    Returns
    -------
    (fig, ax):
        The created figure and its axes, exactly as ``plt.subplots`` returns.
    """
    if figsize is None and (width is not None or height is not None):
        base_w, base_h = mpl.rcParams["figure.figsize"]
        figsize = (width or base_w, height or base_h)

    if style is not None:
        with style_context(style, cycle=cycle):
            return plt.subplots(figsize=figsize, **subplots_kwargs)
    fig, axes = plt.subplots(figsize=figsize, **subplots_kwargs)
    if cycle is not None:
        # fig.axes is exactly the axes just created, whatever shape
        # plt.subplots returned them in.
        for ax in fig.axes:
            set_color_cycle(cycle, ax=ax)
    return fig, axes
