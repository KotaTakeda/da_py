"""Figure export utilities.

Reproducible, side-effect-conscious saving of Matplotlib figures. The atomic
writer is adapted from ``save_figure_safely`` in
``KotaTakeda/da_py`` (``examples/nse2d_torus_forecast.py``): it writes to a
temporary file, optionally verifies it, then atomically replaces the target so
a partial or corrupt file never appears at the destination path.

Export rules favour reproducible publication outputs and avoid hidden state:
callers pass explicit paths and formats; nothing is inferred from a global
"current figure".
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt


def _verify_png(path: Path) -> None:
    """Best-effort integrity check for a written PNG.

    Silently skipped when Pillow is unavailable so export never hard-depends on
    an optional package.
    """
    try:
        from PIL import Image
    except ModuleNotFoundError:
        return
    with Image.open(path) as img:
        img.verify()


def save_figure(
    fig,
    path,
    *,
    dpi: int | None = None,
    bbox_inches: str | None = "tight",
    pad_inches: float = 0.02,
    facecolor: str = "white",
    transparent: bool = False,
    close: bool = True,
    verify: bool = True,
) -> Path:
    """Atomically save ``fig`` to ``path``.

    The parent directory is created if needed. The figure is written to a
    temporary sibling file and then ``os.replace``-d into place, so readers
    never observe a half-written file. For PNG targets the temporary file is
    verified with Pillow when available.

    Parameters
    ----------
    fig:
        The Matplotlib figure to save.
    path:
        Destination path. The suffix determines the format (``.pdf``, ``.png``,
        ``.svg``, ...).
    dpi:
        Resolution for raster formats. Defaults to the current
        ``savefig.dpi`` rcParam.
    close:
        Close the figure after saving (default ``True``) to release memory.
    verify:
        Verify PNG output integrity when Pillow is available.

    Returns
    -------
    Path:
        The path that was written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Keep the real extension so Matplotlib infers the format; the ".tmp"
    # marker goes in front of it (e.g. "fig.tmp.pdf").
    fmt = path.suffix.lstrip(".") or None
    tmp_path = path.with_name(f"{path.stem}.tmp{path.suffix}")
    try:
        fig.savefig(
            tmp_path,
            format=fmt,
            dpi=dpi,
            bbox_inches=bbox_inches,
            pad_inches=pad_inches,
            facecolor=facecolor,
            transparent=transparent,
        )
        if verify and path.suffix.lower() == ".png":
            _verify_png(tmp_path)
        os.replace(tmp_path, path)
    except BaseException:
        Path(tmp_path).unlink(missing_ok=True)
        raise
    finally:
        if close:
            plt.close(fig)
    return path


def save_pdf(fig, path, **kwargs) -> Path:
    """Save ``fig`` as a vector PDF (preferred for publication)."""
    path = Path(path).with_suffix(".pdf")
    return save_figure(fig, path, **kwargs)


def save_png(fig, path, *, dpi: int = 300, **kwargs) -> Path:
    """Save ``fig`` as a raster PNG (useful for slides and previews)."""
    path = Path(path).with_suffix(".png")
    return save_figure(fig, path, dpi=dpi, **kwargs)
