"""Animation helpers.

Lightweight wrappers over :mod:`matplotlib.animation` for animating a time
sequence of 2D fields, plus a safe export utility mirroring
:mod:`~research_figures.export`. Animations are presentation/diagnostic
artifacts; publication figures should remain static panels
(:func:`~research_figures.layouts.multi_panel`).

GIF export uses Matplotlib's Pillow writer (requires the optional ``Pillow``
package); MP4 export requires ``ffmpeg`` on the PATH.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from matplotlib import animation as mpl_animation

from .core import setup_figure
from .primitives import image_plot


def animate_field(
    traj,
    *,
    times=None,
    extent=None,
    cmap=None,
    symmetric: bool = True,
    interval: int = 100,
    title_fmt: str = "t = {:.2f}",
    colorbar_label: str | None = None,
    figsize=(4.0, 4.0),
):
    """Animate a sequence of 2D fields as an evolving image.

    Parameters
    ----------
    traj:
        Array of shape ``(n_frames, ny, nx)``.
    times:
        Optional per-frame times used in the title via ``title_fmt``.
    symmetric:
        Use a symmetric color range from the 99th percentile of ``|traj|``
        across *all* frames, so the colormap is constant over the animation.
    interval:
        Delay between frames in milliseconds.

    Returns
    -------
    (fig, anim):
        The figure and the :class:`~matplotlib.animation.FuncAnimation`. Keep a
        reference to ``anim`` until it is saved or displayed.
    """
    traj = np.asarray(traj)
    if traj.ndim != 3 or len(traj) == 0:
        raise ValueError("traj must have shape (n_frames, ny, nx)")

    vmin = vmax = None
    if symmetric:
        v = float(np.percentile(np.abs(traj), 99.0))
        vmin, vmax = -v, v

    fig, ax = setup_figure(figsize=figsize)
    _, im = image_plot(
        traj[0], ax=ax, extent=extent, cmap=cmap, vmin=vmin, vmax=vmax
    )
    if colorbar_label is not None:
        fig.colorbar(im, ax=ax, label=colorbar_label, fraction=0.046)

    def update(i):
        im.set_data(traj[i])
        if times is not None:
            ax.set_title(title_fmt.format(times[i]))
        return (im,)

    update(0)
    anim = mpl_animation.FuncAnimation(
        fig, update, frames=len(traj), interval=interval, blit=False
    )
    return fig, anim


def save_animation(anim, path, *, fps: int = 10, dpi: int = 100) -> Path:
    """Atomically save an animation to ``path``.

    ``.gif`` uses the Pillow writer; other suffixes (e.g. ``.mp4``) use the
    ffmpeg writer. Like :func:`~research_figures.export.save_figure`, the file
    is written to a temporary sibling and moved into place.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.stem}.tmp{path.suffix}")
    if path.suffix.lower() == ".gif":
        writer = mpl_animation.PillowWriter(fps=fps)
    else:
        writer = mpl_animation.FFMpegWriter(fps=fps)
    try:
        anim.save(str(tmp_path), writer=writer, dpi=dpi)
        os.replace(tmp_path, path)
    except BaseException:
        Path(tmp_path).unlink(missing_ok=True)
        raise
    return path
