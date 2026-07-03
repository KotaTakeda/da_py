"""research_figures: a thin, reusable layer for publication-quality figures.

This package provides domain-independent building blocks for scientific
figures across research repositories. It deliberately contains no simulation
logic and no PDE-/DA-/HMC-specific plotting functions; those belong in the
computational project that owns the data.

Typical use::

    from research_figures import apply_style, line_plot, save_pdf

    apply_style("publication")             # optionally: cycle="okabe-ito"
    ax = line_plot(times, rmse, label="ETKF")
    ax.set_xlabel("time")
    ax.set_ylabel("RMSE")
    ax.legend()
    save_pdf(ax.figure, "figures/rmse.pdf")

The pieces: ``core`` (styles), ``palettes`` (selectable color cycles),
``primitives`` (plot-kind functions returning ``ax``, or ``(ax, im)`` for
image-like plots), ``layouts`` (panels, colorbars, labels), ``export``
(atomic PDF/PNG saving), and ``animation`` (2D field animation). Each
primitive has a runnable sample under ``examples/`` with a rendered preview
in ``examples/preview/``.

Policy: research-result figures, numerical experiment plots, and figures
reproduced from supplied papers or slides should be used as provided and must
not be redrawn with these components by default. See
``research-design-system/common/policies/source-integrity.md``.
"""

from .animation import animate_field, save_animation
from .core import (
    PUBLICATION_STYLE,
    apply_style,
    setup_figure,
    style_context,
    style_path,
)
from .export import save_figure, save_pdf, save_png
from .palettes import (
    COLOR_CYCLES,
    get_color_cycle,
    list_color_cycles,
    property_cycle,
    set_color_cycle,
)
from .layouts import (
    colorbar,
    downsample_for_display,
    hide_unused,
    multi_panel,
    panel_labels,
    shared_colorbar,
    single_panel,
)
from .primitives import (
    heatmap_plot,
    histogram_plot,
    image_plot,
    line_plot,
    line_plot_3d,
    loglog_plot,
    scatter_plot,
    scatter_plot_3d,
    vector_field_plot,
)

__all__ = [
    # core
    "apply_style",
    "style_context",
    "style_path",
    "setup_figure",
    "PUBLICATION_STYLE",
    # export
    "save_figure",
    "save_pdf",
    "save_png",
    # palettes
    "COLOR_CYCLES",
    "list_color_cycles",
    "get_color_cycle",
    "set_color_cycle",
    "property_cycle",
    # primitives
    "line_plot",
    "scatter_plot",
    "loglog_plot",
    "image_plot",
    "heatmap_plot",
    "histogram_plot",
    "vector_field_plot",
    "line_plot_3d",
    "scatter_plot_3d",
    # animation
    "animate_field",
    "save_animation",
    # layouts
    "single_panel",
    "multi_panel",
    "hide_unused",
    "colorbar",
    "shared_colorbar",
    "panel_labels",
    "downsample_for_display",
]
