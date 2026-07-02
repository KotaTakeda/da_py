"""Smoke tests for the da.viz figure layer and its use in da.visualize.

These check that the vendored ``research_figures`` layer is wired up correctly
and that ``plot_loss`` keeps its original signature and return value while
drawing through the shared primitives. Matplotlib is required; the tests are
skipped when it is unavailable, mirroring the examples.
"""

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from da import viz  # noqa: E402
from da.loss import loss_rms  # noqa: E402
from da.visualize import plot_loss  # noqa: E402


def test_bundled_publication_style_is_available():
    assert viz.style_path().name == "publication.mplstyle"
    assert viz.PUBLICATION_STYLE.exists()


def test_reexports_generic_primitives():
    # A representative slice of the re-exported research_figures API.
    for name in ("line_plot", "image_plot", "multi_panel", "shared_colorbar",
                 "panel_labels", "save_png", "style_context"):
        assert hasattr(viz, name), name


def test_line_plot_creates_and_draws():
    ax = viz.line_plot(np.arange(5), np.arange(5) ** 2)
    assert len(ax.lines) == 1
    plt.close(ax.figure)


def test_vorticity_cmap_passthrough_and_fallback():
    # Non-icefire names pass straight through.
    assert viz.vorticity_cmap("viridis") == "viridis"
    # icefire resolves to a colormap object (seaborn) or the coolwarm fallback.
    cmap = viz.vorticity_cmap("icefire")
    assert cmap == "coolwarm" or isinstance(cmap, matplotlib.colors.Colormap)


def test_plot_loss_preserves_signature_and_return():
    a = np.linspace(0.0, 1.0, 12).reshape(12, 1) * np.ones((12, 3))
    b = a + 0.1
    fig, ax = viz.single_panel()
    loss = plot_loss(a, b, loss_rms, ax=ax, label="etkf", lw=0.3)
    assert np.asarray(loss).shape == (12,)
    # Loss was drawn on the provided axes.
    assert len(ax.lines) == 1
    plt.close(fig)


def test_plot_loss_creates_figure_when_ax_missing():
    a = np.linspace(0.0, 1.0, 6).reshape(6, 1) * np.ones((6, 2))
    b = a + 0.5
    before = set(plt.get_fignums())
    loss = plot_loss(a, b, loss_rms)
    new_figs = set(plt.get_fignums()) - before
    assert np.asarray(loss).shape == (6,)
    # A standalone call creates its own (publication-styled) figure.
    assert len(new_figs) == 1
    fig = plt.figure(new_figs.pop())
    assert len(fig.axes[0].lines) == 1
    plt.close(fig)


def test_plot_loss_accepts_x_index():
    a = np.random.default_rng(0).standard_normal((8, 2))
    b = np.random.default_rng(1).standard_normal((8, 2))
    x_index = np.arange(8) * 2
    fig, ax = viz.single_panel()
    loss = plot_loss(a, b, loss_rms, x_index=x_index, ax=ax)
    assert len(loss) == 8
    np.testing.assert_allclose(ax.lines[0].get_xdata(), x_index)
    plt.close(fig)


def test_save_png_is_atomic_and_verifies(tmp_path):
    fig, ax = viz.single_panel()
    viz.line_plot(np.arange(3), ax=ax)
    out = viz.save_png(fig, tmp_path / "curve.png")
    assert out.exists()
    # No leftover temporary sibling from the atomic writer.
    assert not (tmp_path / "curve.tmp.png").exists()
