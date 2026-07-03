"""Named color cycles and property-cycle helpers.

Selectable categorical palettes for the ``axes.prop_cycle``. The vocabulary
combines:

- the default publication cycle from ``styles/publication.mplstyle``;
- palettes collected in ``KotaTakeda/visualizations``
  (``matplotlib/notebooks/demo_styles.ipynb``): ``probnum``, ``sophisticated``,
  ``nordic``, ``blues``, and the Tableau 10 cycle;
- the Okabe-Ito colorblind-safe palette;
- Paul Tol qualitative schemes: ``tol_muted_extended`` and
  ``high_contrast_scientific``;
- ``earth_muted_natural``, a muted natural/earth-tone cycle;
- a grayscale cycle for print-only figures (combine with marker/linestyle
  variation via :func:`property_cycle`).

Palettes are plain color lists, so adding a new one is a one-line edit.
"""

from __future__ import annotations

import matplotlib as mpl
from cycler import cycler

COLOR_CYCLES: dict[str, list[str]] = {
    # Default cycle of styles/publication.mplstyle (token roles: primary,
    # secondary, reference, alert, then supporting hues).
    "publication": [
        "#1f4e79", "#b3541e", "#3c7a3c", "#a01a2f",
        "#6a4c93", "#7f7f7f", "#17807e", "#b58b00",
    ],
    # Okabe-Ito, the standard colorblind-safe 8-color palette.
    "okabe-ito": [
        "#0072B2", "#D55E00", "#009E73", "#CC79A7",
        "#E69F00", "#56B4E9", "#F0E442", "#000000",
    ],
    # Tableau 10 (as used in the demo_styles notebook).
    "tableau10": [
        "#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14E",
        "#EDC949", "#B07AA2", "#FF9DA7", "#9C755F", "#BAB0AC",
    ],
    # From KotaTakeda/visualizations demo_styles.ipynb.
    "probnum": [
        "#107D79", "#FF9933", "#1F77B4", "#D62728", "#9467BD",
        "#8C564B", "#E377C2", "#7F7F7F", "#BCBD22", "#17BECF",
    ],
    "sophisticated": [
        "#107D79", "#FF9933", "#2E5A88", "#E7298A", "#7570B3",
        "#66A61E", "#E6AB02", "#A6761D", "#666666", "#444444",
    ],
    "nordic": [
        "#5E81AC", "#BF616A", "#A3BE8C", "#D08770", "#B48EAD",
        "#88C0D0", "#EBCB8B", "#81A1C1", "#4C566A", "#8FBCBB",
    ],
    # Paul Tol's muted qualitative scheme, extended with its light gray
    # (rose, indigo, sand, green, cyan, wine, teal, olive, purple, gray).
    "tol_muted_extended": [
        "#CC6677", "#332288", "#DDCC77", "#117733", "#88CCEE",
        "#882255", "#44AA99", "#999933", "#AA4499", "#DDDDDD",
    ],
    # Muted natural/earth tones (dark blue-green, teal, sand, orange,
    # terracotta, muted purple, dusty rose, slate blue, sage, ochre).
    "earth_muted_natural": [
        "#264653", "#2A9D8F", "#E9C46A", "#F4A261", "#E76F51",
        "#6D597A", "#B56576", "#355070", "#8AB17D", "#BC6C25",
    ],
    # High-contrast picks drawn from Paul Tol's high-contrast/bright/vibrant
    # schemes (deep blue, gold, muted red, black, teal, purple, light gray,
    # orange, blue, vermilion).
    "high_contrast_scientific": [
        "#004488", "#DDAA33", "#BB5566", "#000000", "#44AA99",
        "#AA4499", "#AAAAAA", "#EE7733", "#0077BB", "#CC3311",
    ],
    # Sequential blues (single-hue ramp; suited to ordered series).
    "blues": ["#011f4b", "#03396c", "#005b96", "#6497b1", "#b3cde0"],
    # Grayscale for print-only output; pair with markers/linestyles.
    "grayscale": ["#000000", "#4d4d4d", "#7f7f7f", "#b0b0b0"],
}


def list_color_cycles() -> list[str]:
    """Return the available palette names."""
    return sorted(COLOR_CYCLES)


def get_color_cycle(name: str) -> list[str]:
    """Return the color list of a named palette."""
    try:
        return list(COLOR_CYCLES[name])
    except KeyError:
        raise KeyError(
            f"unknown color cycle {name!r}; available: {list_color_cycles()}"
        ) from None


def _resolve_colors(cycle) -> list[str]:
    if isinstance(cycle, str):
        return get_color_cycle(cycle)
    return list(cycle)


def set_color_cycle(cycle, *, ax=None) -> list[str]:
    """Set the color cycle globally or on a single axes.

    Parameters
    ----------
    cycle:
        A palette name from :data:`COLOR_CYCLES` or an explicit list of colors.
    ax:
        When given, only that axes' property cycle is changed; otherwise the
        global ``axes.prop_cycle`` rcParam is updated.

    Returns
    -------
    list[str]:
        The resolved colors.
    """
    colors = _resolve_colors(cycle)
    if ax is None:
        mpl.rcParams["axes.prop_cycle"] = cycler(color=colors)
    else:
        ax.set_prop_cycle(color=colors)
    return colors


def property_cycle(colors="publication", *, markers=None, linestyles=None):
    """Compose a property cycle varying color, marker, and linestyle together.

    Series that differ in all three properties stay distinguishable in
    grayscale print and for color-impaired readers. The cycle length is the
    shortest of the given sequences.

    Example
    -------
    >>> ax.set_prop_cycle(property_cycle(
    ...     "okabe-ito",
    ...     markers=["o", "s", "^", "D"],
    ...     linestyles=["-", "--", "-.", ":"],
    ... ))
    """
    props: list[tuple[str, list]] = [("color", _resolve_colors(colors))]
    if markers is not None:
        props.append(("marker", list(markers)))
    if linestyles is not None:
        props.append(("linestyle", list(linestyles)))

    n = min(len(values) for _, values in props)
    combined = None
    for key, values in props:
        part = cycler(**{key: values[:n]})
        combined = part if combined is None else combined + part
    return combined
