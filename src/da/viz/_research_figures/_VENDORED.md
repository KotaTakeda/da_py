# Vendored `research_figures`

This directory is a **verbatim, vendored copy** of the general-purpose
`research_figures` figure layer from
[`KotaTakeda/research-design-system`](https://github.com/KotaTakeda/research-design-system)
(`paper-figures/research_figures/`, tracked by that repo's issue #2).

It is vendored so that `da_py`'s `examples` extra stays self-contained: the
shared publication style, primitives, layouts, and atomic export utility are
reused **as-is**, without adding `research-design-system` as a runtime
dependency. This keeps `da_py` able to *use* the shared figure API while not
*depending* on the design-system package being installed.

Do not edit these modules to add domain logic. `research_figures` is
deliberately domain-independent (no PDE-/DA-specific plotting). All
`da_py`-specific figure functions live one level up in `da.viz` and in the
`examples/` scripts, built on top of these generic primitives.

To refresh this copy, re-copy the package from the design-system repo rather
than patching it in place.
