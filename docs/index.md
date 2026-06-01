---
myst:
  html_meta:
    description: netket_foundation — Foundation models and utilities for NetKet
---

# NetKet Foundation

:::{nkx-hero}
:wordmark: _static/logo-nameonly-transparent.webp
:wordmark-alt: NetKet Foundation
:tagline: Foundation models for Neural Wavefunctions.
:image: _static/hero-placeholder.svg
:primary-button: Get started | getting_started.html
:secondary-button: API reference | api/index.html

A NetKet extension for training and evaluating *foundation neural quantum states*
over families of Hamiltonians. Optimize a single model across many parameter
points, then probe it across an entire quantum phase diagram.
:::

::::{grid} 3
:gutter: 3

:::{grid-item-card} 🚀 Getting Started
:link: getting_started
:link-type: doc

Installation and a minimal working example.
:::

:::{grid-item-card} 💡 Examples
:link: https://github.com/NeuralQXLab/netket_foundation/tree/main/examples
:link-type: url

Step-by-step guides covering the main workflows.
:::

:::{grid-item-card} 📖 API Reference
:link: api/index
:link-type: doc

Full documentation of all public classes and functions.
:::
::::

```{toctree}
:hidden:
:maxdepth: 2
:caption: Documentation

getting_started
tutorials/index
api/index
```
