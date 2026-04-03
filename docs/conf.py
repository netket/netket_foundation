"""Sphinx configuration for netket_foundation docs.

Shared defaults (theme, extensions, intersphinx, napoleon settings) come from
``neuralqxlab_sphinx_theme.conf_base``. That package lives in the sibling repo
``../neuralqxlab-sphinx-theme`` and is installed as a path dependency via the
``docs`` dependency group in ``pyproject.toml``. See ``docs/README.md`` for
build instructions.
"""

from pathlib import Path

from neuralqxlab_sphinx_theme.conf_base import *  # noqa: F401, F403
from neuralqxlab_sphinx_theme.linkcode import make_linkcode_resolve

# Replace myst_parser with myst_nb (superset that adds .ipynb support)
extensions = [e for e in extensions if e != "myst_parser"] + ["myst_nb"]  # noqa: F405

project = "netket_foundation"
copyright = "2024, NeuralQXLab"
author = "NeuralQXLab"
release = "0.1.0"

html_context = {
    **html_context,  # noqa: F405
    "github_repo": "netket_foundation",
}

html_theme_options = {
    **html_theme_options,  # noqa: F405
    "logo": {
        "image_light": "_static/logo-nav.webp",
        "image_dark": "_static/logo-nav.webp",
        "alt": "NetKet Foundation",
    },
}

html_title = "netket_foundation"
html_favicon = "_static/favicon.ico"

html_static_path = ["_static"]


exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "README.md"]

linkcode_resolve = make_linkcode_resolve(
    github_repo="netket_foundation",
    repo_root=Path(__file__).parent.parent,
)

# netket_foundation uses Google-style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = False
