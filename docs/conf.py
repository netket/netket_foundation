"""Sphinx configuration for netket_foundation docs.

Shared defaults (theme, extensions, intersphinx, napoleon settings) come from
``neuralqxlab_sphinx_theme.conf_base``. That package lives in the sibling repo
``../neuralqxlab-sphinx-theme`` and is installed as a path dependency via the
``docs`` dependency group in ``pyproject.toml``. See ``docs/README.md`` for
build instructions.
"""

import sys
from pathlib import Path

from neuralqxlab_sphinx_theme.conf_base import *  # noqa: F401, F403
from neuralqxlab_sphinx_theme.linkcode import make_linkcode_resolve

# Make the local sphinx_extensions/ folder importable (flax_module directive).
sys.path.append(str(Path(__file__).parent / "sphinx_extensions"))

# Replace myst_parser with myst_nb (superset that adds .ipynb support), and add
# the flax_module directive (ported from netket/docs).
extensions = [e for e in extensions if e != "myst_parser"] + [  # noqa: F405
    "myst_nb",
    "flax_module.fmodule",
]

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

# -- Class docstring rendering (ported from netket/docs/conf.py) -------------
# Render the class signature separately from the class name (constructor params
# shown as their own block rather than merged into the header line).
autodoc_class_signature = "separated"
# Use only the class docstring (not __init__) as the class body.
autoclass_content = "class"
# Honour signatures embedded in docstrings, and inherit docstrings from parents.
autodoc_docstring_signature = True
autodoc_inherit_docstrings = True
# Cross-link types mentioned in docstrings and document PEP 526 class attributes.
napoleon_preprocess_types = True
napoleon_attr_annotations = True
# Generate autosummary stub pages for documented objects.
autosummary_generate = True

# Custom autosummary templates (ported from netket/docs). Classes use Sphinx's
# default template; ``flax_module_or_default.rst`` can be selected per-entry with
# ``:template:`` to specially render flax.linen Modules.
templates_path = ["_templates", "_templates/autosummary"]

# Notebooks are pre-executed and stored with outputs; don't re-run at build time.
nb_execution_mode = "off"
nb_execution_allow_errors = False

# Suppress warnings that originate from notebook cell formatting (transitions at
# section boundaries, non-consecutive heading levels) that we cannot easily fix
# without editing the notebook JSON.
suppress_warnings = ["docutils", "myst.header"]


# -- Hide undocumented __init__ (ported from netket/docs/conf.py) ------------
def autodoc_skip_member(app, what, name, obj, skip, options):
    # Ref: https://stackoverflow.com/a/21449475/
    exclusions = (
        "__weakref__",  # special-members
        "__doc__",
        "__module__",
        "__dict__",  # undoc-members
        "__new__",
    )
    exclude = name in exclusions
    if name == "__init__":
        exclude = obj.__doc__ is None
    return True if (skip or exclude) else None


def setup(app):
    app.connect("autodoc-skip-member", autodoc_skip_member)
