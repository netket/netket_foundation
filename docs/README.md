# Building the docs

## Dependencies

The docs use a shared theme package, `neuralqxlab-sphinx-theme`, that lives in
a sibling directory:

```
Codes/Python/
├── netket_foundation/        ← this repo
│   └── docs/
└── neuralqxlab-sphinx-theme/ ← shared theme (separate repo)
```

The shared theme provides:
- The [Sphinx Book Theme](https://sphinx-book-theme.readthedocs.io) base
- A federation org bar (NeuralQXLab brand + packages dropdown) injected above
  the per-package header on every page
- Shared Sphinx configuration defaults (`conf_base.py`)

## Local setup

Both repos must be checked out as siblings. Then install the docs dependencies:

```bash
uv sync --group docs
```

This installs `neuralqxlab-sphinx-theme` as an editable path dependency
(see `pyproject.toml` → `[dependency-groups] docs`), so local changes to the
theme are reflected immediately without reinstalling.

## Building

```bash
# One-shot build
uv run --group docs make -C docs html

# Live-reload (watches both docs/ and the package source)
uv run --group docs make -C docs livehtml
```

Output lands in `docs/_build/html/`.

## Structure

```
docs/
├── conf.py              # Sphinx config — imports shared base, sets project name
├── index.md             # Landing page
├── getting_started.md   # Installation + minimal example
├── tutorials/
│   └── index.md         # Tutorial index (placeholder)
├── api/
│   └── index.md         # API reference index (autosummary)
└── Makefile
```

## Regenerating optimised logo assets

The sidebar and hero use pre-generated WebP/resized variants of the source logos in
`_static/`. If you update `logo.png` or `logo-transparent.png`, regenerate them with:

```bash
uv run python3 - << 'EOF'
from PIL import Image
from pathlib import Path

static = Path("docs/_static")

configs = [
    # (source, out_stem, width, format, save_kwargs)
    ("logo.png",             "logo",             None, "PNG",  {"optimize": True}),
    ("logo-transparent.png", "logo-transparent", None, "PNG",  {"optimize": True}),
    ("logo.png",             "logo",             None, "WEBP", {"quality": 85, "method": 6}),
    ("logo-transparent.png", "logo-transparent", None, "WEBP", {"quality": 85, "method": 6}),
    ("logo-transparent.png", "logo-nav",          200, "PNG",  {"optimize": True}),
    ("logo-transparent.png", "logo-nav",          200, "WEBP", {"quality": 90, "method": 6}),
]

for src, stem, width, fmt, kwargs in configs:
    img = Image.open(static / src).convert("RGBA")
    if width:
        ratio = width / img.width
        img = img.resize((width, int(img.height * ratio)), Image.LANCZOS)
    out = static / f"{stem}.{fmt.lower()}"
    img.save(out, format=fmt, **kwargs)
    print(f"{out.name:35s}  {img.size[0]}x{img.size[1]}  {out.stat().st_size/1024:.1f} KB")
EOF
```

| Output file | Usage |
|-------------|-------|
| `logo-nav.webp` / `logo-nav.png` | SBT sidebar logo (200 px wide) |
| `logo-transparent.webp` | Hero section (full size, WebP) |
| `logo.webp` | Generic use (no transparency) |

## Adding a new page

1. Create a `.md` file in the appropriate subdirectory.
2. Add it to the `toctree` in `index.md` (or the relevant section index).

## Updating the shared theme

The theme is managed in its own repo (`neuralqxlab-sphinx-theme`). To add a new
package to the federation navbar, edit `PACKAGES` in
`neuralqxlab_sphinx_theme/__init__.py` and bump the version. All package docs
pick up the change on their next build.
