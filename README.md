# cbfax

A JAX library to compute control barrier functions related functionality


### Install

Requires the `dynamaxsys` library

```
pip install git+https://github.com/UW-CTRL/dynamaxsys.git
pip install git+https://github.com/UW-CTRL/cbfax.git
```

For editable mode:

```
git clone https://github.com/UW-CTRL/cbfax.git
cd cbfax
pip install -e .
```

### Pre-commit (Ruff Formatting)

Install pre-commit and set up the git hook:

```bash
pip install pre-commit
pre-commit install
```

Run on all files:

```bash
pre-commit run --all-files
```

### Automated Versioning

This repo now uses `setuptools-scm`, so package versions are derived from Git tags.
Packaging and version metadata are managed in `pyproject.toml`.
No manual `version=` edits are needed.

Create a release tag:

```bash
git tag v0.0.2
git push origin v0.0.2
```

Build with the computed version:

```bash
python -m build
```

Notes:

- Tagged commits build exactly that version (for example, `v0.0.2`).
- Untagged commits get an automatic development version.

### One-Command Releases

Use the Makefile helpers to create the next version tag automatically:

```bash
make release-next        # preview next patch tag (dry-run)
make release-patch       # create next patch tag locally
make release-minor       # create next minor tag locally
make release-major       # create next major tag locally
```

Create and push in one command:

```bash
make release-patch-push
make release-minor-push
make release-major-push
```

Then build:

```bash
make build
```


### TODOs
- tidy install instructions
- tidy plotting functionality
- remove dynamics.py