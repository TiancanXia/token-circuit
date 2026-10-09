# Development and contributing

[English](CONTRIBUTING.md) | [简体中文](CONTRIBUTING.zh-CN.md) · [Home](README.md)

## Local setup

Clone once, then keep working in the same checkout:

```sh
git clone https://github.com/TiancanXia/token-circuit.git
cd token-circuit
python game.py
```

Runtime has no pip dependencies. Python 3.10+, Tkinter and a graphical desktop are required. Optional development tools include Ruff and Pillow for preview generation:

```sh
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS / Linux: source .venv/bin/activate
python -m pip install -r requirements-dev.txt
```

## Sync, edit, verify and publish

Before editing, inspect local state and remote updates:

```sh
git status
git fetch origin
git pull --ff-only
```

Pull only after preserving local changes and confirming the current branch can fast-forward. If history has diverged, inspect and merge or rebase deliberately. Do not force-push or reset away other work. Prefer a feature branch for larger changes.

After editing, run:

```sh
python -m unittest discover -s tests -v
python check_ui.py
python -m ruff check tokenwalk game.py check_ui.py preview.py --select F
python -m ruff format --check tokenwalk
python game.py
```

Automated UI checks do not replace a visible playtest. Check movement, entering and leaving maps, one F pass, H/C/L settings and opening/closing the handbook. Compare cached decoding against full-prefix computation when changing the model, and inspect both languages when changing copy.

Review the intended changes, then commit and push:

```sh
git diff --check
git diff
git add <changed-files>
git commit -m "Describe the change"
git push
```

Owners may maintain main; external contributors should fork, create a branch and open a PR. Check remote updates again before publishing and preserve both sides when resolving conflicts. There is no background auto-sync or auto-upload.

## Extension and documentation

Read the [architecture guide](docs/architecture.md) and [model notes](docs/model-audit.md). Register modules and interiors as data rather than adding scattered module checks to keyboard handlers. Update paired UI copy, player documentation and relevant verification with new features.

Public documentation under `docs` is tracked. Historical `docs/archive` and `docs/superpowers`, local agent notes, caches, virtual environments and bulk previews are ignored. Curated assets in `assets/showcase` are tracked. Do not commit credentials, personal paths or temporary files.

English is the default document language; paired Chinese documents use `.zh-CN.md`. Keep each language in its own file and maintain the language-switch links. The MIT license and changelog remain English-only.

Only contribute material you can license under MIT. Preserve the sources and licensing of third-party quotations, images and fonts.
