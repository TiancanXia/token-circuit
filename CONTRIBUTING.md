# Contributing

[English](CONTRIBUTING.md) | [简体中文](CONTRIBUTING.zh-CN.md) · [Home](README.md)

## Setup

```sh
git clone https://github.com/TiancanXia/token-circuit.git
cd token-circuit
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS / Linux: source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python game.py
```

Python 3.10+ and Tkinter are required. Ruff and Pillow are development tools; they are not needed to play.

## Update and verify

Before editing:

```sh
git status
git fetch origin
git pull --ff-only
```

Preserve local changes before pulling. Use a branch for larger changes; resolve diverged history without overwriting others' work.

After editing:

```sh
python -m unittest discover -s tests -v
python check_ui.py
python -m ruff check tokenwalk game.py check_ui.py preview.py --select F
python -m ruff format --check tokenwalk
python game.py
```

Playtest affected controls and both languages. Model changes should also check cached/full-prefix equivalence.

To rebuild formula images, install LaTeX with dvipng, then run `python -m tools.build_attention_assets`. Edit the TeX blocks in `tokenwalk/content/attention_math.md`. TeX is not needed to play.

```sh
git diff --check
git diff
git add <changed-files>
git commit -m "Describe the change"
git push
```

External contributors: fork, create a branch and open a PR.

## Reference

- [Code structure](docs/architecture.md)
- [Model computation](docs/model-audit.md)

Update both language versions when changing documentation or UI text. English documents use the default filename; Chinese uses `.zh-CN.md`.

Keep credentials, caches and temporary files out of commits. Generated previews belong in `assets/previews`; README assets are tracked in `assets/showcase`.

Contributions must be compatible with MIT. Preserve third-party sources and licensing.
