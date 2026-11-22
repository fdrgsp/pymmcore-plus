# Codex

At the end of the edits, before committing/pushing, even if I do not ask, make
sure the changes are done in the best/efficient way, remove useless
comment/lines (also to reduce diff with upstream main), remove AI slop, avoid
duplication, write tests following the repo pattern and, as always, use Python
best practices.

After I ask to push, always monitor CI tests.

## Related repositories

This is the `cite` fork of `pymmcore-plus`, pinned by pymmcore-gui
(`github.com/fdrgsp/pymmcore-gui`, `[tool.uv.sources]` in its
`pyproject.toml`) along with these other repos, all from
`github.com/fdrgsp/<repo>`, branch `cite`:

| Repo | What it holds |
| --- | --- |
| pymmcore-gui | the app; pins every fork below in `[tool.uv.sources]` |
| pymmcore-widgets | Qt widgets: MDA editor (axis tables, autofocus, Surface Focus dialog), HCS wizard, stage/property/config widgets |
| useq-schema | `useq` models: MDA sequence, plans, autofocus, focus surface |
| ome-writers | OME-TIFF / OME-Zarr writers behind the MDA sink |
| ndv | n-dimensional image viewer |

A local clone of each one, on `cite`, normally sits next to this repo, in the
same parent folder (`../<repo>`). That is where to change them: fix code in the
repo that owns it rather than working around it here.

- Each repo has its own `.venv`. To use a local change from another repo:
  `uv pip install --python .venv/bin/python --no-deps -e ../<repo>`, then run
  with `uv run --no-sync` (`uv run`, `uv sync` and `uv lock` reinstall the
  pinned revision and silently drop the editable install).
- To ship a fork change: push its `cite` branch, then in pymmcore-gui run
  `uv lock --upgrade-package <package>` and commit `uv.lock`, the only lock
  that pins the forks' commits.
- CI for `cite` branches runs on `fdrgsp/<repo>`, not upstream: pass
  `-R fdrgsp/<repo>` to `gh`.

## pymmcore-gui branches

pymmcore-gui has three branches: `cite` is the main one and the base of
`cite-stitching` and `cite-unicore`. A general change (not specific to
stitching or UniMMCore) goes on `cite`, is merged into the other two, and is
pushed on all three, including a fork bump in `uv.lock`.
