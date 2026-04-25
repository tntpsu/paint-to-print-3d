# AGENTS

This repo is intended to be a clean, public-facing extraction of reusable 3D color conversion logic.

## Principles

- Prefer small, documented modules over one-off experiment scripts.
- Keep source provenance clear when porting logic from private/internal repos.
- Favor standards-based exports over proprietary assumptions.
- Validate with tests whenever a new export or conversion step is added.
- Treat Bambu import behavior as an external compatibility target, not as the internal source of truth.

## Current Architecture Goals

- `model_io.py`: load textured geometry and resolve texture sources
- `regions.py`: convert texture imagery into stable printable regions
- `export_obj.py`: grouped material OBJ/MTL exports
- `export_3mf.py`: face-color `colorgroup` 3MF exports
- `pipeline.py`: high-level conversion orchestration
- `cli.py`: thin CLI over the package

## Public Repo Standards

- README should explain the problem and current limitations honestly.
- Tests should cover the exported file structure, not only happy-path return values.
- Avoid repo clutter from historical experiments.
- New code should include brief docstrings and stable type hints where useful.
