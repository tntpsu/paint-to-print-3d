# AGENTS

This repo is the reusable paint-to-print conversion layer for turning textured 3D models into cleaner Bambu-friendly grouped color assets.

## Development Loop

- Install locally with `python -m pip install -e ".[dev]"`.
- Run tests with `PYTHONPATH=src python -m pytest`.
- Run syntax verification with `PYTHONPATH=src python -m compileall -q src`.
- Keep generated conversion output under `outputs/` or `/tmp`; do not commit generated model artifacts unless they are intentional small fixtures.
- Prefer deterministic candidate lanes before adding model/provider-dependent cleanup.

## Architecture Boundaries

- `model_io.py`: load textured OBJ, OBJ ZIP, and GLB sources.
- `face_regions.py` and `regions.py`: geometry/texture segmentation primitives.
- `paint_cleanup.py`: deterministic post-label cleanup such as tiny island absorption.
- `pipeline.py`: conversion orchestration and asset writing.
- `production.py`: gated production bundles, acceptance summaries, and paint-intent reports.
- `export_obj.py` and `export_3mf.py`: standards-based printable asset exports.
- `validation.py`: Bambu material/topology compatibility checks.
- `cli.py`: thin CLI over package APIs.

## Safety Rules

- Do not hide nondeterministic AI/provider behavior inside the core production path.
- If adding an AI/provider cleanup, write it as a candidate with before/after metrics, validation gates, and a clear rejection reason.
- Do not make DuckAgent-specific workflow decisions here; this repo should expose reports/artifacts that DuckAgent can consume.
- Treat Bambu Studio as an external compatibility target. The internal source of truth is the generated report plus asset bundle.
- Preserve public-readiness: no secrets, private tokens, or large private sample dumps.

## Ship Expectations

- Add focused tests for new heuristics and output contracts.
- Update README or `docs/` when a command, report shape, or production gate changes.
- Keep report keys backward-compatible where possible; if semantics change, document them.
- Prefer compact, inspectable heuristics over clever code that future agents cannot reason about.
