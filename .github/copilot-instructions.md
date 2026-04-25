# Copilot Instructions

This project converts textured 3D models into printable color-region assets.

- Keep changes deterministic unless explicitly adding a report-only provider/AI candidate.
- Do not commit files under `outputs/` or large generated model artifacts.
- Prefer small modules with focused tests.
- Run `PYTHONPATH=src python -m pytest` before suggesting a change is ready.
- Production readiness requires report gates, Bambu validation, and component/tiny-island metrics, not just a good-looking preview.
- See `AGENTS.md` and `docs/AI_DEVELOPMENT_GUIDE.md` before changing conversion policy.
