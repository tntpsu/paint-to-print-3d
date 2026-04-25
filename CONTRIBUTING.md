# Contributing

Thanks for taking a look at `paint-to-print-3d`. The project is early, but contributions are welcome, especially around real-world sample validation, slicer compatibility, and clearer documentation.

## Good First Contributions

- Add a small reproducible fixture that covers a format or color-transfer edge case.
- Improve README examples or docs that were unclear.
- Add tests around OBJ, GLB, 3MF, or palette behavior.
- Improve reports that explain why an output is or is not print-ready.

## Development Setup

```bash
python -m pip install -e ".[dev]"
PYTHONPATH=src python -m pytest
```

Before opening a pull request, please run:

```bash
PYTHONPATH=src python -m compileall -q src
PYTHONPATH=src python -m pytest
```

## Project Boundaries

- Keep the core converter deterministic unless a change is explicitly marked as a candidate lane.
- Do not commit large generated model outputs, private source models, or API tokens.
- Prefer small, inspectable heuristics with tests over opaque one-off scripts.
- If a change affects output reports, update docs and tests in the same pull request.

## Reporting Compatibility Issues

If Bambu Studio, another slicer, or a model viewer imports an output incorrectly, please include:

- input format, for example OBJ, GLB, OBJ ZIP
- command used
- generated `conversion_report.json`
- screenshots if possible
- slicer/viewer version

Do not upload private or licensed model assets unless you have permission to share them.
