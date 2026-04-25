# Examples

This folder is for small, shareable example assets and walkthrough notes.

## Recommended Example Set

For a public release, the most helpful example bundle is:
- one textured OBJ duck
- one packaged OBJ ZIP bundle
- one GLB duck
- one repaired-geometry plus transferred-color example report

Each example should ideally include:
- the input asset
- the conversion command used
- the resulting `conversion_report.json`
- the resulting `region_colorgroup.3mf`
- the resulting preview images

## Example Commands

### Textured OBJ

```bash
python -m color3dconverter.cli convert-model \
  ./examples/sample_duck/model.obj \
  --regions 6 \
  --out-dir ./examples/sample_duck/out
```

### OBJ ZIP

```bash
python -m color3dconverter.cli convert-model \
  ./examples/sample_duck_bundle.zip \
  --regions 6 \
  --out-dir ./examples/sample_duck_bundle_out
```

### GLB

```bash
python -m color3dconverter.cli convert-model \
  ./examples/sample_duck.glb \
  --regions 8 \
  --out-dir ./examples/sample_duck_glb_out
```

## Notes

- Keep examples small enough to be practical in a public repo.
- Prefer ducks or simple stylized characters over heavy photoreal assets.
- Include at least one comparison/validation bundle so users can see how the source preview and export preview are evaluated.
