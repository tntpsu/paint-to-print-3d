# AI Skin Cleanup Lane Plan

## Why This Exists

The repaired production lane can now produce smooth, watertight, flat-bottomed Bambu-ready assets. The remaining problem is paint-region noise: some provider outputs contain speckled texture, fabric-like shading, or tiny islands that explode into thousands of material components.

The spa duck sample is the current motivating case:

- Geometry: usable
- Visual idea: recognizable
- Bambu format: structurally valid
- Region cleanliness: rejected because tiny islands are far above threshold

## Principle

Keep the deterministic converter as the trusted print path. Add an optional cleanup lane before final region extraction, and require the same validation gates afterward.

The cleanup lane should never silently override the deterministic output. It should produce a candidate, compare against baseline, and let the lane chooser select only when it improves print-readiness without destroying intent.

## Candidate Pipeline

1. Source or repaired mesh enters the normal repaired production lane.
2. Baseline paint-intent report is written.
3. If the baseline fails for excessive tiny islands or component count, generate cleanup candidates.
4. Cleanup candidates can use one or more of:
   - color-space smoothing over adjacent faces
   - connected-component island absorption into the nearest large neighbor
   - palette-aware bilateral smoothing that avoids crossing strong semantic boundaries
   - zone-aware preservation for eyes, beak, towel, shield, stars, logos, and other intentional details
   - optional external/provider cleanup if an API returns a repaired textured mesh with cleaner paint regions
5. Re-run Bambu validation and paint-intent evaluation.
6. Choose the cleanest candidate only if it reduces islands/components and keeps required details.

## Acceptance Gates

- Watertight: required
- Single body: required for auto-ready
- Flat-bottom support: required for ducks
- Palette count: within requested max colors
- Tiny islands: under threshold or explicitly reviewed
- Connected component count: under threshold or explicitly reviewed
- Intent preservation: no loss of protected semantic details
- Comparison preview: generated for operator review

## First Local Implementation

Start with deterministic cleanup before adding a provider or ML dependency:

1. Build `paint_region_cleanup` as a report-only candidate after normal transfer.
2. Add connected-component absorption for tiny islands below a face-count threshold.
3. Add protected-label rules so eyes, beak, light neutral details, red/white shield sections, and high-contrast emblems are not swallowed.
4. Recompute palette/component stats and write a candidate report.
5. Let the lane chooser compare baseline vs cleanup candidate.

## External Cleanup / AI Provider Direction

Treat external cleanup as a separate provider-backed candidate, not as core logic. It may eventually help with:

- smoothing noisy texture/skin regions before palette extraction
- repairing ambiguous small marks into cleaner printable shapes
- turning generated star-like art into cleaner star silhouettes

Risks:

- loss of intentional detail
- invented marks that were not in the source
- hidden provider changes
- harder reproducibility

Required guardrail:

- Every provider cleanup candidate must be evaluated by the same paint-intent report and rejected if it increases drift, loses protected detail, or worsens print-readiness.

## Near-Term Work Items

- Add a `paint-region-cleanup` candidate mode.
- Add tests for tiny-island absorption with protected detail labels.
- Run Captain, Polo, Spa, Lacrosse, and Dapper ducks through baseline vs cleanup.
- Promote cleanup to default only after multiple real samples improve without manual correction.
