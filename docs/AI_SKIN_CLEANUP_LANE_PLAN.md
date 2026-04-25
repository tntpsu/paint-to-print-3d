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

Status: implemented as a deterministic repaired-production candidate.

1. `paint_region_cleanup` runs after normal repaired transfer when component or tiny-island counts exceed trigger thresholds.
2. Connected-component absorption rewrites small unprotected islands into neighboring labels.
3. Protected-label rules preserve beak-like regions, light neutral details, dark eye-like marks, and saturated emblem-like regions.
4. The candidate writes fresh grouped OBJ/MTL, 3MF, palette, preview, labels, and a `paint_cleanup` report.
5. The repaired-production lane promotes the candidate only when it improves component/tiny-island counts and still passes transfer plus Bambu validation.

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

- Run Captain, Polo, Lacrosse, and Dapper ducks through baseline vs cleanup.
- Tune the default cleanup threshold from real samples instead of only synthetic fixtures.
- Add a visual comparison board for baseline vs cleanup candidates.
- Promote cleanup to default only after multiple real samples improve without manual correction.

## First Real Smoke Result

Spa duck cleanup smoke:

- Baseline components: `2,615`
- Cleanup candidate components: `270`
- Baseline tiny islands: `2,518`
- Cleanup candidate tiny islands: `237`
- Decision: candidate written, not promoted
- Reason: tiny islands still exceed the auto threshold of `96`

This is the intended fail-closed behavior. The lane is useful, but it needs another cleanup pass or stronger texture/skin simplification before Spa-like samples should become automatic.
