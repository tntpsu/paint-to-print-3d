# Color Conversion Finish Plan

## Goal

Finish the textured-model-to-Bambu workflow in a way that is:

- technically reliable
- honest about what is already working
- explicit about what still needs research
- good enough to ship as a default DuckAgent lane

This plan is the practical endgame for the current problem.

It is based on the actual evidence from:

- the old `3dcolor` Grinch conversion
- `3dcolorconverter`
- DuckAgent cowgirl and bold-cowgirl runs
- Bambu Studio importer code
- Blender 3MF exporter code

## What We Know Now

### 1. The Grinch path really works

The old Grinch export is not a mystery anymore.

It succeeds because:

- it stays on the original textured OBJ mesh
- it uses UV-space sampling on that same mesh
- it posterizes and quantizes the texture
- it assigns one material label per face
- it exports grouped `usemtl` OBJ/MTL

Important implication:

- the successful Grinch path is primarily a **same-mesh face-region workflow**
- not a repaired-geometry transfer workflow

### 2. Bambu Studio is not the main blocker

Bambu Studio code inspection shows:

- OBJ import supports `vertex_colors`, `face_colors`, and `mtl_colors`
- 3MF import supports `m:colorgroup` and per-triangle color bindings

So the remaining failure is not “Bambu cannot read the file.”

The remaining failure is:

- the regions we generate are not clean enough on the hard cowgirl case

### 3. Cowgirl same-mesh is better than repaired transfer, but not enough

We already tested this.

The best same-mesh cowgirl legacy result:

- stays on the original low-face source mesh
- looks cleaner than repaired transfer
- but still does not look good enough to call solved

Important implication:

- repaired transfer is hurting us
- but repaired transfer is not the only issue
- source-art separability and region quality matter too

### 4. Grinch and cowgirl are not equivalent test cases

Grinch is easier because it has:

- high-contrast regions
- strong black/green/yellow separation
- tiny accent regions that are still clearly separable

Cowgirl is harder because it has:

- many warm tones close to one another
- more shading ambiguity
- weaker part separation in the source art

Important implication:

- the pipeline must either:
  - derive stronger semantic regions
  - or require more print-friendly source art
  - or both

### 5. “Finish” does not mean one algorithm must win every case

We should finish this by choosing a **production decision tree**, not by forcing one universal lane.

The likely production answer is:

- if same-mesh OBJ/OBJZIP conversion is good enough, use it
- if repaired geometry is required, use region-transfer only when it clears a quality bar
- otherwise stop and tell DuckAgent/operator to use the same-mesh export or regenerate the source concept with stronger paint zones

That is still a finished system.

## Definition Of Done

This project is done when all of the following are true.

### Technical done

- `3dcolorconverter` has a stable API for:
  - same-mesh conversion
  - repaired-geometry region transfer
  - grouped OBJ/MTL export
  - colorgroup 3MF export
  - validation report generation

### Product done

- DuckAgent can choose the correct conversion lane automatically
- the chosen lane is recorded in the run artifacts
- the user can see why that lane was chosen

### Visual done

- Grinch remains a clean known-good example
- cowgirl same-mesh output is clearly better than the current repaired-transfer baseline
- repaired transfer is either:
  - visibly good enough to use
  - or automatically rejected in favor of the safer same-mesh lane

### Operational done

- the system does not silently ship a camouflage-like repaired transfer
- a bad conversion produces a clear recommendation:
  - retry with same-mesh export
  - regenerate concept with stronger contrast
  - or hand off for manual cleanup

## Non-Goals

These are not required to call this finished:

- perfectly reproducing Meshy internals
- perfectly reproducing 3D AI Studio internals
- making repaired-geometry transfer win on every artistic source
- solving arbitrary photorealistic texture conversion

## The Real Problem To Solve

The problem is not:

- “how do we export color?”

The problem is:

- “how do we preserve intentional paint regions when the source and target geometry differ?”

That is a region-representation problem.

The wrong objects to transfer are:

- raw pixels
- raw vertex colors
- raw nearest labels alone

The right object to transfer is closer to:

- connected face regions
- region ownership
- region confidence
- semantic part bias

## The Finish Strategy

We should solve this in three production lanes.

### Lane A: Same-Mesh Oracle Lane

This is the path that already works best today.

Input:

- source textured OBJ or OBJZIP
- or source GLB converted to the same-mesh legacy path

Pipeline:

1. UV mask
2. seam padding
3. posterize / flatten shading
4. sample source colors on the same mesh
5. quantize to target palette
6. assign one label per face
7. export grouped OBJ/MTL
8. export colorgroup 3MF

Use this lane when:

- the provider gives us OBJ/OBJZIP
- or the single-source GLB path is visually better than repaired transfer
- or repaired transfer fails the quality bar

Production meaning:

- this is the safe fallback
- it may be lower-poly
- but it preserves paint regions best

### Lane B: Repaired-Geometry Region Transfer Lane

This is the hard lane, and it should remain conditional until it is clearly good enough.

Input:

- source mesh with trusted same-mesh regions
- repaired target mesh

Pipeline:

1. extract clean source face labels
2. build connected source regions
3. infer anchor candidates for:
   - body
   - beak
   - hat
   - bandana
   - lower-body / boot / dark-accent zones
4. transfer source region ownership to target faces
5. clean islands on target geometry
6. smooth boundaries without merging major parts
7. export grouped OBJ/MTL and colorgroup 3MF

Use this lane when:

- repaired geometry is materially better for manufacturing
- and the result clears the visual/region quality threshold

Production meaning:

- this is the preferred lane only after it proves itself
- until then it is an upgrade candidate, not the default winner

### Lane C: Regenerate For Printability

This is the source-art quality lane.

Input:

- concept prompt

Pipeline:

1. request a more print-friendly palette
2. request stronger part separation
3. prefer flatter part colors
4. reduce muddy warm-tone overlaps

Use this lane when:

- same-mesh is still too mottled
- repaired transfer is worse
- the concept itself is not region-friendly enough

Production meaning:

- some creative outputs should be regenerated, not tortured through the converter

## Concrete Execution Phases

## Phase 0: Build The Learning Ladder

Goal:

- stop treating the hard duck case as the first place we learn the math

This phase exists to build understanding from trivial to hard.

We should not move straight from “cube works” to “cowgirl should work.”
We should climb a benchmark ladder where each step isolates one new difficulty.

### Fixture ladder

#### Fixture A: Six-color cube

Shape:

- simple cube
- one flat distinct color per face

Why:

- proves exact face labeling
- proves OBJ/MTL export structure
- proves colorgroup 3MF structure
- proves Bambu imports the expected six regions

Questions this answers:

- do we preserve exact face regions with zero ambiguity
- do we accidentally merge across hard geometric boundaries

#### Fixture B: Cube with smiley face on one side

Shape:

- same cube
- one face has a small high-contrast smiley or eye/mouth detail

Why:

- tests whether small localized features survive
- gives us a tiny “Grinch-eye-like” benchmark without duck complexity

Questions this answers:

- what is the minimum accent size we can preserve
- are tiny dark regions lost during quantization or smoothing

#### Fixture C: Banded sphere or rounded capsule

Shape:

- rounded object
- 3 or 4 clean color bands

Why:

- tests curved-surface region preservation
- removes seams/duck semantics while introducing smooth geometry

Questions this answers:

- does curvature alone break face labeling
- do small triangles on curved surfaces over-fragment

#### Fixture D: Simple toy duck with obvious parts

Shape:

- very simple duck
- body, beak, hat, bandana as explicit flat regions

Why:

- first semantic-part benchmark
- much simpler than real cowgirl styling

Questions this answers:

- can our anchor logic preserve major duck parts
- can repaired transfer keep beak/body/hat separated on a recognizable duck shape

#### Fixture E: Grinch

Shape:

- real known-good case

Why:

- this is the first real-world oracle we already trust

Questions this answers:

- does the new implementation still preserve the old winning behavior

#### Fixture F: Cowgirl original

Shape:

- current difficult real-world case

Why:

- tests same-mesh vs repaired transfer under subtle warm-tone palettes

Questions this answers:

- how much is geometry transfer hurting us
- how much is source palette separability hurting us

#### Fixture G: Cowgirl bold

Shape:

- print-friendly regeneration of cowgirl

Why:

- tests whether stronger source-art separation materially improves the pipeline

Questions this answers:

- when is regeneration a better answer than more conversion effort

### Rules for the ladder

For every fixture above, we should test:

1. same-mesh lane
2. repaired-transfer lane, if a repaired mesh exists
3. grouped OBJ/MTL import behavior
4. colorgroup 3MF import behavior

We should not move up the ladder until the lower rung is explained.

That does not mean “perfect.”
It means:

- we know why it worked
- or we know why it failed
- and the failure is documented clearly enough to predict the next case

### Asset generation strategy

For the synthetic fixtures, we can use whichever is fastest and most controllable:

- hand-built local geometry
- Blender
- or 3D AI Studio if it can generate the exact simple prototypes we want

The important thing is not the source tool.
The important thing is that the benchmark geometry and colors are intentionally simple and repeatable.

Exit condition:

- we have a controlled curriculum from trivial to hard
- each new algorithm can be evaluated on that full ladder

## Phase 1: Lock The Benchmark Harness

Goal:

- stop evaluating by memory

Implement:

- one benchmark manifest with fixed cases:
  - cube
  - cube with smiley
  - banded sphere
  - simple toy duck
  - Grinch known-good
  - cowgirl original
  - cowgirl bold
- one report per run with:
  - source format
  - face count
  - palette size
  - region count
  - top region sizes
  - tiny-island count
  - rendered comparison image

Deliverables:

- stable fixture paths
- one benchmark runner in `3dcolorconverter`

Exit condition:

- every new change can be compared against the same evidence pack

## Phase 2: Make Same-Mesh The Official Oracle

Goal:

- make the best current path explicit and production-ready

Implement:

- promote `legacy_fast_face_labels` and Blender-like same-mesh bake paths to first-class benchmark lanes
- add regression assertions for Grinch:
  - palette size
  - material count
  - per-material face counts
- add same-mesh cowgirl regression outputs

Deliverables:

- same-mesh Grinch must remain stable
- same-mesh cowgirl must be easy to regenerate and inspect

Exit condition:

- we have a trusted baseline that does not depend on repaired transfer

## Phase 3: Improve Region Representation Before Transfer

Goal:

- transfer better regions, not just better labels

Implement:

- formal `RegionTransferSource` object
- include:
  - face labels
  - connected components
  - component sizes
  - palette
  - centroid and normal summaries
  - anchor guesses
  - optional UV bounding statistics

Files:

- `src/color3dconverter/face_regions.py`
- `src/color3dconverter/pipeline.py`

Exit condition:

- source-region state can be inspected directly and reused across transfer strategies

## Phase 4: Make Repaired Transfer Region-First

Goal:

- stop smearing same-mesh success into repaired-geometry failure

Implement:

1. target-face candidate scoring
- nearest source region centroid
- nearest source face centroid
- normal similarity
- optional local density weighting

2. anchor-aware bias
- prefer body region on central bulk
- prefer beak on forward protruding warm-orange region
- prefer hat on upper/head-adjacent region
- prefer bandana on neck/front band region

3. target cleanup
- adjacency-connected components
- merge tiny islands into dominant neighbors
- preserve strong boundary edges

4. boundary rules
- do not let hat and body collapse if both have strong anchor support
- do not let beak merge into face/body because of warm-tone similarity alone

Exit condition:

- repaired transfer visibly preserves the major paint parts on cowgirl

## Phase 5: Add Acceptance Gates

Goal:

- choose the best lane automatically and safely

Implement a lane chooser:

1. run same-mesh lane
2. run repaired-transfer lane if repaired geometry exists
3. compare:
  - region coherence
  - tiny island count
  - palette practicality
  - visual drift
4. choose:
  - repaired lane if clearly better
  - otherwise same-mesh lane

Important:

- this is where we “finish” the workflow even if repaired transfer is not universally best

Exit condition:

- DuckAgent no longer needs a human guess to pick the lane

## Phase 6: Add Regeneration Guidance

Goal:

- avoid feeding unprintable concepts into the converter

Implement:

- concept scoring for printability
- detect:
  - low palette separation
  - muddy warm-tone clustering
  - excessive highlight/shadow dependence
- return rewrite guidance like:
  - “increase body/beak contrast”
  - “make bandana more saturated”
  - “separate hat brim from crown”

Exit condition:

- bad source art is caught before wasting time on conversion

## Phase 7: Ship The Production Decision Tree

Goal:

- make this a real finished feature in DuckAgent

Production logic:

1. If provider OBJ/OBJZIP exists:
- run same-mesh oracle lane first

2. If repaired geometry exists:
- run repaired region-transfer lane too

3. Compare both lanes:
- if repaired lane passes quality threshold and is clearly better, use it
- otherwise use same-mesh lane

4. If both lanes are poor:
- recommend concept regeneration with printability guidance

5. Always expose:
- chosen lane
- rejected lane summary
- validation image
- operator notes

Exit condition:

- DuckAgent has one safe and explainable end-to-end behavior

## Code Work Order

This is the implementation order I would follow.

1. Build the synthetic fixture ladder:
   - six-color cube
   - cube with smiley
   - banded sphere
   - simple toy duck
2. Add benchmark harness and fixture manifest
3. Lock Grinch exact regression in `3dcolorconverter`
4. Add same-mesh cowgirl regression pack
5. Formalize `RegionTransferSource`
6. Improve target-face scoring for region transfer
7. Add stronger anchor constraints for:
   - hat brim
   - hat crown
   - beak
   - bandana
   - lower-body / boots
8. Add acceptance-gate lane chooser
9. Wire DuckAgent to select the winning lane automatically
10. Add printability-regeneration hints

## Success Metrics

We should track these for every benchmark case.

### Region metrics

- palette size
- connected region count
- number of tiny islands
- size of smallest intentional accent region

### Comparison metrics

- mean pixel drift
- dominant region overlap with source preview
- manual visual grade:
  - unusable
  - poor
  - workable
  - good

### Production metrics

- did same-mesh beat repaired transfer
- did repaired transfer beat same-mesh
- did the system choose the correct lane

## Stop/Go Rules

### Stop doing this

- do not keep adding generic k-means tweaks without benchmark proof
- do not assume repaired geometry must be the winning lane
- do not judge progress only from one duck
- do not skip from a trivial success directly to a hard artistic duck without a middle benchmark

### Keep doing this

- compare against Grinch every time
- compare against the simple fixture ladder every time
- compare same-mesh vs repaired every time
- preserve visual boards for each change
- treat same-mesh success as a real product option, not a temporary hack

## Honest Current Best Bet

If I had to finish this with the highest chance of success, I would bet on:

1. shipping a strong same-mesh oracle lane first
2. making repaired transfer conditional, not mandatory
3. improving repaired transfer through region ownership and stronger semantic anchors
4. using concept regeneration when the palette itself is too muddy to print well

That path gives us a real finished system fastest while still leaving room to keep improving repaired geometry later.
