# T1 — DiffAugment, instance noise and adaptive augmentation (training-side, D inputs)

Workstream owner: Roey. Status: **seed 0 complete for all 6 conditions; seed-1 repeats
(baseline, translation, ADA) queued.**
Portable evidence: this report, [results CSV](diffaugment-results.csv),
[export-check CSV](diffaugment-export-check.csv), [workstream ledger](diffaugment-ledger.jsonl),
figures: [metrics](figures/diffaugment-metrics.png), [digit classes](figures/diffaugment-classes.png),
[samples](figures/diffaugment-samples.png),
[what-D-sees figure](figures/diffaugment-what-d-sees.png). Full runs stay on Juliet under
`~/a2-diffaug/repo/runs/` (ignored by Git, per the [artifact policy](README.md)).

## Idea

The fixed MLP discriminator sees raw flattened pixels, so it can separate real from
generated images by memorising exact pixel positions or exact values (real MNIST
backgrounds are exactly −1; a tanh generator never quite reaches it). Its feedback then
stops being useful to G. All methods below change **only what D sees**; the generator,
discriminator, losses and optimiser are unchanged.

- **DiffAugment** (Zhao et al., NeurIPS 2020): the same kind of random, differentiable
  transform is applied to real and generated images every time they enter D, in both
  the D and the G update, so G still receives gradients. MNIST-adapted policies
  (`diffaugment.py`): translation by up to 4 px, cutout of a 14×14 square, and
  brightness/contrast; padding and cutout use the background value −1, not 0.
  No flips or rotations (they change digit identity).
What D sees under each transform, on real training digits
([figure](figures/diffaugment-what-d-sees.png), `scripts/diffaug_explainer_figure.py`):
translation and brightness/contrast keep the digit recognisable; cutout (14×14) often
removes the strokes that define it (an 8 can become a "J" or "?"), and combining all
three is harsher still.

- **Instance noise** (Sønderby et al. 2017; Arjovsky & Bottou 2017): Gaussian noise
  (initial std 0.1, annealed linearly to 0) added to all D inputs.
- **Adaptive augmentation** (ADA idea, Karras et al. 2020): each transform is applied
  with probability p; every 4 steps p moves up if `mean(sign(D(real)−0.5)) > 0.6`
  (D over-confident on reals), down otherwise; p can go 0→1 within 60,000 images.

## Protocol

Same as the shared baseline: 10 epochs, batch 64, Adam 2e-4, training seed 0,
generation seed 1, batch 256, starter export (negative pixels clipped), frozen
[local-MNIST64-k3-v1](local-metrics.md) evaluator. Runs on Juliet (A100), via
`scripts/diffaug_study.sh` (one Slurm array task per condition, then evaluation).
The baseline is re-run on the same GPU so all conditions share hardware (its run is
recorded in the shared `ledger.jsonl` on Juliet; the five augmented runs in the workstream ledger).
`scripts/baseline.py` gained `--augment`, `--instance-noise`, `--ada-target`; with
none of them it behaves exactly as before.

| Condition | Flags | Training (s) | Final p |
|---|---|---:|---:|
| baseline | — | 165 | — |
| noise0.1 | `--instance-noise 0.1` | 167 | — |
| da-translation | `--augment translation` | 179 | 1 |
| da-translation-cutout | `--augment translation,cutout` | 180 | 1 |
| da-translation-cutout-color | `--augment translation,cutout,color` | 183 | 1 |
| ada-translation-cutout | `--augment translation,cutout --ada-target 0.6` | 180 | 0.29 |

## Results

Frozen local-MNIST64-k3-v1 metrics, training seed 0, generation seed 1, σ=1, starter
export ([CSV](diffaugment-results.csv), [figure](figures/diffaugment-metrics.png)):

| Condition | FD-MNIST ↓ | Precision ↑ | Recall ↑ | Largest digit class |
|---|---:|---:|---:|---:|
| baseline | 502.51 | **57.25%** | 0.00% | "5": 38% |
| instance noise 0.1 | 632.42 | 51.72% | 0.00% | "5": 31% |
| DA: translation | 232.82 | 55.65% | 0.17% | "1": 32% |
| **DA: translation + cutout** | **95.98** | 43.29% | 4.14% | "1": 16% |
| DA: translation + cutout + colour | 123.24 | 46.51% | 3.98% | "1": 21% |
| ADA: translation + cutout | 375.98 | 44.47% | **5.77%** | "1": 41% |

Real-vs-real control: FD 0.47, precision 89.5%, recall 90.7%.

**Interpretation (one training seed — see limitations).**

- **DiffAugment works on the fixed MLP**: translation alone more than halves FD (503 → 233)
  at similar precision; adding cutout cuts FD by a further factor of 2.4 (→ 96) and is the
  only setting whose digit classes are close to uniform
  ([class figure](figures/diffaugment-classes.png)): every digit between 2.9% and 15.6%,
  against 38% "5" for the baseline. Mode coverage, not sharper digits, drives the FD gain.
- **Recall appears only with cutout** (≈4%), i.e. once D can no longer rely on any fixed
  stroke. The price is ≈12 points of precision: G now spreads over more of the digit
  manifold, including less clean samples.
- **Stronger is not always better**: brightness/contrast on top of cutout raises FD
  (96 → 123); a plausible reason is that MNIST has no real brightness variation to be robust to.
- **Instance noise alone hurts** (FD 632): at std 0.1, annealed to 0, it did not remove D's
  shortcut on this architecture (stronger or slower-annealed noise was not tested).
- **ADA** reaches the highest recall (5.8%) but collapses toward "1" (41% of samples)
  and has worse FD than fixed cutout; its p peaked at 0.53 and ended at 0.29, so it
  augments far less than fixed cutout (p=1). Kept as a coverage option only if seed 1 confirms the recall.
- **Single-seed caveat**: the same baseline settings gave FD 774 on CPU (shared
  reference) and 503 on GPU here — training variance between runs is large. Seed-1 repeats
  are queued; claims above should be read as preliminary until then.

Visual check, first 64 exported images per condition, no selection
([figure](figures/diffaugment-samples.png)): the baseline gives speckled, noisy
strokes; instance noise gives smoother but mostly the same few digits; all three
DiffAugment policies give clean, solid strokes and visibly more digit classes;
adding brightness/contrast makes strokes more fragmented. Visual impressions are not
metrics.

## Side study: export convention (clip vs rescale)

Same baseline checkpoint (`checkpoints/baseline-seed0-e10`), same noise (seed 1),
two exports: `clip` = starter `save_image(x)` (negatives → 0); `rescale` =
`save_image((x+1)/2)`, the exact inverse of training normalisation. Script:
`scripts/export_convention_check.py`; data: [CSV](diffaugment-export-check.csv).

| σ | Export | FD-MNIST ↓ | Precision ↑ | Recall ↑ | Images classified as "1" |
|---:|---|---:|---:|---:|---:|
| 1 | clip (starter) | 774.39 | 33.71% | 0.01% | 49% |
| 1 | rescale | 540.40 | 62.28% | 0.00% | 29% |
| 2 | clip | 346.68 | 63.98% | 0.01% | 4% |
| 2 | rescale | 345.06 | 70.11% | 0.01% | 3% |

- The σ=1 clip row reproduces the shared baseline exactly (774.39 / 33.71%), so the
  evaluator behaves identically on Juliet.
- At σ=1, rescaling alone lowers FD by 234 and nearly doubles precision.
- At σ=2 rescaling barely changes FD: larger latents push pixels to ±1, which clipping
  does not damage. **Part of the latent-scaling gain may therefore be a correction of
  the export, not better sampling** — relevant to the I1 study.
- Recall stays ≈0 for every export: clipping is not the cause of the missing coverage.
- Changing the export affects every workstream and must be checked against the
  official evaluation rules before `generate.py` is modified. Not changed here.

## Limitations

One training seed per condition so far (repeats queued); local development metrics, not official
FID/precision/recall; 10-epoch budget only.
