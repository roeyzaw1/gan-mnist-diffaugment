# T1 — DiffAugment, instance noise and adaptive augmentation (training-side, D inputs)

Workstream: differentiable augmentation. Status: **complete** — 6 conditions on training seed 0, and
4 training seeds (0–3) for the three main methods.
Portable evidence: this report, [results CSV](diffaugment-results.csv),
[export-check CSV](diffaugment-export-check.csv), [workstream ledger](diffaugment-ledger.jsonl),
figures: [main result](figures/diffaugment-main.png), [collapse](figures/diffaugment-collapse-seed1.png),
[all conditions](figures/diffaugment-metrics.png), [digit classes](figures/diffaugment-classes.png),
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

## Main result: quality, diversity and reliability over 4 training seeds

Same protocol, training seeds 0–3 (generation seed 1). Each cell: FD-MNIST / precision % /
recall %. A run is counted as **collapsed** when the generator outputs one fixed pattern
(precision 0; D loss → 0 within 2–3 epochs). [Figure](figures/diffaugment-main.png).

| Method | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Mean | Collapsed |
|---|---|---|---|---|---|---:|
| Baseline | 503 / 57.2 / 0.00 | 1704 / 0.0 / 0.00 | 494 / 65.2 / 0.04 | 894 / 26.4 / 0.02 | **899** / 37.2 / 0.01 | 1/4 |
| **DiffAugment** (translation + cutout) | 96 / 43.3 / 4.14 | 102 / 55.0 / 2.50 | 108 / 47.7 / 2.33 | 92 / 51.2 / 3.26 | **100** / 49.3 / 3.06 | 0/4 |
| **ADA** (translation + cutout, adaptive p) | 376 / 44.5 / 5.77 | 413 / 48.6 / 10.35 | 299 / 49.2 / 6.85 | 441 / 52.9 / 6.55 | **382** / 48.8 / 7.38 | 0/4 |

- **Normal training is a gamble.** The baseline's FD ranges from 494 to 1704 across seeds;
  in seed 1 D's loss reached 0 in epoch 2, D rejected every fake, and G produced one
  identical noise pattern for every input ([figure](figures/diffaugment-collapse-seed1.png)).
- **DiffAugment (translation + cutout) is both the best and the most stable**: FD 92–108
  on every seed (mean ≈100, ≈9× lower than the baseline mean), 0/4 collapses. Its D loss
  rises steadily to ≈1.3 in every run — D finds the game harder over time instead of
  winning it.
- **ADA is also stable (0/4) and gives the most diversity** (recall 5.8–10.4%, mean 7.4%)
  but worse FD (mean 382). In seed 1, when D started to win, ADA raised p to 0.8 by
  epoch 3 and training recovered. Its samples are dominated by "1"s on every seed (31–51% of
  samples), whereas DiffAugment keeps every digit between 2.9% and 21% on every seed:
  diversity in the recall sense (coverage of real samples) is not the same as balanced
  digit classes.
- **Quality–diversity trade-off**: precision is similar for DiffAugment and ADA (≈49%),
  above the baseline mean (37%, pulled down by the collapsed and weak runs) but below its
  best runs (57–65%). Augmentation buys reliability and coverage, not sharper digits.

## Ablation: which augmentation? (training seed 0)

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

**Interpretation of the ablation (one training seed per condition).**

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
  augments far less than fixed cutout (p=1). Its recall advantage repeated on seeds 1–3 (see main result).
- **Translation alone is not enough**: on seed 1 it collapsed exactly like the baseline
  (FD 1691, precision 0) — the shift is easy for D to see through; cutout is what keeps
  D from winning.
- Instance noise, colour and translation-only were run on seed 0 only (plus seed 1 for
  translation): treat their ranking as indicative.

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

Four training seeds for the main methods, one or two for the ablation conditions; local
development metrics, not official FID/precision/recall; 10-epoch budget only; augmentation
strengths (4-px shift, 14×14 cutout, ADA target 0.6) were not tuned. The same baseline gave
FD 774 on the shared CPU reference and 494–894 on our GPU runs when it did not collapse:
compare methods within this study, not across machines.
