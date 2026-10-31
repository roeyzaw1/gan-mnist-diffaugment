# I1 — Latent scaling on the shared baseline

The full study is consolidated here: the initial grid, adaptive extensions,
independent-seed confirmation and stopping decision. Portable evidence consists
of one [CSV](latent-scaling-all-local.csv), one
[provenance manifest](latent-scaling-manifest.json), the [workstream ledger](latent-scaling-ledger.jsonl)
and one [paired comparison](figures/latent-scaling.png). See the
[artifact policy](README.md) for what stays local.

## Protocol

Use the same frozen `checkpoints/baseline-seed0-e10/G.pth` in every condition:
10 training epochs, training seed 0, original architecture, standard Gaussian
latent prior scaled as `z = sigma * eps`. This is scaling, not coordinate-wise
truncation. Generator weights, discriminator and dependencies remain unchanged.

Each condition exports 10,000 numbered RGB PNGs of size 28×28, batch size 256,
two CPU threads, deterministic settings and the starter's negative-pixel clipping.
The final batch draws 240 unused candidates. For every generation seed, all
scales share the exact same unscaled noise, verified by hash. Explicit seeds
include model initialization, matching the original baseline RNG convention.

FD-MNIST and precision/recall use the same frozen 64-D MNIST classifier and
10,000 real development-reference images, with k=3. See the complete
[local evaluation protocol](local-metrics.md). These are **local development
metrics, not official FID/precision/recall**.

## Results: discovery seeds 1 and 2

Each row averages the two independent generation seeds on one trained GAN.
Repeated controls appear once in the CSV and are not counted as independent
training or generation replications. JSON/CSV scores use fractions; percentages
are displayed below.

| Sigma | FD-MNIST ↓ | Local precision ↑ | Local recall ↑ |
|---:|---:|---:|---:|
| 0.5 | 1531.80 | 0.76% | 0.000% |
| 0.7 | 1227.58 | 8.29% | 0.015% |
| 0.85 | 981.30 | 20.16% | 0.000% |
| 1 | 771.05 | 34.23% | 0.010% |
| 1.15 | 609.83 | 47.18% | 0.005% |
| 1.3 | 495.93 | 56.78% | 0.005% |
| 1.5 | 403.37 | 63.56% | 0.000% |
| 1.75 | 354.06 | 65.79% | 0.000% |
| 2 | 347.97 | 63.75% | 0.005% |
| 2.25 | 363.34 | 61.17% | 0.010% |
| 2.5 | 387.96 | 58.65% | 0.010% |
| 3 | 441.78 | 54.90% | 0.020% |

**Best tested local precision: sigma 1.75. Best tested FD-MNIST: sigma 2.0.**
FD worsens above 2.0 in both seeds, and precision falls above 1.75. This brackets
the best sampled FD point; it does not identify an exact continuous optimum.
Recall stays almost zero: even sigma 3.0 covers only 3 and 1 real reference
images out of 10,000 in the two runs. Those tiny differences do not justify a
new coverage candidate.

## Independent confirmation: generation seed 3

Candidates 1.75 (precision) and 2.0 (FD) were selected using discovery seeds
1 and 2 before looking at seed 3. The controls were 1.0 and 1.3.

| Sigma | FD-MNIST ↓ | Local precision ↑ | Local recall ↑ |
|---:|---:|---:|---:|
| 1 | 771.67 | 34.31% | 0.010% |
| 1.3 | 492.34 | 56.93% | 0.010% |
| 1.75 | 351.23 | 65.93% | 0.000% |
| 2 | 346.64 | 64.13% | 0.000% |

The confirmation preserves both rankings. No sampler default was changed:
sigma 1 remains the reference. Seed 3 is an inference replication on the same
GAN, not an independently trained generator.

## Stage history and stopping rule

1. Initial grid: 0.5, 0.7, 0.85, 1.0, 1.15, 1.3; seeds 1 and 2.
2. First extension: 1.0, 1.3, 1.5, 1.75, 2.0; seeds 1 and 2.
3. Confirmation: 1.0, 1.3, 1.75, 2.0; seed 3, fixed before scoring it.
4. Closing upper grid: 1.0, 1.75, 2.0, 2.25, 2.5, 3.0; seeds 1 and 2.

The final grid's stopping rule was recorded before metric evaluation. No new
FD/precision candidate improved the previously confirmed ones, so conditional
seed-4 confirmation was unnecessary and was not run. The manifest embeds the
original plans, decisions, hashes and run configurations; no range was omitted.
Across stages, 38 conditions exported 380,000 PNGs, including repeated controls.
The portable CSV deduplicates exact controls into 28 unique `(seed, sigma)`
conditions: 24 discovery scores and four confirmation scores.

## Visual observations and limitations

The [single retained figure](figures/latent-scaling.png) shows seed 1, the same
first 16 latent indices per row, across all 12 tested scales. There is no
selection of attractive examples. Low scales make faint, often repetitive
strokes; larger scales make thicker, more visible structures, many still noisy.
Brightness is not a quality metric, and the preserved export clipping affects
appearance and evaluation.

Local precision means feature-space manifold membership, not human-verified
correct digit percentage. Recall near zero is a finite-sample estimate, not a
proof of zero theoretical distributional recall. The real development reference
was held out from classifier training but was used during the original GAN
training. Results concern one weak generator and one imperfect feature
extractor. Official metric fields remain null.

Generation timings in the ledger include model loading, integrity snapshots,
noise hashing, raw diagnostics, inference and PNG writes; they exclude Python
imports and post-export validation/grids. Conditions ran in declared order and
tests overlapped the initial sweep, so timings are descriptive, not controlled
speed rankings. The original baseline used a different subprocess-based timer.

## Validation and reproduction

Every condition checked the exact output count/names, PNG/RGB/28×28 format,
finite outputs, paired noise, unchanged G weights and checkpoint/source hashes.
Repeated controls match image-stream hashes and all three metric values exactly.
Sampler tests cover sigma-1 baseline equivalence, scaling arithmetic, RNG
isolation, partial batches, invalid parameters and default CLI compatibility
against all 10,000 starter PNGs. Metric tests cover analytical Fréchet cases,
singular covariances, k-NN self exclusion and brute-force manifold membership.

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/latent_scaling.py \
  --run-dir runs/scaling-reproduction \
  --sigmas 1.0 1.75 2.0 2.25 2.5 3.0 --seeds 1 2
.venv/bin/python scripts/evaluate_frozen.py \
  --sweep runs/scaling-reproduction --run-dir runs/metrics-reproduction
```

Other stage configurations, hardware, versions and hashes are in the manifest.
Full per-run samples, metadata, grids and exact source snapshots remain under
`runs/`, ignored by Git. The shared baseline and frozen evaluator are each
versioned once; do not promote every run directory into the repository.

## Workstream files

The shared tools and evaluation protocol live in the common base. This study's
runner is `scripts/latent_scaling.py`; its default ledger is
`experiments/latent-scaling-ledger.jsonl`. The common baseline ledger remains
separate. To reproduce an earlier grid, pass its declared sigma/seed list to the
runner, then use `evaluate_frozen.py --sweep` as above. Keep the generation batch
size at 256 for paired comparison with the recorded results.

The complete original evidence and the files before the common-base split are
retained locally under `runs/artifact-archive-before-compaction/` and
`runs/before-common-split/`. No experiment images or recorded scores were deleted.
