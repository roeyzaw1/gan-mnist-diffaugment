# I2 — Discriminator diagnostics and practical DRS

**Latest result: probabilistic approximate DRS was implemented and tested.**
Both fresh generation seeds completed 10,000 outputs within the declared
candidate/time budgets. DRS worsened local FD-MNIST and precision while costing
more than the paired sigma-2 unfiltered controls. See the final trial section.

The earlier sections preserve the prerequisite diagnostics and decisions made
before DRS was implemented. Their top-k results concern different procedures;
the final section contains the actual probabilistic DRS evidence.

The original baseline discriminator separates real from generated inputs reasonably
well, but selecting its highest scores gives only a small local precision gain
and worsens FD-MNIST against random selection in both generation seeds. This is
a **top-k diagnostic, not Discriminator Rejection Sampling**, and does not
establish a calibrated density ratio. No sampler default or trained weight changed.

Portable evidence is this report, one [manifest](drs-diagnostic-manifest.json),
one [workstream ledger](drs-ledger.jsonl), and a curated figure for each major
stage: [original D](figures/discriminator-diagnostic.png),
[paired auxiliary diagnostics](figures/drs-auxiliary.png), and
[actual DRS](figures/drs.png). Complete images, scores, indices and snapshots
remain under ignored `runs/`; each stage below gives its own run paths.

## Declared protocol

- G and D: shared `baseline-seed0-e10`, one training seed (0), 10 epochs,
  unchanged starter architectures and weights. Original equal-prior BCE/sigmoid D.
- Native D inputs: raw G tanh outputs and real MNIST pixels normalized to [-1,1].
  The actual pre-sigmoid logits are extracted with a temporary final-layer hook;
  probabilities are not clipped/inverted to reconstruct scores.
- Calibration assessment: 10,000 real images from the frozen evaluator's
  `real_control` indices vs 10,000 generated images from a separate seed (91).
  Equal class priors; AUC, Brier, stable log loss and 15-bin ECE/reliability.
  This assesses the existing scores; no calibrator was fitted.
- Real controls were not used to train the independent feature extractor, but
  **were used in the original GAN/D training**. This is a local development
  assessment, not held-out discriminator generalization or calibration.
- Selection: generation seeds 1 and 2, each with a fixed 20,000-candidate pool,
  sigma 1, batch size 256. Keep 10,000 images for each of `first`, `random`,
  and highest native-logit `top`. Random selection uses private RNG seed
  `generation_seed XOR 0x5EED`. Top-score ties prefer earlier candidate IDs.
  All selections are without replacement and exported in candidate order.
- Each pool requires 20,224 forward candidates because full batches are drawn;
  224 are unused batch overflow. The 50% retained fraction is a top-k budget,
  **not a measured DRS acceptance probability**.
- Evaluation: all 10,000 actual PNGs per condition, original clipping/no rescaling,
  the unchanged shared MNIST64/k3 extractor and real-reference set. Official
  FID/precision/recall remain null. See [common protocol](local-metrics.md).
- Seed/count/selection settings were fixed and written before scoring outputs.
  Both unfiltered controls reproduce their archived sigma-1 PNG fingerprints
  and all three local scores exactly.

## Local selection results

Means of two generation seeds, on one trained GAN/D pair; these are not
independent training replications or significance estimates.

| Selection | FD-MNIST ↓ | Local precision ↑ | Local recall ↑ |
|---|---:|---:|---:|
| First 10,000 (unfiltered control) | 771.05 | 34.225% | 0.010% |
| Random 10,000 of 20,000 | 770.70 | 34.745% | 0.035% |
| Top-score 10,000 of 20,000 | 776.07 | 35.220% | 0.025% |

| Seed | Selection | FD-MNIST | Precision | Recall |
|---:|---|---:|---:|---:|
| 1 | first | 774.3911 | 33.71% | 0.010% |
| 1 | random | 774.6016 | 34.15% | 0.000% |
| 1 | top | 779.9877 | 34.50% | 0.000% |
| 2 | first | 767.7003 | 34.74% | 0.010% |
| 2 | random | 766.8009 | 35.34% | 0.070% |
| 2 | top | 772.1568 | 35.94% | 0.050% |

Against random selection, top-score precision rises by **0.475 percentage points**
on average, while FD-MNIST worsens by **5.37**. Precision rises and FD worsens
in each seed. Recall corresponds to just 0/7 covered reference images for random
and 0/5 for top selection; these tiny counts do not establish a recall effect.
No threshold sweep or "best-looking" subset was used to rescue the result.

## Discrimination, calibration and export shift

| Input space (real vs fake, 10,000 each) | AUC ↑ | Brier ↓ | Log loss ↓ | ECE ↓ |
|---|---:|---:|---:|---:|
| Native raw generator output | 0.9209 | 0.1040 | 0.4253 | 0.1098 |
| Fake after starter PNG mapping | 0.8880 | 0.1104 | 0.4570 | 0.0876 |

The constant 0.5 predictor has Brier 0.25 and log loss 0.6931 on this balanced
set. D therefore has a real-vs-generated signal, but that is distinct from
ranking perceptual quality within generated samples. Brier/log loss measure
probability quality, including discrimination; they are not pure calibration
errors. ECE is a finite-bin descriptive estimate, not a density-ratio guarantee.
Full reliability bins and quantiles are in the manifest. See the
[scikit-learn calibration documentation](https://scikit-learn.org/stable/modules/calibration.html).

Native mean predicted real probability is **0.7339** on real controls and
**0.0465** on generated controls. No native generated scores hit the near-zero
or near-one thresholds (1e-4 and 1−1e-4). On real controls, 11.22% exceed
1−1e-4 and 0.35% round to exactly 1 in float32 sigmoid. Reading logits avoids
losing their ordering to sigmoid saturation.

Applying the starter PNG mapping changes the same fake images' scores:
mean predicted real probability becomes **0.1029** and native-vs-exported
Spearman correlation is only **0.4233** (0.4375/0.4347 in the two candidate pools).
This mapping exactly matches save_image quantization followed by evaluation
normalization, checked against actual PNG pixels. It diagnoses export shift;
the experiment still selects using native raw scores, and the final metrics
always read actual exported PNGs. A lower ECE after export does not establish
a better quality ranking or a valid density ratio.

## Visuals, costs and decision

The [single figure](figures/discriminator-diagnostic.png) shows the first 16
selected candidate IDs in each condition, in original order, enlarged 2× with
nearest-neighbor solely for display. Many outputs remain fragmented or noisy;
no clear broad quality improvement is visually established. Class predictions
shift under selection but are diagnostic labels, not verified coverage.

The complete run took **61.82 seconds** on CPU with two threads.
Per-condition costs include shared-pool generation and native scoring plus
selection/export; shared costs repeat in each condition and must not be summed
as independent executions. Tests ran concurrently during the initial phase,
so these timings are descriptive, not controlled speed comparisons.
No extra training, rejection loop or maximum-attempt behavior was tested.

**Decision:** keep sigma 1 and the existing generation default. The small
precision signal with worse FD does not justify treating this D as a calibrated
DRS ratio estimator. It also does not prove that DRS cannot work: top-k is a
different procedure, and the local evaluator is imperfect. Before an acceptance
sweep, train/validate a dedicated equal-prior logistic scorer against the fixed
G using the same final image preprocessing, with separate fitting/calibration
and diagnostic subsets. Document the target/proposal space explicitly.
The [DRS paper](https://arxiv.org/abs/1810.06758) distinguishes exact assumptions
from practical approximate corrections. No DRS performance claim is made here.

## Reproduction and checks

```bash
python -m unittest discover -s tests -v
python scripts/diagnose_discriminator.py \
  --run-dir runs/drs-diagnostic-reproduction \
  --seeds 1 2 --pool-size 20000 --batch-size 256 --threads 2
```

The runner refuses existing run directories, verifies G/D pair metadata and
data/split hashes, writes source snapshots, scores and selected-index arrays,
exports 10,000 PNGs per condition, and invokes the shared frozen evaluator using
a generic input manifest. Successful validation checks unchanged G/D files,
G/D parameters and source hashes. Complete results remain local; portable
evidence is curated into the four files linked above.

**31 tests passed**, including seven new checks for native logits, saturation,
weight/RNG invariants, selection/ties, analytical calibration controls, invalid
inputs and exact PNG mapping. The required generator CLI and its all-10,000-PNG
compatibility regression remain unchanged. CUDA/MPS were not exercised.

## Auxiliary scorer, sigma 1 — completed negative control

A fresh logistic scorer was trained against the same frozen generator, using
its **actual starter PNG distribution** rather than raw tanh images. The input
mapping is exact uint8 save_image quantization/clipping followed by the common
normalization; a regression test and rescoring actual exported PNGs check it.
G, the original GAN discriminator and the shared evaluator remain unchanged.

### Prespecified training and data separation

- Auxiliary architecture: 784→256→128→1 MLP, LeakyReLU(0.2), raw logits;
  **233,985 parameters**, fresh initialization, training seed 2028.
- Five epochs of Adam(lr=0.0002, default betas), BCEWithLogitsLoss, batch size
  256, equal real/fake totals, CPU, two threads. No architecture/epoch sweep.
- Seeded real partition (3028): 20,000 training, 5,000 validation, 5,000
  calibration and 10,000 diagnostic examples. All four groups are disjoint.
  The evaluator's 10,000 reference and 10,000 real-control IDs are excluded.
  These are still MNIST **train** images originally seen by G; MNIST test is unused.
- Fake split seeds: train 10001, validation 10002, calibration 10003, diagnostic
  10004, with the same frozen G and sigma 1, generation batch size 256.
- Select the epoch with lowest balanced validation log loss. Epoch 5 was selected.
  Calibration and diagnostic data did not select the checkpoint or hyperparameters.
- Fit `calibrated_logit = a * raw_logit + b`, with positive `a`, only on the
  separate balanced calibration set. A positive affine map preserves score
  ranking, so it cannot change which candidates top-k selects.
- The loader binds scorer weights to G hash, sigma, input mapping, split hash
  and scorer/preprocessing source hashes. It rejects reuse at sigma 2.
  Sigma 2 requires an independently trained matched scorer.

### Independent real/fake diagnostic

10,000 real and 10,000 fake examples, excluded from fitting, epoch selection
and calibration. Their metrics were read only after all choices were fixed.

| Scorer probabilities | AUC | Brier | Log loss | ECE (15 bins) |
|---|---:|---:|---:|---:|
| Before affine calibration | 1.000000 | 1.85423e-05 | 0.000243274 | 0.000231425 |
| After affine calibration | 1.000000 | 1.63326e-08 | 1.21975e-06 | 1.2115e-06 |

Fitted calibration: **a=7.686512, b=-1.501191**.
Classification is nearly perfectly separated: raw diagnostic real logits range
from +0.822 to +29.618, and fake logits from −15.739 to −0.339. After calibration,
99.99% of diagnostic fake probabilities are ≤1e-4. The small losses demonstrate
good binary prediction on this development split, **not accurate density ratios
in rare generated regions or useful perceptual-quality ranking**. The strong
separation makes practical rejection efficiency a concern; no acceptance rate
has been measured. Disjoint calibration follows the
[calibration protocol guidance](https://scikit-learn.org/stable/modules/calibration.html).

### Fresh candidate-pool diagnostic

Prespecified generation seeds **11 and 12**, distinct from all scorer fitting,
validation, calibration and diagnostic fake seeds. Each pool has 20,000 retained
candidates (20,224 forward candidates including full-batch overflow). Export
10,000 per condition: first, random without replacement, or largest calibrated
logit. Top ties prefer earlier IDs; all PNGs are written in candidate order.
The actual six 10,000-PNG batches use the unchanged frozen MNIST64/k3 evaluator.
No threshold, epoch or scorer change was made after these results.

| Selection, mean of seeds 11/12 | FD-MNIST ↓ | Precision ↑ | Recall ↑ |
|---|---:|---:|---:|
| Unfiltered first 10,000 | 766.41 | 34.775% | 0.025% |
| Random 10,000 of 20,000 | 770.49 | 34.995% | 0.030% |
| Auxiliary top-score 10,000 of 20,000 | 900.77 | 22.755% | 0.005% |

| Seed | Selection | FD-MNIST | Precision | Recall |
|---:|---|---:|---:|---:|
| 11 | first | 762.2333 | 34.89% | 0.010% |
| 11 | random | 765.5105 | 35.67% | 0.040% |
| 11 | top | 894.8457 | 22.99% | 0.010% |
| 12 | first | 770.5891 | 34.66% | 0.040% |
| 12 | random | 775.4615 | 34.32% | 0.020% |
| 12 | top | 906.6919 | 22.52% | 0.000% |

Top selection worsens FD by **130.28** and precision by **12.240 percentage
points** relative to random, averaged over the two seeds. The direction holds
in both seeds. Recall remains based on only a few covered reference images.
These are two generation seeds on one G and one auxiliary fit, not a statistical
confidence interval or independent training replications. The new seeds also
mean cross-stage comparisons to the original D table are descriptive; this
stage's random and unfiltered controls are the paired comparators.

The [auxiliary figure](figures/drs-auxiliary.png) shows the first 16 selected
candidate IDs in order, enlarged 2× solely for display. Its first six rows are
sigma 1; the next six rows show the paired sigma-2 stage added below. Many top-selected
outputs remain faint and fragmented. Predictions and confidence are diagnostics,
not human-verified digit quality or true class coverage.

### Decision and reproduction

**Retain this as a negative sigma-1 control. Do not enable auxiliary v1 as a
DRS quality improvement.** Calibrated real/fake discrimination did not translate
into better top-k outputs. This does not establish the outcome of probabilistic
DRS, which is a different procedure and remains unimplemented. Any future DRS
experiment must use explicit approximation/budget settings and compare against
its own unfiltered control. The prior sigma-2 candidate remains available, but
requires its own matched scorer; this sigma-1 scorer is deliberately rejected
when loaded for sigma 2.

Training took **4.16 s** (6.40 s including data preparation, calibration and diagnostics).
The six-condition selection run took **60.69 s**. No tests overlapped these two runs.
Per-condition cost fields repeat shared-pool generation/scoring costs; do not
sum them as independent executions. Official scores remain null.

```bash
python scripts/train_auxiliary_scorer.py \
  --run-dir runs/drs-auxiliary-sigma1-reproduction \
  --sigma 1 --epochs 5 --seed 2028 --split-seed 3028 --threads 2
python scripts/diagnose_auxiliary_scorer.py \
  --run-dir runs/drs-auxiliary-sigma1-selection-reproduction \
  --scorer-dir runs/drs-auxiliary-sigma1-reproduction \
  --sigma 1 --seeds 11 12 --pool-size 20000 --batch-size 256 --threads 2
python -m unittest discover -s tests -v
```

The scorer weights, split indices, calibration metadata and logits remain in
ignored `runs/drs-auxiliary-sigma1-v1/`; PNGs, selection indices and evaluator
output remain in `runs/drs-auxiliary-sigma1-selection-v1/`. The existing manifest
now contains `auxiliary_scorer_sigma1_v1`, and the existing ledger includes the
training record and six new conditions. Only one additional curated figure was
added; no per-run artifact directory or experimental weight was promoted to Git.

**36 tests passed.** New checks cover split exclusion/disjointness, calibration
and rank preservation, actual export-space scoring, proposal/hash guards, and
auxiliary training. Independent integration checks reloaded the actual scorer,
verified top-k/unique indices, rescored PNGs against cached pool scores, and
confirmed unchanged G/original-D weights, source hashes and frozen evaluator.

## Matched auxiliary scorer, sigma 2 — completed negative control

The sigma-2 comparison used a **separately fitted scorer and calibration**,
with the same architecture, five-epoch budget, seeds, learning rate, and real
partitions as the sigma-1 stage. Nothing was tuned after the sigma-2 scores.
G and the evaluator remain frozen. The initial scorer weights and training
shuffle use seed 2028; real split seed is 3028. The fake split seeds and all
base latent pools are identical across scales before multiplication by sigma.
Actual selected weights and calibration parameters differ. The loaders reject
cross-sigma reuse in either direction.

The 20,000/5,000/5,000/10,000 real groups are exactly the same index arrays as
sigma 1, remain disjoint, and exclude the evaluator reference/control. Fresh
candidate-pool seeds 11 and 12 reuse the exact same unscaled noise hashes at
both sigmas; each condition evaluates all 10,000 exported PNGs. Comparing each
scorer to its own random control therefore controls pool size, output count,
noise, image mapping, feature extractor and training budget.

### Probability diagnostic

Epoch 5 again had the lowest validation log loss. On the untouched auxiliary
diagnostic split (10,000 real vs 10,000 fake):

| Scorer probabilities | AUC | Brier | Log loss | ECE (15 bins) |
|---|---:|---:|---:|---:|
| Before affine calibration | 1.000000 | 8.50712e-06 | 0.000214387 | 0.00020973 |
| After affine calibration | 1.000000 | 1.41426e-13 | 2.65961e-09 | 2.65957e-09 |

The separate fit gave **a=10.238504, b=2.712994**.
Raw real logits range from +1.821 to +21.764 and fake logits from −18.711
to −1.226. After calibration all 10,000 diagnostic fake probabilities are
≤1e-4. This is excellent real/fake separation on this development set, but
does not establish reliable ratios in rare proposal regions or good quality
ranking. The following paired selection test directly checks the ranking.

### Image metrics

| Sigma | Selection | Mean FD-MNIST ↓ | Mean precision ↑ | Mean recall ↑ |
|---:|---|---:|---:|---:|
| 1 | first | 766.41 | 34.775% | 0.025% |
| 1 | random | 770.49 | 34.995% | 0.030% |
| 1 | top | 900.77 | 22.755% | 0.005% |
| 2 | first | 349.74 | 63.875% | 0.005% |
| 2 | random | 348.95 | 63.685% | 0.005% |
| 2 | top | 462.87 | 49.010% | 0.020% |

Individual sigma-2 conditions:

| Seed | Selection | FD-MNIST | Precision | Recall |
|---:|---|---:|---:|---:|
| 11 | first | 348.2615 | 64.06% | 0.010% |
| 11 | random | 348.2572 | 63.72% | 0.010% |
| 11 | top | 461.8765 | 48.63% | 0.040% |
| 12 | first | 351.2176 | 63.69% | 0.000% |
| 12 | random | 349.6449 | 63.65% | 0.000% |
| 12 | top | 463.8572 | 49.39% | 0.000% |

At sigma 2, top selection worsens FD by **113.92** and loses **14.675 percentage points**
of precision against random. Both seeds show the same direction. The recall
counts are only 1/0 covered reference images for random and 4/0 for top; this
does not establish a meaningful coverage improvement. No significance or
independent-G training replication claim is made.

**Conclusion:** sigma 2 without auxiliary filtering remains preferable among
these tested configurations. Neither auxiliary v1 supports a top-k quality
improvement. No DRS acceptance sampler was run, so this is not evidence that
probabilistic DRS itself fails. If DRS is tested next, it must be labelled as an
approximate controlled experiment with explicit candidate/time budgets and
its own unfiltered baseline, rather than a validated consequence of high AUC.
The generation default remains sigma 1.

The [same auxiliary figure](figures/drs-auxiliary.png) now contains both sigmas:
first six rows sigma 1, next six sigma 2. First 16 retained candidate IDs per
condition, sorted by original pool index, with 2× nearest-neighbor enlargement.
No attractive-sample selection was used. Original per-stage figures remain
under the ignored run directories.

Training took **4.72 s** (7.08 s total); the six-condition
selection/evaluation run took **64.19 s**. No tests overlapped either run.
All scores remain local development metrics; official fields remain null.

```bash
python scripts/train_auxiliary_scorer.py \
  --run-dir runs/drs-auxiliary-sigma2-reproduction \
  --sigma 2 --epochs 5 --seed 2028 --split-seed 3028 --threads 2
python scripts/diagnose_auxiliary_scorer.py \
  --run-dir runs/drs-auxiliary-sigma2-selection-reproduction \
  --scorer-dir runs/drs-auxiliary-sigma2-reproduction \
  --sigma 2 --seeds 11 12 --pool-size 20000 --batch-size 256 --threads 2
```

Full weights/splits/metadata/logits remain under `runs/drs-auxiliary-sigma2-v1/`;
60,000 PNG exports, selected indices and metric outputs remain under
`runs/drs-auxiliary-sigma2-selection-v1/`. The existing manifest gains
`auxiliary_scorer_sigma2_v1`; the existing ledger gains one fit and six
conditions. No new portable artifact directory, report, figure file or
experimental checkpoint was added. Before-update portable evidence is backed
up locally under `runs/before-sigma2-evidence/`.

The experiment reused the existing tested code without source changes.
Integration checks verify actual sigma-2 loading and sigma-1 rejection,
identical real split arrays and base-noise hashes across sigmas, unique/top-k
indices, actual PNG score mapping, unchanged source/G/D hashes, and consistency
of the portable manifest/ledger with completed run summaries.

## Actual probabilistic DRS — sigma 2, completed bounded trial

This stage ran acceptance/rejection draws until each trial retained exactly
10,000 outputs. It is distinct from the earlier fixed-pool top-k diagnostics.
The scorer, generator and evaluator were frozen; no further training occurred.
No method parameter was selected using this trial's image metrics.

### Acceptance rule and approximation

Following the practical [DRS rule, Eq. 8](https://arxiv.org/html/1810.06758),
let `s` be the auxiliary calibrated logit and `m` an empirical maximum logit.
For each production batch, update `m` before computing:

```text
delta = s - m
F = delta - log(-expm1(delta - 1e-6))
gamma = median(F(pilot_scores; current m))
p_accept = sigmoid(F - gamma)
accept when independent Uniform(0,1) < p_accept
```

The independent 10,000-image pilot estimates the initial maximum. Its scores
are fixed; gamma is recomputed on those same scores whenever the running
maximum increases. This declared pilot-median variant maintains an operating
point near 50% without tuning image metrics. It differs from a per-production-
batch percentile shift. The empirical maximum, learned logits and nonzero
shift make this **approximate practical DRS**, without an exact target-
distribution guarantee. No temperature or gamma sweep was run.

Float64 `expm1` avoids subtracting nearly equal exponentials. Sigmoid yields
bounded probabilities without reconstructing logits from rounded probabilities.
G samples are scored after the exact starter PNG mapping, then accepted raw
outputs are exported using the same convention; the evaluation reads actual
PNGs. A new module `drs.py` implements the bounded iterator and stable rule.
`scripts/run_drs.py` records controls, pilots, costs, audits and final metrics.

### Declared protocol and budgets

- Sigma **2**, original shared G checkpoint, separately trained sigma-2 auxiliary
  scorer from the previous stage. Scorer/G/sigma/preprocessing/source hashes
  are checked before use. Generator architecture and weights remain unchanged.
- Fresh output seeds **21 and 22**, batch size 256, two CPU threads. Neither seed
  was used for auxiliary fitting, calibration or earlier top-k diagnostics.
- Independent pilot seed **20001**, shared as a fixed pilot distribution across
  repeats and regenerated/charged in each DRS condition. It is not an output pool.
- Acceptance RNG is private, seed `generation_seed XOR 0xD25`, independent of
  the private proposal RNG. Unfiltered and DRS paths share the same proposal
  sequence prefix; acceptance draws do not change G's latent sequence.
- Per DRS condition: at most **102,400 production candidates**, plus the pilot.
  Pilot size 10,000 entails 10,240 forward candidates with full final batches.
  Maximum total forward budget is therefore **112,640**. No uncounted overflow
  occurs at the production cap, which must be a multiple of batch size.
- Per-condition deadline **60 seconds**, including G/scorer load, pilot, scoring,
  acceptance and all 10,000 PNG writes. Checks occur around batches, including
  consumer write time; native kernels/writes are not asynchronously preempted.
  Common Python imports, protocol snapshots, audit serialization and metric
  evaluation are outside the reported sampling deadline/timer.
- The second seed runs only after the first DRS trial completes its output
  target inside both budgets. It ran because that condition was satisfied,
  before inspecting image-metric outcomes.
- Budget exhaustion records a partial batch and stops repeats, without filling
  it with rejected samples or evaluating it as a 10,000-image batch. This path
  was checked by unit tests and an actual CLI timeout smoke test.

### Output integrity and genuinely probabilistic acceptance

Both seeds completed. Per seed: **20,224 production candidates** and **10,240
pilot forward candidates**, totaling **30,464** G forwards for 10,000 output
images. Each unfiltered control used 10,240 forward candidates, including the
starter's 240 unused final-batch outputs.

| Seed | Bernoulli accepted / tested | Effective acceptance | Final accepted surplus dropped | Accepted with p<0.5 |
|---:|---:|---:|---:|---:|
| 21 | 10006 / 20224 | 49.476% | 6 | 234 |
| 22 | 10062 / 20224 | 49.753% | 62 | 276 |

Acceptance rate counts every Bernoulli success in the tested final batch.
Only the first 10,000 successes are exported; surplus successes are recorded
and discarded without replacement. The retained-output/production-candidate
rate is **49.446%** in both seeds. Pilot candidates are charged separately.
Probabilities lie strictly between 0.01 and 0.99 for 3,138/3,194 tested
candidates in seeds 21/22. Thus outputs are stochastic, not an exact top-k set.
Accepted candidate IDs are strictly increasing and unique; distinct latent
candidates may naturally quantize to identical pixels, without algorithmic
duplication or artificial deduplication.

Local NPZ audits contain retained candidate IDs, calibrated logits, acceptance
probabilities and uniform draws; bound-update logs capture online parameters.
Independent NumPy recomputation reproduced the acceptance probabilities, and
the independently reconstructed private uniform sequence matches every saved
draw. All saved draws satisfy `u < p`. Rescoring actual exported PNGs matches
the cached retained logits. The manifest stores audit/index hashes.

### Local image metrics and cost

Means of the two generation seeds, on one trained G/scorer pair:

| Sampling | FD-MNIST ↓ | Precision ↑ | Recall ↑ | Seconds for 10,000 PNGs | G forward candidates, including pilot |
|---|---:|---:|---:|---:|---:|
| Sigma 2, unfiltered | 351.27 | 63.62% | 0.000% | 3.353 | 10240 |
| Sigma 2, practical approximate DRS | 468.76 | 48.12% | 0.015% | 4.485 | 30464 |

| Seed | Sampling | FD-MNIST | Precision | Recall | Seconds |
|---:|---|---:|---:|---:|---:|
| 21 | unfiltered | 347.4546 | 63.65% | 0.000% | 3.176 |
| 21 | drs | 465.6893 | 47.94% | 0.030% | 4.164 |
| 22 | unfiltered | 355.0841 | 63.59% | 0.000% | 3.530 |
| 22 | drs | 471.8246 | 48.30% | 0.000% | 4.805 |

DRS worsened mean FD by **117.49** and lost **15.50 precision percentage points**.
Both seeds show worse FD and precision. Recall covers 3/0 real-reference images
for DRS and 0/0 for controls, too few to establish improved coverage.
Observed sampling time increased **1.34×**, with **2.975×** the forward candidates.
Timings include each DRS pilot and cannot be compared directly to earlier
fixed-pool diagnostic timers. These sequential CPU timings are descriptive,
not controlled microbenchmarks; no tests overlapped the sampling run.
The full trial, including frozen metric evaluation, took **45.87 seconds**.
All official FID/precision/recall fields remain null.

The [actual DRS figure](figures/drs.png) shows the first 16 retained outputs in
each of four rows, enlarged 2× for display only, with no attractive-image
selection. The DRS rows show first accepted candidates in stream order, rather
than matching the unfiltered row's latent IDs after rejection.
The displayed DRS outputs emphasize repeated similar forms, with noisy and
fragmented samples still present; visual inspection does not establish a quality
improvement or ground-truth class coverage.

**Decision:** this approximate DRS configuration is a completed negative result.
Keep sigma-2 generation without auxiliary rejection as the preferable tested
configuration for this checkpoint; the default generation CLI remains sigma 1.
This result concerns the declared median-pilot shift, sigma, scorer, G and
budget. It does not establish the behavior of every DRS configuration or a
different trained generator. No additional acceptance/temperature search was
performed to replace the negative result.

### Reproduction and checks

Use the sigma-2 scorer reproduction command from the preceding section, then:

```bash
python scripts/run_drs.py \
  --run-dir runs/drs-sigma2-reproduction \
  --scorer-dir runs/drs-auxiliary-sigma2-reproduction \
  --sigma 2 --seeds 21 22 --pilot-seed 20001 --pilot-size 10000 \
  --max-candidates 102400 --max-seconds 60 --batch-size 256 --threads 2
python -m unittest discover -s tests -v
```

**44 tests passed**, including eight new DRS tests: stable Eq.8 arithmetic,
monotonic bounded probabilities, maximum updates, probabilistic constant-score
control, reproducible private RNG/exact count/unique IDs/final surplus,
candidate exhaustion without fallback, time accounting for export, short or
malformed streams and invalid parameters. The additional real CLI timeout smoke
verified zero outputs, no fallback, no evaluation and no second seed.
Weights, source and unchanged official generator CLI checks also passed.

The existing manifest gains `practical_drs_sigma2_v1`; the ledger gains four
completed conditions. One curated DRS figure is added. All 40,000 PNGs, pilots,
source snapshots, uniform/ID/logit audits and metric caches remain under ignored
`runs/drs-sigma2-pilot-v1/`. Before-update portable evidence is preserved locally
under `runs/before-probabilistic-drs-evidence/`. No experimental scorer or run
directory was promoted to Git.
