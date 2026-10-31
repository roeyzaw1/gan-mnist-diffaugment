# Starter baseline — 31 October 2026

The initial shared reference is **`baseline-seed0-e10`**, trained from scratch
for the 10-epoch budget used by `scripts/train.sh`. The preceding one-epoch run
is a pipeline pilot. These runs establish a reproducible reference; they do not
establish convergence or the superiority of any intervention.

## Protocol

- Base commit: `1a96c9a3096ffe3d70d669030e0bed437dd24bb0`, plus the uncommitted
  baseline runner and documentation. Each run includes exact source snapshots,
  their SHA-256 hashes, and the tracked working-tree diff.
- MNIST train split: all 60,000 examples per epoch; no augmentation. Normalize
  using mean/std 0.5, then flatten to 784 dimensions. Test data is not used for
  training or tuning. Raw dataset file hashes are recorded in run metadata.
- Unchanged starter generator: `100 → 256 → 512 → 1024 → 784`, bias in all four
  linear layers, LeakyReLU(0.2) after hidden layers, final tanh; 1,486,352 parameters.
- Unchanged starter discriminator: `784 → 1024 → 512 → 256 → 1`, LeakyReLU(0.2)
  hidden activations and sigmoid output. BCE with equal real/fake training priors.
- Original `D_train` and `G_train`; one update of each per minibatch. Adam for
  both, learning rate 0.0002, betas (0.9, 0.999), batch size 64, including the
  final 32-example batch. Latent prior is standard Gaussian in 100 dimensions.
- Training seed 0; independent generation seed 1, set before generator
  construction/loading. Deterministic algorithms enabled. DataLoader workers 0,
  pinned memory disabled, two PyTorch CPU threads. No pretrained weights loaded.
- CPU: AMD Ryzen 3 PRO 5450U; eight logical CPUs visible. No CUDA or MPS device.
  Python 3.14.7, PyTorch 2.15.0+cpu, torchvision 0.29.2+cpu. Complete package
  versions are in [baseline-environment.txt](baseline-environment.txt).
- Generation runs the original `generate.py` in an isolated run directory with
  batch size 256. It saves 10,000 PNGs, numbered 0–9999, as 28×28 RGB images.
  Forty batches generate 10,240 candidates; the final 240 are unused batch
  overflow, not discriminator rejections. There is no rejection sampling.
- Export preserves the starter's clipping of negative tanh outputs, without
  rescaling to [0,1]. This convention still needs confirmation against the
  official evaluation protocol; it was not changed to improve the samples.

## Recorded runs

| Run | Epochs | D / G updates | Training (s) | 10,000-image generation (s) | Final mean D loss | Final mean G loss |
|---|---:|---:|---:|---:|---:|---:|
| `baseline-seed0-e1-r1` — pilot | 1 | 938 / 938 | 46.034 | 5.884 | 1.075873 | 2.512808 |
| `baseline-seed0-e10` — reference | 10 | 9,380 / 9,380 | 467.465 | 5.973 | 0.667381 | 2.650655 |

Losses are sample-weighted training means, not quality metrics. Training time
excludes dataset setup and checkpoint serialization. Generation time includes
process startup, checkpoint loading, inference and all PNG writes; output
validation and sample-grid creation follow that timer. Total reference-run time,
including setup, checkpoint saving and validation, was 474.749 seconds.

**Official FID, precision and recall: not yet evaluated.** No official evaluator
or reference feature protocol is present in the repository. The ledger uses
`null` official metrics with an explicit explanation; no local metric is reported
as an official result.

Subsequent [local development evaluation](local-metrics.md) separately records
FD-MNIST and k-NN precision/recall. For this baseline's seed-1 exported images:
FD-MNIST **774.39**, precision **33.71%**, recall **0.010%**. These local values
are not official FID/precision/recall and do not overwrite their null fields.

## Visual observations and limitations

The pilot's first 64 exported images are nearly black and highly repetitive.
After 10 epochs the first 64 show more structure, but many remain fragmented
and noisy, with limited variety of recognizable digits. This is a weak visual
baseline, not a submission-quality claim. The grids show the first 64 outputs
in generation order without selection; no class-coverage or recall estimate
can be inferred from them. The export clipping may contribute to the appearance.

This is one training seed. Later training interventions should use the same
declared budget and evaluator, and serious quality claims need repeated seeds.
Inference comparisons should freeze the reference checkpoint. The saved
sigmoid/BCE discriminator is available for DRS work, but its calibration and
density-ratio accuracy have not been validated.

## Artifacts and reproduction

The [ledger](ledger.jsonl) contains both completed runs. The selected reference
checkpoints and compact evidence are versioned for the whole team. Full image
batches remain local under `runs/`, which is ignored by Git. Archived metadata
preserves the original run paths and source state:

- Reference [metadata](baseline-seed0-e10/metadata.json),
  [training log](baseline-seed0-e10/training.jsonl), and
  [sample grid](baseline-seed0-e10/sample-grid.png). The original source snapshot
  is preserved in [source/](baseline-seed0-e10/source/).
- Reference [G checkpoint](../checkpoints/baseline-seed0-e10/G.pth) and
  [D checkpoint](../checkpoints/baseline-seed0-e10/D.pth), with architecture,
  preprocessing, loss, seed and optimizer metadata alongside them.
- Reference G SHA-256:
  `5b7db542088d964e85592ffe2faf9acb351e5ee9639ccee4f047791a32b0628e`.
- Reference D SHA-256:
  `e8525613cbc422405b2c20941a9a72866763b393dec59fb7b9a61f004b02e221`.
- Pilot [metadata](baseline-seed0-e1-r1/metadata.json) and
  [sample grid](baseline-seed0-e1-r1/sample-grid.png). Pilot checkpoints stay
  local; only the 10-epoch reference weights are shared.

Reproduce from the repository root with the recorded environment and a new
output directory:

```bash
.venv/bin/python scripts/baseline.py --run-dir runs/baseline-reproduction-e10 \
  --epochs 10 --seed 0 --generation-seed 1 --generation-batch-size 256
```

Every completed run validates all 10,000 filenames, PNG format, RGB mode and
dimensions; finite training losses; actual G weight updates; unchanged model
source; and unchanged checkpoint files after generation. Additional checks
confirmed bitwise reproducibility of a seeded starter G/D update, exact agreement
between both runs' first-epoch losses, and byte-for-byte reproduction of the
first 64 PNGs of both completed runs without changing G weights. Cross-hardware bitwise equality
is not claimed.

The original tracked `checkpoints/G.pth` was preserved. The first download
attempt failed before training because the restricted environment could not
resolve MNIST hosts; its metadata remains in `runs/baseline-seed0-e1/`. MNIST was
then downloaded and verified, and both completed runs used that same local data.
