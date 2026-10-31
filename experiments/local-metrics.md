# Shared local evaluation — MNIST64, k=3

This page defines the reusable **local-MNIST64-k3-v1** protocol for all training
and inference workstreams. These are local development metrics, not official
platform scores. Share the same frozen extractor, reference split and image
export convention when comparing methods.

The evaluator is stored once as [protocol metadata](local-metrics-mnist64-k3/summary.json),
[frozen weights](local-metrics-mnist64-k3/extractor.pth), and
[split indices](local-metrics-mnist64-k3/split-indices.npz).
Full source snapshots and feature caches stay under ignored `runs/`.

## Controls and extractor quality

| Control (10,000 vs 10,000) | FD-MNIST ↓ | Precision | Recall |
|---|---:|---:|---:|
| Disjoint real development sets | 0.47 | 89.52% | 90.65% |
| All-black images vs real reference | 1753.13 | 0.00% | 0.00% |

The independent classifier achieves **97.70% validation accuracy**, **97.86% on
the reference set**, and **97.52% on the real control**. Epoch 5 was selected by
validation accuracy from a fixed five-epoch budget, before generated images were
evaluated. The real control's scores need not be 100%: k-NN manifolds are finite
sample estimates. The shared baseline remains far from the real-control scores.

## Frozen protocol: local-MNIST64-k3-v1

- Only the MNIST **training split** is used. A seeded permutation (2026) produces
  four disjoint groups: 35,000 classifier-training, 5,000 classifier-validation,
  10,000 real-reference and 10,000 real-control images. The MNIST test split is
  untouched. Reference and control images were not used to train the feature
  extractor, but **were included in the original GAN training data**. Therefore
  this is a development comparison, not a held-out generalization estimate.
- The extractor is a separate CNN: conv(1→16,3×3,pad1), ReLU, max-pool2;
  conv(16→32,3×3,pad1), ReLU, max-pool2; flatten; linear(1568→64), ReLU;
  linear(64→10). Raw 64-D hidden ReLU activations define the feature space.
  It changes no GAN layers or weights and never trains on generated samples.
- Classifier training uses Adam(lr=0.001), batch size 256, five epochs, seed 2026,
  deterministic CPU operations and two threads. The checkpoint hash, epoch
  losses, split hash and dataset hashes are recorded in the summary.
- Actual exported PNGs are read, checked for replicated grayscale RGB channels,
  converted from uint8 to [0,1], then normalized with `(x-0.5)/0.5`. Real uint8
  images receive the same normalization. Generator outputs are not regenerated
  or rescaled to replace the samples; the original clipping convention remains.
- **FD-MNIST** uses the Gaussian Fréchet distance of feature means and unbiased
  covariances, with float64 eigendecomposition of symmetric PSD matrices. It uses
  the [Fréchet formula](https://arxiv.org/abs/1706.08500) but **is not Inception
  FID**; numbers are specific to this frozen feature extractor.
- **Precision/recall** use [k-NN manifold membership](https://arxiv.org/abs/1904.06991),
  k=3: each feature defines a closed ball whose radius is its third other nearest
  neighbor. Precision is the generated fraction inside the union of real balls;
  recall is the real fraction inside the union of generated balls. Exact pairwise
  squared Euclidean distances are computed in float64 blocks of 512, with self
  neighbors excluded. There is no subsampling or feature whitening.
## Interpretation and limits

Local precision is a feature-space membership fraction, not the percentage of
human-verified correct digits. Near-zero estimated recall does not prove that
true distributional recall is mathematically zero. Classifier counts and
confidence are diagnostics; generated garbage can receive confident labels.

The extractor, reference data and k affect the scores. The controls and ~98%
real-image accuracy are useful checks, not a guarantee of perceptual validity.
The shared baseline covers one training seed. Method claims require repeated
training seeds where feasible. Export clipping can also affect local scores.
Official FID/precision/recall remain null until the official evaluator is used.

## Evaluate any method

Create a JSON manifest, for example `runs/comparison-inputs.json`:

```json
{
  "conditions": [
    {
      "run_id": "ttur-training-seed0-generation-seed1",
      "method": "TTUR",
      "samples": "ttur-seed0/samples",
      "checkpoint": "ttur-seed0/checkpoints/G.pth",
      "checkpoint_sha256": "REPLACE_WITH_ACTUAL_SHA256",
      "config": {
        "training_seed": 0,
        "generation_seed": 1,
        "epochs": 10,
        "lr_G": 0.0002,
        "lr_D": 0.0004,
        "generation_batch_size": 256,
        "sigma": 1.0
      }
    }
  ]
}
```

Paths are relative to the **manifest file**, or may be absolute. `run_id`,
`method`, `samples`, `checkpoint`, `checkpoint_sha256` and a `config` object are
required for each condition. IDs must be unique. Record the actual method
configuration; the learning rates above are an illustration, not a selected
training recommendation. Different conditions may use different checkpoints.
Obtain each checkpoint's hash with `sha256sum path/to/G.pth`.

```bash
python scripts/evaluate_frozen.py --manifest runs/comparison-inputs.json \
  --run-dir runs/comparison-local-metrics
```

Every directory must contain exactly `0.png` through `9999.png`: 28×28 RGB PNGs
with replicated grayscale channels, using the shared starter export convention.
Validation checks manifest structure, generator checkpoint hashes and filenames
before loading the dataset or evaluator. PNG contents are validated when read.
Results retain method, configuration, paths and hashes alongside scores.
Changing only latent scale is not required. No sigma-specific layout is needed.
The existing `--sweep` adapter remains available for paired latent-scale sweeps.
Use either `--manifest` or `--sweep`, never both.

Reuse the frozen weights for comparisons. To reproduce extractor training in a
new directory, rather than silently replacing the shared feature space:

```bash
python scripts/evaluate_local.py --manifest runs/comparison-inputs.json \
  --run-dir runs/rebuilt-local-evaluator --epochs 5 --seed 2026
python -m unittest discover -s tests -v
```

The shared protocol metadata retains the seed-1 baseline control and its image
fingerprint, checked against the original 10-epoch exports. The original full
fit-and-evaluation run remains local under `runs/local-metrics-mnist64-k3/`.
Local metrics never replace the separate official metric fields in ledgers.
