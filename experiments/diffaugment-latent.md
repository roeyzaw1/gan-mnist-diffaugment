# Latent scaling on the shared DiffAugment checkpoint

This study uses the original Juliet/A100 translation+cutout checkpoint shared
in `main` at `166bda6`, not the locally retrained CPU replicas.

- G: `checkpoints/diffaugment-seed0-e10/G.pth`, SHA-256
  `cfefa20e6f28c863f23e19a0521a64fbaf92364341de6b724feeb5ca6bc0522d`.
- Training provenance: seed 0, 10 epochs, original architecture/BCE/Adam;
  full training metadata is shipped with the checkpoint. No retraining or DRS.
- Declared scales: **1, 1.5, 2**, generation seeds **1, 2**, batch 256,
  10,000 exported images per condition. Identical base noise within each seed.
- Export: unchanged starter clipping, RGB PNG 28x28. Generation and frozen
  evaluation run locally on CPU with two threads.
- Metrics: shared `local-MNIST64-k3-v1`; **FD-MNIST is not Inception FID**.
  Official scores remain unavailable. No expensive Inception/VGG job is run.
- The recorded GPU sigma-1/seed-1 FD is 95.9753. We regenerate/evaluate the
  paired sigma-1 control locally rather than substituting that older score.
- Two generation seeds on one G do not constitute independent training repeats.

```bash
python scripts/compare_diffaugment_scaling.py \
  --run-dir runs/shared-diffaugment-latent-seed0-e10 \
  --sigmas 1 1.5 2 --seeds 1 2 --threads 2
```

The runner validates checkpoint/training provenance and source hashes before
creating output, refuses to overwrite an existing run, verifies paired noise,
and records per-seed differences against sigma 1. Full PNGs, snapshots, the
generation ledger, metrics, means and CSV stay in the ignored run directory.

Earlier local CPU studies were preserved privately before integrating main:
`runs/private/before-main-integration-20261101T150026Z/files/experiments/`.
Their full original run directories remain intact.

## Results

Means over generation seeds 1 and 2, on the same trained checkpoint.

| Sigma | FD-MNIST ↓ | Precision ↑ | Recall ↑ |
|---:|---:|---:|---:|
| 1 | 96.9360 | 42.940% | 4.420% |
| 1.5 | 103.0715 | 39.740% | 5.630% |
| 2 | 121.7525 | 36.820% | 8.235% |

| Generation seed | Sigma | FD-MNIST | Precision | Recall |
|---:|---:|---:|---:|---:|
| 1 | 1 | 95.9753 | 43.290% | 4.140% |
| 1 | 1.5 | 103.1765 | 40.170% | 5.020% |
| 1 | 2 | 122.3730 | 36.980% | 7.490% |
| 2 | 1 | 97.8967 | 42.590% | 4.700% |
| 2 | 1.5 | 102.9666 | 39.310% | 6.240% |
| 2 | 2 | 121.1319 | 36.660% | 8.980% |

Both generation seeds show the same trend: increasing sigma from 1 to 1.5
to 2 worsens FD-MNIST and precision while improving recall. On this shared
checkpoint, sigma 1 is the tested FD/precision candidate; sigma 2 is a
recall-oriented trade-off, not a general improvement. Sigma 1.5 is intermediate.
These are local development results on one training seed, not a significance
claim or evidence of the official FID ranking.

The local sigma-1/seed-1 result reproduces the published GPU values to their
reported precision (FD 95.9753, precision 43.29%, recall 4.14%). This confirms
the intended checkpoint/control, without assuming cross-hardware bitwise identity.

Validation: 49 unit tests pass, including DiffAugment gradient/no-op equivalence,
CLI provenance/invalid-scale guards and the full 10,000-PNG starter regression.
The executed study additionally verifies completed sampling/evaluation,
paired-noise hashes, exact output counts and unchanged checkpoint/source hashes.
Per-seed differences vs sigma 1, raw evaluation, source snapshots and CSV
are recorded in the run directory. No default sampler or submission weight changed.
