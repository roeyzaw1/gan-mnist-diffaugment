"""Local MNIST feature metrics; these are not the official platform evaluator."""

import numpy as np
import torch
from torch import nn


class MNISTFeatures(nn.Module):
    """Independent digit classifier. Its 64-D hidden activations define features."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                                  nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.hidden = nn.Linear(32 * 7 * 7, 64)
        self.head = nn.Linear(64, 10)

    def features(self, x):
        return torch.relu(self.hidden(self.conv(x).flatten(1)))

    def forward(self, x):
        return self.head(self.features(x))


def frechet_distance(real, fake):
    """Gaussian feature distance with an eigenvalue-based PSD square root."""
    real, fake = np.asarray(real, dtype=np.float64), np.asarray(fake, dtype=np.float64)
    if real.ndim != 2 or fake.ndim != 2 or real.shape[1] != fake.shape[1] or min(len(real), len(fake)) < 2:
        raise ValueError("Need two feature matrices with matching dimensions and at least two samples")
    if not np.isfinite(real).all() or not np.isfinite(fake).all():
        raise ValueError("Non-finite features")
    cr, cf = np.atleast_2d(np.cov(real, rowvar=False)), np.atleast_2d(np.cov(fake, rowvar=False))
    eigenvalues, eigenvectors = np.linalg.eigh(cr)
    root = (eigenvectors * np.sqrt(eigenvalues.clip(0))) @ eigenvectors.T
    middle = root @ cf @ root
    trace_root = np.sqrt(np.linalg.eigvalsh((middle + middle.T) / 2).clip(0)).sum()
    value = np.square(real.mean(0) - fake.mean(0)).sum() + np.trace(cr) + np.trace(cf) - 2 * trace_root
    if value < -1e-6:
        raise ArithmeticError(f"Negative Frechet distance: {value}")
    return float(max(value, 0))


def squared_distances(a, b):
    return (a.square().sum(1)[:, None] + b.square().sum(1)[None, :] - 2 * a @ b.T).clamp_min_(0)


def knn_radii(features, k=3, chunk_size=512):
    """Squared distance to the kth OTHER neighbor; exact, blockwise float64."""
    features = torch.as_tensor(features, dtype=torch.float64)
    if features.ndim != 2 or not 1 <= k < len(features) or chunk_size < 1:
        raise ValueError("Invalid features, k or chunk size")
    if not torch.isfinite(features).all():
        raise ValueError("Non-finite features")
    radii = []
    for start in range(0, len(features), chunk_size):
        block = features[start:start + chunk_size]
        distance = squared_distances(block, features)
        distance[torch.arange(len(block)), torch.arange(start, start + len(block))] = torch.inf
        radii.append(distance.kthvalue(k, dim=1).values)
    return torch.cat(radii)


def precision_recall(real, fake, k=3, chunk_size=512, real_radii=None):
    """Kynkaanniemi-style manifold PR with closed k-NN balls in feature space."""
    real, fake = torch.as_tensor(real, dtype=torch.float64), torch.as_tensor(fake, dtype=torch.float64)
    if real.ndim != 2 or fake.ndim != 2 or real.shape[1] != fake.shape[1]:
        raise ValueError("Feature dimensions do not match")
    rr = knn_radii(real, k, chunk_size) if real_radii is None else real_radii
    rf = knn_radii(fake, k, chunk_size)
    generated_inside = 0
    covered_real = torch.zeros(len(real), dtype=torch.bool)
    for start in range(0, len(fake), chunk_size):
        block = fake[start:start + chunk_size]
        distance = squared_distances(block, real)
        generated_inside += (distance <= rr[None, :]).any(1).sum().item()
        covered_real |= (distance <= rf[start:start + len(block), None]).any(0)
    return {"precision": generated_inside / len(fake), "recall": covered_real.double().mean().item()}
